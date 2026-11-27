"""Training (GPU required; designed for interruptible cloud sessions). Patient-level splits only; the TEST/EXTERNAL splits are never loaded here.

    python -m braintumor.train --data-root <BraTS folder> --manifest splits/legacy_seed42.json --out <persistent dir>/segresnet_v2 --epochs 150
    python -m braintumor.train ... --benchmark          # time a few steps and estimate the full cost, then exit
    python -m braintumor.train ... --resume             # continue from <out>/last.pth (model, optimiser, scheduler, scaler, history)
    python -m braintumor.train ... --max-hours 3.5      # stop cleanly after the epoch that crosses the budget (session limits); resume later

Cloud-session safety: ``last.pth`` is rewritten atomically every epoch, the best model is versioned under ``versions/``, every epoch appends one
line to ``metrics.jsonl``, and ``--cache-dir`` lets the (large, rebuildable) MONAI cache live on fast local disk while ``--out`` sits on persistent storage.

Differences from the notebook (each justified; none has been ablated yet - see REPORT.md):
  * Dice + BCE loss (also dicefocal / gdice): the notebook's Dice-only loss (smooth_nr=0) has *zero gradient* for a region absent from the patch.
  * Left-right flips only + mild affine/noise instead of flips on all three axes (A-P / S-I flips produce impossible anatomy).
  * Linear warm-up + cosine schedule over the *real* number of epochs (the notebook stopped at 30 epochs with loss still falling).
  * Validation Dice uses the BraTS empty-region convention (the notebook's MONAI DiceMetric silently dropped empty-GT cases).
  * Micro-batch / gradient accumulation adapt to the GPU actually present; atomic checkpoints, resume, early stopping, run-directory protection.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from . import REGIONS, __version__
from .inference import pick_device
from .model import build_model, save_checkpoint, update_card
from .splits import load_manifest

log = logging.getLogger("braintumor.train")
LOSSES = ("dice", "dicece", "dicefocal", "gdice")


@dataclass
class TrainConfig:
    data_root: str = ""                 # BraTS-2020-style root (back-compat); use data_roots for several datasets
    data_roots: dict | None = None      # {registry dataset name: path}
    manifest: str = "splits/legacy_seed42.json"
    out_dir: str = "runs/segresnet"
    cache_dir: str = ""                 # default: <out_dir>/cache (set to local disk when out_dir is on network storage)
    arch: str = "segresnet"
    epochs: int = 150
    warmup_epochs: int = 3
    lr: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 2                 # EFFECTIVE batch (micro-batch x accum_steps)
    accum_steps: int = 1
    auto_batch: bool = True             # pick micro-batch from free VRAM, raise accum_steps to keep the effective batch
    roi: tuple = (128, 128, 128)
    val_interval: int = 5
    patience: int = 12                  # validation checks without improvement before early stopping
    min_delta: float = 1e-3
    loss: str = "dicece"                # dice (notebook) | dicece | dicefocal | gdice
    amp: bool = True
    num_workers: int = 2
    cache: bool = True
    seed: int = 42
    deterministic: bool = False
    device: str = "auto"
    flip_axes: tuple = (0,)
    sw_overlap: float = 0.25
    max_hours: float = 0.0              # 0 = no wall-clock budget
    max_train_cases: int = 0            # 0 = all (smoke tests only)
    max_val_cases: int = 0
    allow_cpu: bool = False

    @staticmethod
    def load(path: str | None, **overrides) -> "TrainConfig":
        d = yaml.safe_load(open(path)) if path else {}
        d.update({k: v for k, v in overrides.items() if v is not None})
        for k in ("roi", "flip_axes"):
            if k in d:
                d[k] = tuple(d[k])
        return TrainConfig(**d)


class CombinedLoss(torch.nn.Module):
    """Squared-Dice (sigmoid, multi-label) plus an optional pixel-wise term (BCE / focal) or Generalized Dice with BCE."""

    def __init__(self, kind: str = "dicece"):
        super().__init__()
        from monai.losses import DiceLoss, FocalLoss, GeneralizedDiceLoss
        if kind not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES}")
        self.kind = kind
        self.dice = GeneralizedDiceLoss(sigmoid=True, to_onehot_y=False) if kind == "gdice" else \
            DiceLoss(to_onehot_y=False, sigmoid=True, squared_pred=True, smooth_nr=0, smooth_dr=1e-5)
        self.pix = {"dice": None, "dicece": torch.nn.BCEWithLogitsLoss(), "gdice": torch.nn.BCEWithLogitsLoss(),
                    "dicefocal": FocalLoss(to_onehot_y=False, gamma=2.0, use_softmax=False)}[kind]

    def forward(self, logits, target):
        loss = self.dice(logits, target)
        return loss + self.pix(logits, target) if self.pix is not None else loss


def batch_dice(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """pred, gt: bool (3, X, Y, Z) -> Dice per region with the BraTS convention (both empty = 1)."""
    p, g = pred.flatten(1).float(), gt.flatten(1).float()
    inter, s = (p * g).sum(1), p.sum(1) + g.sum(1)
    return torch.where(s == 0, torch.ones_like(s), 2 * inter / s.clamp_min(1))


def adapt_batch(cfg: TrainConfig, device: torch.device) -> dict:
    """Choose micro-batch / accumulation for the GPU that is actually present (CPU: untouched). Effective batch is preserved."""
    info = {"device": str(device)}
    if device.type != "cuda":
        return info
    props = torch.cuda.get_device_properties(device)
    free, total = (x / 2 ** 30 for x in torch.cuda.mem_get_info(device))
    info.update(gpu=props.name, vram_total_gb=round(total, 1), vram_free_gb=round(free, 1), compute_capability=f"{props.major}.{props.minor}")
    eff = cfg.batch_size * cfg.accum_steps
    if cfg.auto_batch:
        micro = 4 if free >= 20 else 2 if free >= 11 else 1       # 128^3 SegResNet with AMP: ~7 GB/sample on a T4 in the notebook (batch 2 fit 15 GB)
        micro = min(micro, eff)
        cfg.batch_size, cfg.accum_steps = micro, math.ceil(eff / micro)
        info["adapted"] = f"micro-batch {cfg.batch_size} x accumulation {cfg.accum_steps} (effective {cfg.batch_size * cfg.accum_steps})"
    return info


def estimate_cost(cfg: TrainConfig, device: torch.device, n_train: int, n_val: int, steps: int = 3) -> dict:
    """Time a few optimisation steps on random tensors (speed only - no data needed) and extrapolate."""
    model = build_model(cfg.arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), cfg.lr)
    loss_fn = CombinedLoss(cfg.loss)
    x = torch.randn(cfg.batch_size, 4, *cfg.roi, device=device)
    y = (torch.rand(cfg.batch_size, 3, *cfg.roi, device=device) > 0.9).float()
    use_amp = cfg.amp and device.type == "cuda"
    times = []
    for i in range(steps + 1):
        t = time.time()
        with torch.autocast(device.type, enabled=use_amp):
            loss = loss_fn(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        if i:  # skip warm-up step
            times.append(time.time() - t)
    step_s = float(np.mean(times))
    n_steps = int(np.ceil(n_train / cfg.batch_size))
    train_epoch_s = step_s * n_steps
    model.eval()
    with torch.inference_mode():
        t = time.time()
        model(x[:1])
        win_s = time.time() - t
    val_s = win_s * 8 * n_val                               # ~8 windows / patient at 0.25 overlap on a 160x190x140 crop
    n_checks = cfg.epochs // cfg.val_interval
    total_h = (train_epoch_s * cfg.epochs + val_s * n_checks) / 3600
    return {"device": str(device), "seconds_per_step": round(step_s, 2), "seconds_per_epoch": round(train_epoch_s, 1),
            "validation_seconds_per_check": round(val_s, 1), "estimated_total_hours": float(f"{total_h:.3g}")}


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def plot_history(hist: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(hist["epoch_train"], hist["train_loss"])
    ax[0].set(title="Training loss", xlabel="epoch")
    for r in REGIONS:
        ax[1].plot(hist["epoch_val"], hist[f"val_{r}"], marker="o", label=r)
    ax[1].plot(hist["epoch_val"], hist["val_mean"], "k--", label="mean")
    ax[1].set(title="Validation Dice (val patients, BraTS empty convention)", xlabel="epoch", ylim=(0, 1))
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run(cfg: TrainConfig, resume: bool = False, benchmark: bool = False) -> dict:
    from monai.data import DataLoader, Dataset, PersistentDataset
    from monai.inferers import sliding_window_inference

    from .data import build_transforms, discover_many, select
    from .datasets import load_registry

    device = pick_device(cfg.device)
    hw = adapt_batch(cfg, device)
    man = load_manifest(cfg.manifest)
    train_ids, val_ids = man["splits"]["train"], man["splits"]["val"]       # NOTE: 'test', 'test_legacy' and 'external' are deliberately never read here
    if cfg.max_train_cases:
        train_ids = train_ids[: cfg.max_train_cases]
    if cfg.max_val_cases:
        val_ids = val_ids[: cfg.max_val_cases]

    est = estimate_cost(cfg, device, len(train_ids), len(val_ids))
    print("Hardware:", json.dumps(hw))
    print("Cost estimate:", json.dumps(est))
    if benchmark:
        return {"hardware": hw, "estimate": est}
    if device.type == "cpu" and not cfg.allow_cpu:
        raise SystemExit(f"Refusing to train on CPU: ~{est['seconds_per_epoch'] / 3600:.1f} h per epoch, ~{est['estimated_total_hours']:.0f} h in total "
                         f"for {cfg.epochs} epochs. Use a cloud GPU (see USER_GUIDE.md) or pass --allow-cpu for a smoke test.")

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    last_p, best_p = out / "last.pth", out / "best.pth"
    if last_p.exists() and not resume:
        raise FileExistsError(f"{last_p} exists: pass --resume to continue, or choose a new --out (existing runs are never overwritten)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(out / "train.log")])
    cfg_dict = {k: (list(v) if isinstance(v, tuple) else v) for k, v in dataclasses.asdict(cfg).items()}
    (out / "config.yaml").write_text(yaml.safe_dump(cfg_dict))
    (out / "environment.json").write_text(json.dumps({
        "braintumor": __version__, "git_commit": _git_commit(), "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "hardware": hw, "platform": platform.platform(), "monai": __import__("monai").__version__, "manifest_sha256": man["file_sha256"],
        "manifest_method": man["method"], "n_train": len(train_ids), "n_val": len(val_ids), "cost_estimate": est,
        "config_sha256": hashlib.sha256(json.dumps(cfg_dict, sort_keys=True, default=str).encode()).hexdigest()}, indent=1))

    reg = load_registry()
    roots = cfg.data_roots or {"brats2020": cfg.data_root}
    cases, skipped = discover_many(roots, reg)
    train_files, val_files = select(cases, train_ids), select(cases, val_ids)
    if cfg.deterministic:
        from monai.utils import set_determinism
        set_determinism(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    cache = Path(cfg.cache_dir) if cfg.cache_dir else out / "cache"
    tr_t, va_t = build_transforms(True, cfg.roi, cfg.flip_axes), build_transforms(False)
    train_ds = PersistentDataset(train_files, tr_t, cache_dir=cache / "train") if cfg.cache else Dataset(train_files, tr_t)
    val_ds = PersistentDataset(val_files, va_t, cache_dir=cache / "val") if cfg.cache else Dataset(val_files, va_t)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=min(cfg.num_workers, 2))

    model = build_model(cfg.arch).to(device)
    loss_fn = CombinedLoss(cfg.loss).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warm = max(cfg.warmup_epochs, 0)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [torch.optim.lr_scheduler.LinearLR(opt, 0.1, 1.0, total_iters=max(warm, 1)),
                                                        torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(cfg.epochs - warm, 1), 1e-6)], [warm])
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    hist = {"epoch_train": [], "train_loss": [], "epoch_val": [], "val_mean": [], **{f"val_{r}": [] for r in REGIONS}}
    start, best, bad_checks = 1, -1.0, 0
    if resume and last_p.exists():
        ck = torch.load(last_p, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start, best, bad_checks, hist = ck["epoch"] + 1, ck["best"], ck["bad_checks"], ck["history"]
        log.info("resumed from epoch %d (best val mean Dice %.4f)", ck["epoch"], best)

    t_start = time.time()
    budget_s = cfg.max_hours * 3600 if cfg.max_hours else None
    paused = False
    epoch = start - 1
    for epoch in range(start, cfg.epochs + 1):
        t_ep = time.time()
        g = torch.Generator().manual_seed(cfg.seed + epoch)                  # per-epoch seed -> resume-consistent shuffling
        if hasattr(train_ds, "set_random_state"):
            train_ds.set_random_state(cfg.seed + epoch)
        loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, generator=g, drop_last=False)
        model.train()
        run_loss, n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, b in enumerate(loader, 1):
            x, y = b["image"].to(device, non_blocking=True), b["label"].to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=use_amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss / cfg.accum_steps).backward()
            if i % cfg.accum_steps == 0 or i == len(loader):                 # accumulate, then step (also flush the last partial group)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            run_loss += float(loss.item())
            n += 1
        sched.step()
        hist["epoch_train"].append(epoch)
        hist["train_loss"].append(run_loss / max(n, 1))
        rec = {"epoch": epoch, "train_loss": run_loss / max(n, 1), "lr": opt.param_groups[0]["lr"], "epoch_seconds": round(time.time() - t_ep, 1),
               "peak_gpu_gb": round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 2) if device.type == "cuda" else None, "val": None}
        log.info("epoch %d/%d loss %.4f lr %.2e (%.0fs elapsed)", epoch, cfg.epochs, rec["train_loss"], rec["lr"], time.time() - t_start)

        if epoch % cfg.val_interval == 0 or epoch == cfg.epochs:
            model.eval()
            scores = []
            with torch.inference_mode():
                for vb in val_loader:
                    vx, vy = vb["image"].to(device), vb["label"].to(device)
                    with torch.autocast(device.type, enabled=use_amp):
                        logit = sliding_window_inference(vx, roi_size=cfg.roi, sw_batch_size=2, predictor=model, overlap=cfg.sw_overlap)
                    scores.append(batch_dice(torch.sigmoid(logit.float())[0] >= 0.5, vy[0] > 0.5).cpu().numpy())
            sc = np.mean(scores, 0)
            mean = float(sc.mean())
            hist["epoch_val"].append(epoch)
            hist["val_mean"].append(mean)
            for i, r in enumerate(REGIONS):
                hist[f"val_{r}"].append(float(sc[i]))
            rec["val"] = {"mean": mean, **{r: float(sc[i]) for i, r in enumerate(REGIONS)}, "n": len(scores)}
            log.info("  [VAL n=%d] mean %.4f | TC %.4f WT %.4f ET %.4f", len(scores), mean, *sc)
            if mean > best + cfg.min_delta:
                best, bad_checks = mean, 0
                meta = {"epoch": epoch, "val_mean_dice": mean, "val_dice": dict(zip(REGIONS, map(float, sc))), "run": str(out)}
                save_checkpoint(best_p, model, cfg.arch, meta)
                save_checkpoint(out / "versions" / f"best_e{epoch:03d}.pth", model, cfg.arch, meta)       # versioned copy
                update_card(best_p, trained_by=f"braintumor {__version__}", git_commit=_git_commit(), run_dir=str(out), epoch=epoch, val_mean_dice=mean,
                            manifest_sha256=man["file_sha256"], manifest_method=man["method"], config=cfg_dict, datasets=sorted(roots), evaluation=None,
                            note="Validation Dice was used for checkpoint selection; it is NOT a test result. Run braintumor.evaluate on the held-out splits.")
                log.info("  new best saved (epoch %d)", epoch)
            else:
                bad_checks += 1
            plot_history(hist, out / "history.png")
        _atomic_save({"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                      "best": best, "bad_checks": bad_checks, "history": hist}, last_p)
        (out / "history.json").write_text(json.dumps(hist, indent=1))
        with open(out / "metrics.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        if bad_checks >= cfg.patience:
            log.info("early stopping: no validation improvement for %d checks", bad_checks)
            break
        if budget_s and time.time() - t_start >= budget_s and epoch < cfg.epochs:
            paused = True
            log.info("time budget (%.1f h) reached after epoch %d: stopping cleanly - resume with --resume", cfg.max_hours, epoch)
            break
    finished = (epoch >= cfg.epochs or bad_checks >= cfg.patience) and not paused
    status = {"finished": bool(finished), "paused_for_time_budget": paused, "last_epoch": epoch, "best_val_mean_dice": best,
              "best_checkpoint": str(best_p) if best_p.exists() else None}
    (out / "status.json").write_text(json.dumps(status, indent=1))
    return status


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--data-root")
    ap.add_argument("--data-roots", nargs="*", metavar="NAME=PATH", help="several datasets, e.g. brats2020=/data/b20 brats2021=/data/b21")
    ap.add_argument("--manifest")
    ap.add_argument("--out", dest="out_dir")
    ap.add_argument("--cache-dir")
    ap.add_argument("--arch", choices=["segresnet", "unet"])
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--accum-steps", type=int)
    ap.add_argument("--num-workers", type=int)
    ap.add_argument("--device")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--loss", choices=list(LOSSES))
    ap.add_argument("--max-hours", type=float)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--benchmark", action="store_true", help="estimate cost and exit")
    ap.add_argument("--allow-cpu", action="store_true")
    a = ap.parse_args(argv)
    roots = dict(s.split("=", 1) for s in a.data_roots) if a.data_roots else None
    cfg = TrainConfig.load(a.config, **{k: getattr(a, k) for k in ("data_root", "manifest", "out_dir", "cache_dir", "arch", "epochs", "batch_size", "accum_steps",
                                                                  "num_workers", "device", "seed", "loss", "max_hours")},
                           data_roots=roots, allow_cpu=True if a.allow_cpu else None)
    print(json.dumps(run(cfg, resume=a.resume, benchmark=a.benchmark), indent=1))


if __name__ == "__main__":
    main()

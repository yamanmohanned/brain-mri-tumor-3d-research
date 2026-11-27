#!/usr/bin/env python
"""Stage runner for cloud GPU sessions (Colab / Kaggle / any Linux GPU VM). Thin wrapper around the braintumor package.

Stages (each is safe to re-run; long ones resume):
  check-gpu        report the GPU; with --require exit 3 when none is usable (training never silently falls back to CPU)
  get-brats2020    download the BraTS 2020 training data INTO THE CLOUD VM (kagglehub; resumable) and verify its layout
  audit            dataset integrity audit (+ legacy-split consistency check)
  eval             held-out evaluation of a checkpoint (val / test / test_legacy / external)
  train            training with automatic resume, wall-clock budget and persistent run dir
  infer3d          end-to-end 3D inference test on one case: segmentation, measurements, meshes, NIfTI/HTML export

Examples (see notebooks/cloud_gpu_pipeline.ipynb):
  python scripts/cloud_run.py check-gpu --require
  python scripts/cloud_run.py train --runs-dir /content/drive/MyDrive/bt/runs --run-name segresnet_v2 --data-root /content/data --max-hours 3.5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import braintumor._preload  # noqa: E402,F401  (torch before Qt-free imports; harmless here)


def gpu_info() -> dict:
    import torch
    info = {"torch": torch.__version__, "cuda_build": torch.version.cuda, "cuda": torch.cuda.is_available()}
    if info["cuda"]:
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, vram_gb=round(p.total_memory / 2 ** 30, 1), capability=f"{p.major}.{p.minor}", n_gpus=torch.cuda.device_count(),
                    arch_supported=f"sm_{p.major}{p.minor}" in torch.cuda.get_arch_list())
    return info


def cmd_check_gpu(a) -> int:
    info = gpu_info()
    print(json.dumps(info, indent=1))
    if a.require and not (info["cuda"] and info.get("arch_supported", False)):
        print("NO USABLE CUDA GPU: training is refused. In Colab choose Runtime > Change runtime type > GPU; in Kaggle enable a GPU accelerator.", file=sys.stderr)
        return 3
    return 0


def verify_brats2020(root: str | Path) -> dict:
    """Layout/count verification of a BraTS 2020 training download (does not read voxel data)."""
    from braintumor.data import discover_cases, locate_root
    from braintumor.datasets import get_spec
    spec = get_spec("brats2020")
    r = locate_root(root, spec)
    cases, skipped = discover_cases(r, spec)
    listing = sorted(p.name for p in r.iterdir() if p.is_dir())
    return {"root": str(r), "n_folders": len(listing), "n_complete": len(cases), "skipped": [{"id": i, "reason": w} for i, w in skipped],
            "folder_listing_sha256": hashlib.sha256("\n".join(listing).encode()).hexdigest(),
            "expected": "369 folders, 368 complete (BraTS20_Training_355 lacks a standard *_seg file)", "matches_expected": (len(listing), len(cases)) == (369, 368)}


def cmd_get_brats2020(a) -> int:
    if a.root:
        info = verify_brats2020(a.root)
    else:
        try:
            import kagglehub
        except ImportError:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kagglehub"])
            import kagglehub
        path = kagglehub.dataset_download("awsaf49/brats20-dataset-training-validation")      # public dataset; stays in the cloud VM
        info = verify_brats2020(Path(path) / "BraTS2020_TrainingData")
        info["source"] = "kaggle:awsaf49/brats20-dataset-training-validation"
    print(json.dumps(info, indent=1))
    if a.record:
        Path(a.record).parent.mkdir(parents=True, exist_ok=True)
        Path(a.record).write_text(json.dumps(info, indent=1))
    return 0 if info["matches_expected"] else 4


def cmd_audit(a) -> int:
    from braintumor.__main__ import main
    argv = ["audit", "--data-root", a.data_root, "--out", a.out, "--dataset", a.dataset]
    if a.manifest:
        argv += ["--manifest", a.manifest]
    main(argv)
    return 0


def cmd_eval(a) -> int:
    from braintumor.evaluate import main
    argv = ["--checkpoint", a.checkpoint, "--data-root", a.data_root, "--manifest", a.manifest, "--split", a.split, "--out", a.out]
    for flag, on in (("--tta", a.tta), ("--no-card-update", a.no_card_update), ("--assert-trained-on-manifest", a.assert_trained_on_manifest)):
        if on:
            argv.append(flag)
    if a.limit:
        argv += ["--limit", str(a.limit)]
    main(argv)
    return 0


def cmd_train(a) -> int:
    import yaml
    from braintumor.train import TrainConfig, run
    rc = cmd_check_gpu(argparse.Namespace(require=not a.allow_cpu))
    if rc:
        return rc
    cfg_d = yaml.safe_load(open(a.config)) if a.config else {}
    for kv in a.set or []:
        k, _, v = kv.partition("=")
        cfg_d[k] = yaml.safe_load(v)
    out = Path(a.runs_dir) / a.run_name
    cfg_d.update(data_root=a.data_root, manifest=a.manifest, out_dir=str(out), max_hours=a.max_hours, allow_cpu=a.allow_cpu)
    if a.cache_dir:
        cfg_d["cache_dir"] = a.cache_dir
    for k in ("roi", "flip_axes"):
        if k in cfg_d:
            cfg_d[k] = tuple(cfg_d[k])
    cfg = TrainConfig(**cfg_d)
    resume = (out / "last.pth").exists()
    print(f"{'RESUMING' if resume else 'STARTING'} run {a.run_name} in {out}")
    status = run(cfg, resume=resume)
    print(json.dumps(status, indent=1))
    if status.get("paused_for_time_budget"):
        print("Paused for the time budget. Re-run this stage (same arguments) to continue from the last epoch.")
    return 0


def cmd_infer3d(a) -> int:
    from braintumor.__main__ import main
    from braintumor.data import discover_cases
    cases, _ = discover_cases(a.data_root)
    cid = a.case_id or sorted(cases)[0]
    f = cases[cid]["files"]
    main(["analyze", "--t1", f["t1"], "--t1ce", f["t1ce"], "--t2", f["t2"], "--flair", f["flair"], "--checkpoint", a.checkpoint, "--out", a.out,
          "--case-id", cid, "--device", "cuda" if gpu_info()["cuda"] else "cpu", "--overwrite"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("check-gpu")
    p.add_argument("--require", action="store_true")
    p.set_defaults(fn=cmd_check_gpu)
    p = sub.add_parser("get-brats2020")
    p.add_argument("--root", help="verify an existing folder instead of downloading")
    p.add_argument("--record", help="write the verification record (json) here")
    p.set_defaults(fn=cmd_get_brats2020)
    p = sub.add_parser("audit")
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--manifest")
    p.add_argument("--dataset", default="brats2020")
    p.set_defaults(fn=cmd_audit)
    p = sub.add_parser("eval")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--tta", action="store_true")
    p.add_argument("--no-card-update", action="store_true")
    p.add_argument("--assert-trained-on-manifest", action="store_true")
    p.set_defaults(fn=cmd_eval)
    p = sub.add_parser("train")
    p.add_argument("--runs-dir", required=True, help="PERSISTENT storage (e.g. a Google Drive folder)")
    p.add_argument("--run-name", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--manifest", default="splits/legacy_seed42.json")
    p.add_argument("--config", default=str(ROOT / "configs" / "train.yaml"))
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config value (YAML syntax), repeatable")
    p.add_argument("--max-hours", type=float, default=0.0)
    p.add_argument("--cache-dir", default="")
    p.add_argument("--allow-cpu", action="store_true", help="smoke tests only")
    p.set_defaults(fn=cmd_train)
    p = sub.add_parser("infer3d")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--case-id")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_infer3d)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return int(a.fn(a) or 0)


if __name__ == "__main__":
    sys.exit(main())

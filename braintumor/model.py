"""Network definitions and checkpoint handling.

The architectures are *identical* to the original notebook (MONAI SegResNet / 3D U-Net, 4 input
channels [T1,T1ce,T2,FLAIR], 3 sigmoid outputs [TC,WT,ET]) so checkpoints trained there load unchanged.
Checkpoints are loaded with ``weights_only=True`` (no arbitrary pickle execution).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

from . import MODALITIES, REGIONS

ARCHS: dict[str, dict] = {
    "segresnet": dict(spatial_dims=3, init_filters=16, in_channels=4, out_channels=3,
                      blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1), dropout_prob=0.2),
    "unet": dict(spatial_dims=3, in_channels=4, out_channels=3, channels=(16, 32, 64, 128, 256),
                 strides=(2, 2, 2, 2), num_res_units=2),
}


class CheckpointError(ValueError):
    """Checkpoint missing, unreadable or incompatible (message is user-facing)."""


def build_model(arch: str, **overrides) -> torch.nn.Module:
    from monai.networks.nets import SegResNet, UNet
    if arch not in ARCHS:
        raise CheckpointError(f"Unknown architecture {arch!r}; choose from {sorted(ARCHS)}")
    kw = {**ARCHS[arch], **overrides}
    return (SegResNet if arch == "segresnet" else UNet)(**kw)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def card_path(ckpt: str | Path) -> Path:
    p = Path(ckpt)
    return p.with_name(p.stem + ".card.json")


def read_card(ckpt: str | Path) -> dict:
    p = card_path(ckpt)
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"error": "card file is corrupted"}
    return {}


def update_card(ckpt: str | Path, **fields) -> dict:
    card = read_card(ckpt)
    card.update(fields)
    _atomic_write_text(card_path(ckpt), json.dumps(card, indent=2, default=str))
    return card


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def save_checkpoint(path: str | Path, model: torch.nn.Module, arch: str, meta: dict | None = None) -> Path:
    """Atomically write a self-describing inference checkpoint (never overwrites silently - callers choose the path)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "braintumor_v1", "arch": arch, "arch_kwargs": {k: (list(v) if isinstance(v, tuple) else v) for k, v in ARCHS[arch].items()},
        "modalities": list(MODALITIES), "regions": list(REGIONS),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "meta": json.loads(json.dumps(meta or {}, default=str)),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def _extract_state(obj) -> tuple[dict, str, dict]:
    """Return (state_dict, format, meta) for the checkpoint layouts we know."""
    if isinstance(obj, dict) and obj.get("format") == "braintumor_v1":
        return obj["state_dict"], "braintumor_v1", obj.get("meta", {}) | {"arch": obj.get("arch")}
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):  # notebook *_last.pth
        meta = {k: obj[k] for k in ("epoch", "best_dice") if k in obj}
        return obj["model"], "legacy_training_checkpoint", meta
    if isinstance(obj, dict) and obj and all(isinstance(v, torch.Tensor) for v in obj.values()):  # notebook *_best.pth
        return obj, "legacy_state_dict", {}
    raise CheckpointError("Unrecognised checkpoint layout (expected a state_dict or a braintumor/notebook checkpoint)")


def load_checkpoint(path: str | Path, arch: str | None = None, device: str | torch.device = "cpu"):
    """Load weights into the matching architecture. Returns ``(model, info)``; raises ``CheckpointError``."""
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"Checkpoint not found: {path}")
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CheckpointError(f"Cannot read checkpoint {path.name}: {exc}") from exc
    state, fmt, meta = _extract_state(obj)
    candidates = [arch or meta.get("arch")] if (arch or meta.get("arch")) else list(ARCHS)
    errors = {}
    for name in candidates:
        try:
            model = build_model(name)
            model.load_state_dict(state, strict=True)
            break
        except Exception as exc:
            errors[name] = str(exc).splitlines()[0][:200]
    else:
        raise CheckpointError(f"Weights do not match architecture(s) {list(errors)}: {errors}")
    model.to(device).eval()
    info = {
        "path": str(path), "sha256": sha256_file(path), "arch": name, "format": fmt,
        "n_params": int(sum(p.numel() for p in model.parameters())), "meta": meta, "card": read_card(path),
    }
    return model, info


def find_checkpoints(weights_dir: str | Path) -> list[Path]:
    d = Path(weights_dir)
    return sorted(p for p in d.glob("*.pth") if not p.name.endswith(".tmp")) if d.is_dir() else []

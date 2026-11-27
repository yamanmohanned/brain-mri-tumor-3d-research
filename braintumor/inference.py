"""Sliding-window inference with progress, cancellation and optional flip-TTA uncertainty."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from . import REGIONS

# Flip sets for test-time augmentation (axes of the (X,Y,Z) volume). Left-right flip is anatomically plausible;
# the others are used only as perturbations to measure prediction disagreement.
TTA_SETS = {"none": [()], "flips4": [(), (0,), (1,), (2,)]}


class Cancelled(Exception):
    """Raised inside inference when the caller asked to stop."""


@dataclass
class InferenceConfig:
    roi: tuple = (128, 128, 128)
    overlap: float = 0.5
    sw_batch_size: int = 1
    threshold: float = 0.5
    mode: str = "constant"        # notebook used MONAI's default ('constant')
    tta: str = "none"
    device: str = "auto"
    threads: int = 0              # 0 = torch default


def pick_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(pref)


def n_windows(shape, roi, overlap) -> int:
    tot = 1
    for s, r in zip(shape, roi):
        if s <= r:
            continue
        interval = max(int(r * (1 - overlap)), 1)
        tot *= math.ceil((s - r) / interval) + 1
    return tot


@dataclass
class Prediction:
    probs: np.ndarray                      # (3, X', Y', Z') sigmoid probabilities, channel order = REGIONS
    tta_std: np.ndarray | None = None      # (3, X', Y', Z') std over TTA passes
    passes: int = 1
    notes: list = field(default_factory=list)


def predict_probabilities(model: torch.nn.Module, tensor: np.ndarray, cfg: InferenceConfig,
                          progress: Callable[[float, str], None] | None = None,
                          cancel: Callable[[], bool] | None = None) -> Prediction:
    from monai.inferers import sliding_window_inference
    device = pick_device(cfg.device)
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    model = model.to(device).eval()
    x = torch.from_numpy(np.ascontiguousarray(tensor, dtype=np.float32))[None].to(device)
    flips = TTA_SETS[cfg.tta]
    per_pass = n_windows(x.shape[2:], cfg.roi, cfg.overlap)
    total = per_pass * len(flips)
    done = 0

    def predictor(w):
        nonlocal done
        if cancel and cancel():
            raise Cancelled()
        out = model(w)
        done += w.shape[0]
        if progress:
            progress(min(done / total, 1.0), f"window {min(done, total)}/{total}")
        return out

    acc, acc2 = None, None
    with torch.inference_mode():
        for axes in flips:
            xi = torch.flip(x, dims=[a + 2 for a in axes]) if axes else x
            logits = sliding_window_inference(xi, roi_size=cfg.roi, sw_batch_size=cfg.sw_batch_size, predictor=predictor,
                                              overlap=cfg.overlap, mode=cfg.mode)
            p = torch.sigmoid(logits.float())
            if axes:
                p = torch.flip(p, dims=[a + 2 for a in axes])
            p = p[0].cpu().numpy()
            acc = p if acc is None else acc + p
            acc2 = p ** 2 if acc2 is None else acc2 + p ** 2
    n = len(flips)
    mean = acc / n
    std = np.sqrt(np.maximum(acc2 / n - mean ** 2, 0)) if n > 1 else None
    return Prediction(mean.astype(np.float32), None if std is None else std.astype(np.float32), n)


def threshold_masks(probs: np.ndarray, threshold: float = 0.5) -> dict[str, np.ndarray]:
    return {r: probs[i] >= threshold for i, r in enumerate(REGIONS)}

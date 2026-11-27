"""Model-input preprocessing: working grid, foreground crop, per-channel z-score.

This mirrors the training pipeline of the original notebook (Orientationd RAS -> CropForegroundd ->
NormalizeIntensityd(nonzero, channel_wise)) so inference sees exactly what training saw.
``tests/test_preprocess.py`` checks numerical equality against the MONAI transforms.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import GridTransform, make_working_grid
from .io import MRVolume


@dataclass
class PreparedInput:
    tensor: np.ndarray            # (C, X', Y', Z') normalised model input (cropped)
    raw_work: np.ndarray          # (C, X, Y, Z) raw intensities on the working grid (display / brain shell)
    grid: GridTransform
    crop: tuple                   # 3 slices into the working grid

    @property
    def work_shape(self) -> tuple:
        return self.grid.work_shape

    @property
    def work_affine(self) -> np.ndarray:
        return self.grid.work_affine


def foreground_slices(img: np.ndarray, margin: int = 0) -> tuple:
    """Bounding box of voxels where ANY channel > 0 (MONAI CropForeground default)."""
    fg = (img > 0).any(axis=0)
    if not fg.any():
        return tuple(slice(0, s) for s in fg.shape)
    sl = []
    for ax in range(3):
        other = tuple(a for a in range(3) if a != ax)
        idx = np.flatnonzero(fg.any(axis=other))
        sl.append(slice(max(int(idx[0]) - margin, 0), min(int(idx[-1]) + 1 + margin, fg.shape[ax])))
    return tuple(sl)


def normalize_nonzero(img: np.ndarray) -> np.ndarray:
    """Channel-wise z-score over non-zero voxels; zeros stay zero (== NormalizeIntensity(nonzero=True))."""
    out = img.astype(np.float32, copy=True)
    for c in range(out.shape[0]):
        m = out[c] != 0
        if not m.any():
            continue
        v = out[c][m]
        sd = float(v.std())
        out[c][m] = (v - v.mean()) / (sd if sd > 0 else 1.0)
    return out


def prepare(vol: MRVolume, spacing: float = 1.0) -> PreparedInput:
    grid = make_working_grid(vol.affine, vol.shape, spacing)
    raw = grid.to_work(vol.data, order=1).astype(np.float32)
    crop = foreground_slices(raw)
    tensor = normalize_nonzero(raw[(slice(None), *crop)])
    return PreparedInput(tensor, raw, grid, crop)


def embed(arr: np.ndarray, prep: PreparedInput, fill=0) -> np.ndarray:
    """Place a cropped-grid array (..., X', Y', Z') back onto the full working grid."""
    out = np.full((*arr.shape[:-3], *prep.work_shape), fill, dtype=arr.dtype)
    out[(..., *prep.crop)] = arr
    return out

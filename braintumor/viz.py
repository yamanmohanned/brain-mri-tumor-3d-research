"""Display helpers shared by the GUI and exported images.

Planes are shown in radiological convention on the RAS working grid: axial/coronal have the patient's RIGHT on the
screen LEFT; sagittal has anterior on the screen left; superior is always up.
"""
from __future__ import annotations

import numpy as np

PLANES = ("axial", "coronal", "sagittal")
PLANE_AXIS = {"axial": 2, "coronal": 1, "sagittal": 0}
REGION_COLORS = {"WT": (0.20, 0.80, 0.30), "TC": (0.95, 0.25, 0.25), "ET": (1.00, 0.85, 0.10)}   # same as the notebook
SIDE_LABELS = {"axial": ("R", "L", "A", "P"), "coronal": ("R", "L", "S", "I"), "sagittal": ("A", "P", "S", "I")}  # left,right,top,bottom


def extract_plane(arr: np.ndarray, plane: str, idx: int) -> np.ndarray:
    """2D image (rows x cols) for ``arr[x,y,z]`` in radiological display orientation."""
    ax = PLANE_AXIS[plane]
    sl = np.take(arr, int(np.clip(idx, 0, arr.shape[ax] - 1)), axis=ax)
    return np.ascontiguousarray(sl.T[::-1, ::-1])


def pixel_to_index(plane: str, shape: tuple, row: int, col: int) -> tuple[int, int]:
    """Image pixel -> the two array indices (a, b) of the plane's remaining axes."""
    axes = [a for a in range(3) if a != PLANE_AXIS[plane]]
    A, B = shape[axes[0]], shape[axes[1]]
    return A - 1 - col, B - 1 - row


def set_cursor(cursor: list, plane: str, row: int, col: int, shape: tuple) -> list:
    axes = [a for a in range(3) if a != PLANE_AXIS[plane]]
    a, b = pixel_to_index(plane, shape, row, col)
    out = list(cursor)
    out[axes[0]], out[axes[1]] = int(np.clip(a, 0, shape[axes[0]] - 1)), int(np.clip(b, 0, shape[axes[1]] - 1))
    return out


def cursor_pixel(plane: str, shape: tuple, cursor) -> tuple[int, int]:
    """Array cursor -> (row, col) of the crosshair in the displayed image."""
    axes = [a for a in range(3) if a != PLANE_AXIS[plane]]
    A, B = shape[axes[0]], shape[axes[1]]
    return B - 1 - int(cursor[axes[1]]), A - 1 - int(cursor[axes[0]])


def window(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)


def intensity_window(vol: np.ndarray, p_lo: float = 0.5, p_hi: float = 99.5) -> tuple[float, float]:
    v = vol[vol > 0]
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(v, [p_lo, p_hi])
    return float(lo), float(hi)


def to_rgba_overlay(masks_2d: dict, opacity: float = 0.45, visible=("WT", "TC", "ET")) -> np.ndarray:
    """Stack WT (bottom) -> TC -> ET (top) as a single RGBA image."""
    shape = next(iter(masks_2d.values())).shape
    out = np.zeros((*shape, 4), np.float32)
    for r in ("WT", "TC", "ET"):
        if r in visible and r in masks_2d:
            m = masks_2d[r]
            out[m, :3] = REGION_COLORS[r]
            out[m, 3] = opacity
    return out


def default_cursor(masks: dict | None, shape: tuple) -> list:
    """Centre of the whole-tumour mask if present, else the volume centre."""
    if masks and masks.get("WT") is not None and masks["WT"].any():
        return [int(round(v)) for v in np.argwhere(masks["WT"]).mean(0)]
    return [s // 2 for s in shape]


def save_slices_png(path, background: np.ndarray, masks: dict, cursor=None, title: str = "", opacity: float = 0.45) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cursor = cursor or default_cursor(masks, background.shape)
    lo, hi = intensity_window(background)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), facecolor="black")
    for ax, plane in zip(axes, PLANES):
        idx = cursor[PLANE_AXIS[plane]]
        ax.imshow(window(extract_plane(background, plane, idx), lo, hi), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ov = to_rgba_overlay({r: extract_plane(m, plane, idx).astype(bool) for r, m in masks.items()}, opacity)
        ax.imshow(ov, interpolation="nearest")
        left, right, top, bottom = SIDE_LABELS[plane]
        kw = dict(color="#ffd54a", fontsize=11, transform=ax.transAxes, fontweight="bold")
        ax.text(0.02, 0.5, left, va="center", **kw)
        ax.text(0.98, 0.5, right, va="center", ha="right", **kw)
        ax.text(0.5, 0.97, top, ha="center", va="top", **kw)
        ax.text(0.5, 0.03, bottom, ha="center", **kw)
        ax.set_title(f"{plane} (slice {idx})", color="white", fontsize=10)
        ax.axis("off")
    fig.suptitle(title + "   [green=WT  red=TC  yellow=ET; prediction overlay, not ground truth]", color="white", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="black")
    plt.close(fig)

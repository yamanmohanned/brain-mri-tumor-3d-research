"""Mask post-processing (conservative: never silently drops tumour) and prediction-uncertainty summaries."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import REGIONS

STRUCT26 = np.ones((3, 3, 3), bool)


def enforce_hierarchy(masks: dict[str, np.ndarray]) -> tuple[dict, dict]:
    """ET ⊂ TC ⊂ WT by *adding* voxels (union), so no predicted tumour voxel is removed."""
    tc = masks["TC"] | masks["ET"]
    wt = masks["WT"] | tc
    log = {"TC_voxels_added": int(tc.sum() - masks["TC"].sum()), "WT_voxels_added": int(wt.sum() - masks["WT"].sum())}
    return {"TC": tc, "WT": wt, "ET": masks["ET"]}, log


def remove_small_components(mask: np.ndarray, min_voxels: int) -> tuple[np.ndarray, list[int]]:
    if min_voxels <= 0 or not mask.any():
        return mask, []
    lab, n = ndimage.label(mask, structure=STRUCT26)
    sizes = np.bincount(lab.ravel())[1:]
    drop = np.flatnonzero(sizes < min_voxels) + 1
    if not len(drop):
        return mask, []
    return mask & ~np.isin(lab, drop), [int(sizes[d - 1]) for d in drop]


def postprocess(masks: dict[str, np.ndarray], voxel_mm3: float, hierarchy: bool = True, min_component_mm3: float = 0.0):
    """Return (masks, log). Default settings remove nothing."""
    log: dict = {"hierarchy": hierarchy, "min_component_mm3": min_component_mm3}
    if hierarchy:
        masks, hlog = enforce_hierarchy(masks)
        log.update(hlog)
    min_vox = int(np.ceil(min_component_mm3 / voxel_mm3)) if min_component_mm3 > 0 else 0
    removed = {}
    out = {}
    for r in REGIONS:
        out[r], rm = remove_small_components(masks[r], min_vox)
        removed[r] = {"n_components": len(rm), "voxels": int(sum(rm))}
    log["components_removed"] = removed
    return out, log


def uncertainty_summary(probs: np.ndarray, masks: dict, voxel_mm3: float, tta_std: np.ndarray | None = None,
                        lo: float = 0.2, hi: float = 0.8) -> dict:
    """Voxel-probability based uncertainty descriptors. These are *not* calibrated confidence values.

    * ``volume_range_cm3``: region volume if the decision threshold were hi / lo instead of 0.5.
    * ``uncertain_band_pct``: share of (p>=lo) voxels whose probability lies in [lo, hi].
    * ``mean_prob_in_mask``: mean sigmoid output inside the final mask.
    * ``tta_mean_std``: mean flip-TTA standard deviation inside the mask (if TTA was run).
    """
    out = {}
    for i, r in enumerate(REGIONS):
        p = probs[i]
        n_lo, n_hi = int((p >= lo).sum()), int((p >= hi).sum())
        band = int(((p >= lo) & (p < hi)).sum())
        m = masks[r]
        d = {
            "volume_range_cm3": [round(n_hi * voxel_mm3 / 1000, 3), round(n_lo * voxel_mm3 / 1000, 3)],
            "uncertain_band_pct": round(100.0 * band / max(n_lo, 1), 1),
            "mean_prob_in_mask": round(float(p[m].mean()), 3) if m.any() else None,
        }
        if tta_std is not None:
            d["tta_mean_std"] = round(float(tta_std[i][m].mean()), 4) if m.any() else None
        out[r] = d
    return out

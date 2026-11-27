"""Segmentation evaluation metrics (physical units) and patient-level statistics.

Differences from the original notebook, which silently *dropped* empty / infinite cases
(``ignore_empty=True`` + ``np.isfinite`` filter) and so hid failures:

* Empty cases follow the BraTS convention: both empty -> Dice 1, HD95 0; exactly one empty -> Dice 0 and a
  distance penalty (image diagonal). Penalised and unpenalised statistics are both reported.
* HD95 follows MONAI's definition (max of the two directed 95th percentiles) so numbers stay comparable with the notebook.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from . import REGIONS
from .postprocess import STRUCT26

_CROSS = ndimage.generate_binary_structure(3, 1)


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask, structure=_CROSS, border_value=0)


def _bbox(a: np.ndarray, b: np.ndarray, pad: int = 2) -> tuple:
    u = a | b
    idx = [np.flatnonzero(u.any(axis=tuple(j for j in range(3) if j != i))) for i in range(3)]
    return tuple(slice(max(int(i[0]) - pad, 0), min(int(i[-1]) + 1 + pad, u.shape[k])) for k, i in enumerate(idx))


def surface_distances(pred: np.ndarray, gt: np.ndarray, spacing) -> tuple[np.ndarray, np.ndarray]:
    """Directed surface distances (mm): pred-surface -> nearest gt-surface and gt-surface -> nearest pred-surface."""
    sl = _bbox(pred, gt)
    p, g = pred[sl], gt[sl]
    sp, sg = _surface(p), _surface(g)
    d_to_g = ndimage.distance_transform_edt(~sg, sampling=spacing)
    d_to_p = ndimage.distance_transform_edt(~sp, sampling=spacing)
    return d_to_g[sp], d_to_p[sg]


def lesion_stats(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Lesion-wise detection with 26-connected components (a GT lesion is detected if any predicted voxel overlaps it)."""
    lg, ng = ndimage.label(gt, structure=STRUCT26)
    lp, npred = ndimage.label(pred, structure=STRUCT26)
    detected = int(len(np.unique(lg[pred & (lg > 0)])))
    fp_lesions = int(npred - len(np.unique(lp[gt & (lp > 0)])))
    return {"n_gt_lesions": int(ng), "n_gt_detected": detected, "n_pred_lesions": int(npred), "n_fp_lesions": fp_lesions}


def region_metrics(pred: np.ndarray, gt: np.ndarray, spacing, empty_penalty_mm: float, tolerances_mm=(1.0, 2.0)) -> dict:
    pred, gt = pred.astype(bool), gt.astype(bool)
    vv = float(np.prod(spacing))
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = int(pred.size - tp - fp - fn)
    pe, ge = (tp + fp) == 0, (tp + fn) == 0
    nan = float("nan")
    out = {
        "gt_empty": ge, "pred_empty": pe,
        "gt_vol_mm3": (tp + fn) * vv, "pred_vol_mm3": (tp + fp) * vv, "fp_vol_mm3": fp * vv, "fn_vol_mm3": fn * vv,
        "precision": tp / (tp + fp) if not pe else nan,
        "recall": tp / (tp + fn) if not ge else nan,           # sensitivity
        "specificity": tn / (tn + fp) if (tn + fp) else nan,
    }
    if pe and ge:
        out.update(dice=1.0, iou=1.0, f1=1.0, hd95=0.0, hd95_raw=0.0, assd=0.0, assd_raw=0.0)
        out.update({f"sdice_{t:g}mm": 1.0 for t in tolerances_mm})
    elif pe or ge:
        out.update(dice=0.0, iou=0.0, f1=0.0, hd95=empty_penalty_mm, hd95_raw=nan, assd=empty_penalty_mm, assd_raw=nan)
        out.update({f"sdice_{t:g}mm": 0.0 for t in tolerances_mm})
    else:
        out.update(dice=2 * tp / (2 * tp + fp + fn), iou=tp / (tp + fp + fn))
        out["f1"] = out["dice"]                                # F1 == Dice for binary masks (kept as a separate column for completeness)
        a, b = surface_distances(pred, gt, spacing)
        out["hd95"] = out["hd95_raw"] = float(max(np.percentile(a, 95), np.percentile(b, 95)))
        out["assd"] = out["assd_raw"] = float((a.sum() + b.sum()) / (len(a) + len(b)))
        for t in tolerances_mm:
            out[f"sdice_{t:g}mm"] = float(((a <= t).sum() + (b <= t).sum()) / (len(a) + len(b)))
    out["vol_error_pct"] = 100.0 * (out["pred_vol_mm3"] - out["gt_vol_mm3"]) / out["gt_vol_mm3"] if not ge else nan
    out.update(lesion_stats(pred, gt))
    return out


def evaluate_case(pred_masks: dict, gt_masks: dict, spacing, empty_penalty_mm: float | None = None) -> dict:
    """{region: metrics dict}. ``empty_penalty_mm`` defaults to the volume diagonal."""
    if empty_penalty_mm is None:
        empty_penalty_mm = float(np.linalg.norm(np.array(next(iter(gt_masks.values())).shape) * np.asarray(spacing)))
    return {r: region_metrics(pred_masks[r], gt_masks[r], spacing, empty_penalty_mm) for r in REGIONS}


def brats_labels_to_regions(labels: np.ndarray, et_label: int = 4) -> dict[str, np.ndarray]:
    """BraTS label map -> {TC, WT, ET} (same definition as MONAI ConvertToMultiChannelBasedOnBratsClasses)."""
    return {"TC": (labels == 1) | (labels == et_label), "WT": (labels == 1) | (labels == 2) | (labels == et_label), "ET": labels == et_label}


# ----------------------------------------------------------------------------- statistics

def bootstrap_ci(values, stat=np.mean, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Patient-level percentile bootstrap. NaNs (undefined metric) are excluded and counted."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"n": 0, "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boots = np.apply_along_axis(stat, 1, v[idx]) if stat is not np.mean else v[idx].mean(1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"n": int(len(v)), "mean": float(stat(v)), "ci_low": float(lo), "ci_high": float(hi),
            "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "median": float(np.median(v)),
            "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)), "min": float(v.min()), "max": float(v.max())}


def summarize_metrics(rows: list[dict], metrics=("dice", "iou", "precision", "recall", "specificity", "f1", "hd95", "assd",
                                                 "sdice_1mm", "sdice_2mm", "vol_error_pct"), seed: int = 0, n_boot: int = 2000) -> dict:
    """``rows``: one dict per (patient, region) with keys patient_id, region and metric columns."""
    import pandas as pd
    df = pd.DataFrame(rows)
    out: dict = {}
    for r in REGIONS:
        d = df[df.region == r]
        if d.empty:
            continue
        present = d[~d.gt_empty]
        s = {"n_patients": int(len(d)), "n_gt_present": int(len(present)), "n_gt_empty": int(d.gt_empty.sum()),
             "n_missed": int((~d.gt_empty & d.pred_empty).sum()), "n_false_positive_region": int((d.gt_empty & ~d.pred_empty).sum()),
             "metrics": {}}
        s["missed_rate"] = s["n_missed"] / max(s["n_gt_present"], 1)
        for m in metrics:
            if m in d.columns:
                s["metrics"][m] = bootstrap_ci(d[m].to_numpy(), seed=seed, n_boot=n_boot)
        # hd95 / assd restricted to cases where both masks are non-empty (no penalties) - reported alongside the penalised numbers
        both = d[~d.gt_empty & ~d.pred_empty]
        for m in ("hd95_raw", "assd_raw"):
            if m in d:
                s["metrics"][m] = bootstrap_ci(both[m].to_numpy(), seed=seed, n_boot=n_boot)
        les = present[["n_gt_lesions", "n_gt_detected", "n_pred_lesions", "n_fp_lesions"]].sum()
        s["lesion_wise"] = {"gt_lesions": int(les.n_gt_lesions), "detected": int(les.n_gt_detected),
                            "lesion_sensitivity": float(les.n_gt_detected / max(les.n_gt_lesions, 1)),
                            "false_positive_lesions": int(d.n_fp_lesions.sum()),
                            "fp_lesions_per_patient": float(d.n_fp_lesions.sum() / len(d))}
        s["fp_vol_cm3_median"] = float(np.median(d.fp_vol_mm3) / 1000)
        s["fn_vol_cm3_median"] = float(np.median(d.fn_vol_mm3) / 1000)
        out[r] = s
    return out


def size_strata(rows: list[dict], region: str, edges_cm3: tuple, metric: str = "dice") -> list[dict]:
    """Performance by GT size bin (small-lesion analysis). ``edges_cm3`` e.g. (0, 1, 5, inf)."""
    import pandas as pd
    d = pd.DataFrame(rows)
    d = d[(d.region == region) & (~d.gt_empty)].copy()
    d["gt_cm3"] = d.gt_vol_mm3 / 1000
    res = []
    for lo, hi in zip(edges_cm3[:-1], edges_cm3[1:]):
        b = d[(d.gt_cm3 >= lo) & (d.gt_cm3 < hi)]
        if len(b):
            res.append({"range_cm3": [lo, None if hi == np.inf else hi], "n": int(len(b)), f"mean_{metric}": float(b[metric].mean()),
                        "median_" + metric: float(b[metric].median()), "n_missed": int(b.pred_empty.sum())})
    return res


@dataclass
class Calibration:
    """Accumulates binned reliability statistics over many patients without keeping voxels in memory."""
    n_bins: int = 15
    counts: np.ndarray = field(default=None)
    sum_p: np.ndarray = field(default=None)
    sum_y: np.ndarray = field(default=None)
    sq_err: float = 0.0
    n: int = 0

    def __post_init__(self):
        z = np.zeros((len(REGIONS), self.n_bins))
        self.counts, self.sum_p, self.sum_y = z.copy(), z.copy(), z.copy()
        self.sq = np.zeros(len(REGIONS))

    def update(self, probs: np.ndarray, gt: dict, stride: int = 1) -> None:
        """probs (3,X,Y,Z) on the cropped model grid; gt masks cropped the same way."""
        for i, r in enumerate(REGIONS):
            p = probs[i].ravel()[::stride].astype(np.float64)
            y = gt[r].ravel()[::stride].astype(np.float64)
            b = np.minimum((p * self.n_bins).astype(int), self.n_bins - 1)
            self.counts[i] += np.bincount(b, minlength=self.n_bins)
            self.sum_p[i] += np.bincount(b, weights=p, minlength=self.n_bins)
            self.sum_y[i] += np.bincount(b, weights=y, minlength=self.n_bins)
            self.sq[i] += float(((p - y) ** 2).sum())

    def result(self) -> dict:
        out = {}
        for i, r in enumerate(REGIONS):
            n = self.counts[i].sum()
            if n == 0:
                continue
            nz = self.counts[i] > 0
            conf = self.sum_p[i][nz] / self.counts[i][nz]
            acc = self.sum_y[i][nz] / self.counts[i][nz]
            ece = float((self.counts[i][nz] / n * np.abs(conf - acc)).sum())
            out[r] = {"ece": ece, "brier": float(self.sq[i] / n), "n_voxels": int(n),
                      "reliability": {"confidence": conf.round(4).tolist(), "accuracy": acc.round(4).tolist(),
                                      "count": self.counts[i][nz].astype(int).tolist()}}
        return out


# ----------------------------------------------------------------------------- acceptance & evidence level

DEFAULT_CRITERIA = {
    "note": ("RESEARCH ASSUMPTIONS, fixed before any evaluation with this pipeline. Anchored to published BraTS-2020 challenge "
             "performance, not tuned to the legacy notebook results. They are not clinically derived and need review by a qualified "
             "medical-imaging specialist."),
    "dice_mean_ci_lower": {"WT": 0.85, "TC": 0.75, "ET": 0.65},
    "hd95_raw_median_mm_max": {"WT": 8.0, "TC": 10.0, "ET": 10.0},
    "missed_rate_max": 0.05,
    "wt_abs_median_volume_error_pct_max": 15.0,
}


def check_acceptance(summary: dict, criteria: dict | None = None) -> dict:
    c = criteria or DEFAULT_CRITERIA
    checks = []

    def add(name, value, ok, rule):
        checks.append({"check": name, "value": None if value is None or (isinstance(value, float) and np.isnan(value)) else round(float(value), 4),
                       "rule": rule, "pass": bool(ok)})
    for r, thr in c["dice_mean_ci_lower"].items():
        v = summary.get(r, {}).get("metrics", {}).get("dice", {}).get("ci_low", float("nan"))
        add(f"{r} Dice mean, 95% CI lower bound", v, v >= thr, f">= {thr}")
    for r, thr in c["hd95_raw_median_mm_max"].items():
        v = summary.get(r, {}).get("metrics", {}).get("hd95_raw", {}).get("median", float("nan"))
        add(f"{r} HD95 median (both masks non-empty), mm", v, v <= thr, f"<= {thr}")
    for r in c["dice_mean_ci_lower"]:
        v = summary.get(r, {}).get("missed_rate", float("nan"))
        add(f"{r} missed-region rate (GT present, nothing predicted)", v, v <= c["missed_rate_max"], f"<= {c['missed_rate_max']}")
    ve = summary.get("WT", {}).get("metrics", {}).get("vol_error_pct", {}).get("median", float("nan"))
    add("WT |median relative volume error|, %", abs(ve) if ve == ve else float("nan"), abs(ve) <= c["wt_abs_median_volume_error_pct_max"],
        f"<= {c['wt_abs_median_volume_error_pct_max']}")
    return {"passed": all(x["pass"] for x in checks), "checks": checks, "criteria": c}


EVIDENCE_LEVELS = ("Not validated", "Experimental / preliminary", "Validated on an internal held-out dataset",
                   "Independently evaluated on external data")


def classify_evidence(*, heldout_eval: bool, patient_level_split: bool, test_untouched: bool, consistent_pipeline: bool,
                      ci_reported: bool, criteria_passed: bool | None, external_eval_passed: bool = False) -> tuple[str, list[str]]:
    """Map the available evidence to one of the four research-evidence categories (NOT clinical approval)."""
    why = []
    if not heldout_eval:
        return EVIDENCE_LEVELS[0], ["no held-out evaluation record for this checkpoint"]
    for ok, msg in ((patient_level_split, "split is not verified patient-level"), (test_untouched, "test set not verified untouched by training/selection"),
                    (consistent_pipeline, "evaluation did not use the deployed pre/post-processing"), (ci_reported, "no patient-level confidence intervals")):
        if not ok:
            why.append(msg)
    if criteria_passed is False:
        why.append("pre-specified acceptance criteria not met")
    if criteria_passed is None:
        why.append("acceptance criteria not evaluated")
    if why:
        return EVIDENCE_LEVELS[1], why
    if external_eval_passed:
        return EVIDENCE_LEVELS[3], []
    return EVIDENCE_LEVELS[2], ["no external dataset evaluation"]

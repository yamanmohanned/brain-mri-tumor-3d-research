"""Dataset integrity audit: file/mask correspondence, geometry, labels, corruption, class balance, duplicates."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from . import MODALITIES
from .data import discover_cases, locate_root
from .geometry import axcodes, voxel_sizes
from .metrics import brats_labels_to_regions


def _fingerprint(vol: np.ndarray, n: int = 20) -> np.ndarray:
    z = ndimage.zoom(vol, [n / s for s in vol.shape], order=1)
    z = z.ravel().astype(np.float64)
    return (z - z.mean()) / (z.std() + 1e-9)


def _file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def audit_case(case: dict) -> dict:
    rec = {"id": case["id"], "problems": []}
    try:
        imgs = {m: nib.load(f) for m, f in zip(MODALITIES, case["image"])}
        seg_img = nib.load(case["label"])
        ref = imgs["t1"]
        rec.update(shape=list(ref.shape), spacing=[round(float(s), 3) for s in voxel_sizes(ref.affine)], orientation=axcodes(ref.affine))
        for m, im in imgs.items():
            if im.shape != ref.shape or not np.allclose(im.affine, ref.affine, atol=1e-3):
                rec["problems"].append(f"{m} grid differs from t1")
        if seg_img.shape != ref.shape or not np.allclose(seg_img.affine, ref.affine, atol=1e-3):
            rec["problems"].append("segmentation grid differs from images (mis-aligned mask)")
        data = {m: np.asarray(im.get_fdata(dtype=np.float32)) for m, im in imgs.items()}
        for m, d in data.items():
            if not np.isfinite(d).all():
                rec["problems"].append(f"{m} has NaN/Inf")
            if not d.any():
                rec["problems"].append(f"{m} is empty")
            elif d.std() == 0:
                rec["problems"].append(f"{m} is constant")
        seg = np.rint(np.asarray(seg_img.get_fdata(dtype=np.float32))).astype(np.int16)
        vals = sorted(int(v) for v in np.unique(seg))
        rec["label_values"] = vals
        if not set(vals) <= {0, 1, 2, 4}:
            rec["problems"].append(f"unexpected label values {vals} (BraTS 2020 uses 0,1,2,4)")
        vv = float(np.prod(voxel_sizes(ref.affine)))
        reg = brats_labels_to_regions(seg)
        rec.update({f"gt_{r}_cm3": float(m.sum() * vv / 1000) for r, m in reg.items()})
        rec["wt_components"] = int(ndimage.label(reg["WT"], structure=np.ones((3, 3, 3)))[1])
        fg = (np.stack(list(data.values())) > 0).any(0)
        rec["brain_cm3"] = float(fg.sum() * vv / 1000)
        rec["zero_fraction"] = float(1 - fg.mean())
        rec["gt_in_brain_fraction"] = float((reg["WT"] & fg).sum() / max(reg["WT"].sum(), 1))
        if reg["WT"].sum() == 0:
            rec["problems"].append("no tumour in ground truth")
        if rec["gt_in_brain_fraction"] < 0.99 and reg["WT"].any():
            rec["problems"].append("ground truth lies outside image foreground")
        rec["brain_fractions"] = {r: float(m.sum() / max(fg.sum(), 1)) for r, m in reg.items()}
        rec["sha1_flair"] = _file_sha1(case["image"][MODALITIES.index("flair")])
        rec["_fp"] = _fingerprint(data["flair"])
    except Exception as exc:  # corrupted / unreadable
        rec["problems"].append(f"unreadable: {type(exc).__name__}: {exc}")
    return rec


def audit_dataset(root: str | Path, workers: int = 4, near_dup_corr: float = 0.995, progress=None) -> dict:
    root = locate_root(root)
    cases, skipped = discover_cases(root)
    ids = list(cases)
    recs = []
    with ThreadPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(audit_case, (cases[k] for k in ids)), 1):
            recs.append(r)
            if progress:
                progress(i, len(ids))
    ok = [r for r in recs if "gt_WT_cm3" in r]
    # duplicates
    by_hash = {}
    for r in ok:
        by_hash.setdefault(r["sha1_flair"], []).append(r["id"])
    exact = [v for v in by_hash.values() if len(v) > 1]
    near = []
    if len(ok) > 1:
        fp = np.stack([r["_fp"] for r in ok])
        c = fp @ fp.T / fp.shape[1]
        np.fill_diagonal(c, -1)
        for i, j in zip(*np.where(np.triu(c >= near_dup_corr, 1))):
            near.append({"pair": [ok[i]["id"], ok[j]["id"]], "corr": round(float(c[i, j]), 5)})
        nn = np.sort(c.max(1))[::-1]
        nn_summary = {"max_pair_corr": float(nn[0]), "median_nearest_neighbour_corr": float(np.median(nn))}
    else:
        nn_summary = {}
    for r in recs:
        r.pop("_fp", None)

    def q(key):
        v = np.array([r[key] for r in ok]) if ok else np.array([np.nan])
        return {"min": float(v.min()), "q25": float(np.percentile(v, 25)), "median": float(np.median(v)), "q75": float(np.percentile(v, 75)),
                "max": float(v.max())}

    summary = {
        "root": str(root), "n_folders": len(ids) + len(skipped), "n_complete": len(ids), "skipped_incomplete": [{"id": i, "reason": w} for i, w in skipped],
        "n_with_problems": sum(1 for r in recs if r["problems"]), "problem_cases": {r["id"]: r["problems"] for r in recs if r["problems"]},
        "shapes": Counter(map(tuple, (r["shape"] for r in ok))).most_common(), "spacings": Counter(map(tuple, (r["spacing"] for r in ok))).most_common(),
        "orientations": Counter(r["orientation"] for r in ok).most_common(), "label_sets": Counter(tuple(r["label_values"]) for r in ok).most_common(),
        "gt_volume_cm3": {k: q(f"gt_{k}_cm3") for k in ("WT", "TC", "ET")}, "n_without_ET": sum(1 for r in ok if r["gt_ET_cm3"] == 0),
        "multifocal_WT_cases": sum(1 for r in ok if r["wt_components"] > 1),
        "class_balance_mean_fraction_of_brain": {k: float(np.mean([r["brain_fractions"][k] for r in ok])) for k in ("WT", "TC", "ET")} if ok else {},
        "brain_cm3": q("brain_cm3") if ok else {}, "exact_duplicate_groups": exact, "near_duplicate_candidates": near, "similarity_summary": nn_summary,
        "near_duplicate_note": f"candidates are FLAIR fingerprint correlations >= {near_dup_corr}; brains co-registered to one atlas correlate "
                               "strongly, so candidates need manual review before being treated as duplicates.",
    }
    # tertile strata on WT volume (for stratified splitting)
    if ok:
        edges = np.percentile([r["gt_WT_cm3"] for r in ok], [33.3, 66.7])
        for r in ok:
            r["size_stratum"] = int(np.searchsorted(edges, r["gt_WT_cm3"]))
        summary["wt_tertile_edges_cm3"] = [float(e) for e in edges]
    return {"summary": summary, "cases": recs}


def legacy_split_consistency(audit: dict, split_test_ids: list[str], first_ids=("BraTS20_Training_296", "BraTS20_Training_186", "BraTS20_Training_151"),
                             chosen="BraTS20_Training_283") -> dict:
    """The notebook chose the FIRST test patient (in its shuffled test order) with 40<=WT<=100 cm3 and printed that it was 283 (46.9 cm3).
    If this reproduced split is right, the patients before 283 in the test list must fall outside [40,100] cm3 and 283 inside."""
    vol = {r["id"]: r.get("gt_WT_cm3") for r in audit["cases"]}
    order = split_test_ids[: split_test_ids.index(chosen) + 1]
    checks = {pid: vol.get(pid) for pid in order}
    ok_before = all(v is not None and not (40 <= v <= 100) for pid, v in checks.items() if pid != chosen)
    ok_chosen = vol.get(chosen) is not None and 40 <= vol[chosen] <= 100
    return {"checked_in_order": checks, "earlier_patients_outside_40_100": ok_before, "chosen_inside_40_100": ok_chosen,
            "notebook_reported_chosen_wt_cm3": 46.86, "consistent": bool(ok_before and ok_chosen)}


def save_audit(audit: dict, out_dir: str | Path) -> None:
    import pandas as pd
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit_summary.json").write_text(json.dumps(audit["summary"], indent=1, default=str), encoding="utf-8")
    df = pd.DataFrame([{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()} for r in audit["cases"]])
    df.to_csv(out / "audit_cases.csv", index=False)

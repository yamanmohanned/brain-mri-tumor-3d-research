"""Paired comparison of two evaluations (e.g. legacy baseline vs new model) under IDENTICAL conditions.

    python -m braintumor.compare --a runs/eval_baseline --b runs/eval_new --out runs/compare_baseline_vs_new

Both runs must cover the same patients; differences are computed per patient (B - A) and summarised with patient-level bootstrap CIs and a
Wilcoxon signed-rank test. This is descriptive: do NOT use a test-split comparison to choose between candidate models (use `val`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import REGIONS
from .metrics import bootstrap_ci

PAIRED = {"dice": "higher is better", "iou": "higher is better", "hd95_raw": "lower is better", "assd_raw": "lower is better", "sdice_1mm": "higher is better",
          "m_vol_abs_rel_err_pct": "lower is better", "m_max_diam_abs_err_mm": "lower is better"}


def _conditions(d: Path) -> dict:
    p = d / "summary.json"
    if not p.exists():
        return {}
    ev = json.loads(p.read_text())["evaluation"]
    return {k: ev.get(k) for k in ("split", "split_manifest_sha256", "tta", "overlap", "hierarchy", "min_component_mm3")}


def compare(dir_a: str | Path, dir_b: str | Path, n_boot: int = 2000, seed: int = 0) -> dict:
    a, b = Path(dir_a), Path(dir_b)
    da, db = pd.read_csv(a / "per_patient_metrics.csv"), pd.read_csv(b / "per_patient_metrics.csv")
    ca, cb = _conditions(a), _conditions(b)
    identical = bool(ca) and ca == cb
    out = {"A": str(a), "B": str(b), "conditions_A": ca, "conditions_B": cb, "identical_evaluation_conditions": identical,
           "warning": None if identical else "evaluation conditions differ or are missing: this is NOT a like-for-like comparison", "regions": {}}
    ids_a, ids_b = set(da.patient_id), set(db.patient_id)
    out["n_common_patients"], out["only_in_A"], out["only_in_B"] = len(ids_a & ids_b), len(ids_a - ids_b), len(ids_b - ids_a)
    for r in REGIONS:
        x = da[da.region == r].set_index("patient_id")
        y = db[db.region == r].set_index("patient_id")
        common = sorted(set(x.index) & set(y.index))
        res = {}
        for m, direction in PAIRED.items():
            if m not in x or m not in y:
                continue
            xa, yb = x.loc[common, m].astype(float), y.loc[common, m].astype(float)
            ok = np.isfinite(xa) & np.isfinite(yb)
            diff = (yb - xa)[ok].to_numpy()
            if len(diff) < 2:
                continue
            ci = bootstrap_ci(diff, n_boot=n_boot, seed=seed)
            try:
                p = float(stats.wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
            except ValueError:
                p = float("nan")
            res[m] = {"direction": direction, "n_pairs": int(len(diff)), "mean_A": float(xa[ok].mean()), "mean_B": float(yb[ok].mean()),
                      "mean_diff_B_minus_A": ci["mean"], "ci95": [ci["ci_low"], ci["ci_high"]], "median_diff": ci["median"], "wilcoxon_p": p,
                      "ci_excludes_zero": bool(ci["ci_low"] > 0 or ci["ci_high"] < 0)}
        out["regions"][r] = res
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="evaluation dir of the baseline")
    ap.add_argument("--b", required=True, help="evaluation dir of the candidate")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = compare(a.a, a.b)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "comparison.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"identical_conditions": res["identical_evaluation_conditions"], "n_common": res["n_common_patients"],
                      "dice_diff": {r: v.get("dice", {}).get("mean_diff_B_minus_A") for r, v in res["regions"].items()}}, indent=1))


if __name__ == "__main__":
    main()

"""Held-out evaluation with the DEPLOYED pipeline (same preprocessing / inference / post-processing as the app).

    python -m braintumor.evaluate --checkpoint weights/segresnet_best.pth --data-root <BraTS> --manifest splits/legacy_seed42.json \
        --split test --out runs/eval_segresnet [--tta] [--assert-trained-on-manifest]

Outputs: per_patient_metrics.csv (one row per patient x region), summary.json (means, patient-level bootstrap CIs, size strata,
lesion-wise detection, calibration, acceptance checks, evidence level), failures.csv, dice_boxplot.png. The checkpoint's
``*.card.json`` is updated so the GUI/report show the evidence level next to every analysis.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import REGIONS
from .geometry import voxel_sizes
from .inference import InferenceConfig
from .io import load_label_map, load_modalities
from .metrics import (DEFAULT_CRITERIA, Calibration, brats_labels_to_regions, check_acceptance, classify_evidence, evaluate_case, size_strata,
                      summarize_metrics)
from .model import load_checkpoint, update_card
from .pipeline import AnalysisConfig, require_valid, segment_volume
from .preprocess import prepare
from .splits import load_manifest

SIZE_EDGES_CM3 = {"WT": (0, 20, 60, float("inf")), "TC": (0, 5, 20, float("inf")), "ET": (0, 1, 5, float("inf"))}   # research assumptions


def _registry_path(manifest_path: str) -> Path:
    return Path(manifest_path).with_suffix(".test_registry.json")


def _register_test_use(manifest_path: str, ckpt_sha: str, n_cases: int) -> int:
    """Record which checkpoints touched the test split. Returns the number of DISTINCT checkpoints evaluated on it."""
    p = _registry_path(manifest_path)
    reg = json.loads(p.read_text()) if p.exists() else {"entries": []}
    reg["entries"].append({"checkpoint_sha256": ckpt_sha, "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), "n_cases": n_cases})
    p.write_text(json.dumps(reg, indent=1))
    return len({e["checkpoint_sha256"] for e in reg["entries"]})


def evaluate(checkpoint: str, data_root: str, manifest: str, split: str = "test", out_dir: str = "runs/eval", tta: str = "none", limit: int = 0,
             device: str = "auto", overlap: float = 0.5, hierarchy: bool = True, min_component_mm3: float = 0.0, n_boot: int = 2000, seed: int = 0,
             calibration: bool = True, attest_trained_on_manifest: bool = False, resume: bool = True, update_checkpoint_card: bool = True,
             roi: tuple = (128, 128, 128)) -> dict:
    from .data import discover_cases, select
    model, info = load_checkpoint(checkpoint, device="cpu" if device == "cpu" else device if device != "auto" else "cpu")
    man = load_manifest(manifest)
    ids = man["splits"][split][: limit or None]
    cases, _ = discover_cases(data_root)
    files = select(cases, ids)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = AnalysisConfig(infer=InferenceConfig(roi=tuple(roi), overlap=overlap, tta=tta, device=device), hierarchy=hierarchy, min_component_mm3=min_component_mm3)
    csv_p = out / "per_patient_metrics.csv"
    done = pd.read_csv(csv_p) if (resume and csv_p.exists()) else pd.DataFrame()
    rows = done.to_dict("records") if len(done) else []
    done_ids = set(done.patient_id) if len(done) else set()
    calib = Calibration() if calibration else None
    cal_partial = out / "calibration_partial.json"
    print(f"Evaluating {info['arch']} ({info['sha256'][:12]}) on {split}: {len(ids)} patients ({len(done_ids)} already done)")
    t_all = time.time()
    for k, case in enumerate(files, 1):
        if case["id"] in done_ids:
            continue
        t0 = time.time()
        vol = load_modalities(dict(zip(("t1", "t1ce", "t2", "flair"), case["image"])))
        require_valid(vol)
        labels, _ = load_label_map(case["label"], like=vol)
        prep = prepare(vol)
        pred, masks, cropped, _ = segment_volume(prep, model, cfg)
        gt_work = {r: prep.grid.to_work(m.astype(np.uint8), order=0).astype(bool) for r, m in brats_labels_to_regions(labels).items()}
        res = evaluate_case(masks, gt_work, voxel_sizes(prep.work_affine))
        for r in REGIONS:
            rows.append({"patient_id": case["id"], "region": r, **res[r], "seconds": round(time.time() - t0, 1)})
        if calib is not None:
            calib.update(pred.probs, {r: gt_work[r][prep.crop] for r in REGIONS}, stride=2)
        pd.DataFrame(rows).to_csv(csv_p, index=False)
        print(f"  [{k}/{len(files)}] {case['id']}  Dice " + " ".join(f"{r}={res[r]['dice']:.3f}" for r in REGIONS) + f"  ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    recs = df.to_dict("records")
    summary = summarize_metrics(recs, seed=seed, n_boot=n_boot)
    strata = {r: size_strata(recs, r, SIZE_EDGES_CM3[r]) for r in REGIONS}
    acc = check_acceptance(summary)
    # --- evidence level
    n_models_on_test = _register_test_use(manifest, info["sha256"], len(ids)) if split == "test" else 0
    card = info.get("card", {})
    trained_here = bool(card.get("manifest_sha256")) and card["manifest_sha256"] == man["file_sha256"]
    untouched = split == "test" and (trained_here or attest_trained_on_manifest) and n_models_on_test <= 1
    level, why = classify_evidence(heldout_eval=split == "test", patient_level_split=True, test_untouched=untouched, consistent_pipeline=True,
                                   ci_reported=True, criteria_passed=acc["passed"])
    notes = []
    if split != "test":
        notes.append(f"evaluated on '{split}', not the held-out test split")
    if not trained_here and not attest_trained_on_manifest:
        notes.append("training provenance not recorded: cannot verify the test patients were unseen (re-run with --assert-trained-on-manifest only if you KNOW "
                     "the checkpoint was trained/selected on this manifest's train/val patients only)")
    if attest_trained_on_manifest and not trained_here:
        notes.append("test-set independence is ATTESTED BY THE USER, not verified from the checkpoint")
    if n_models_on_test > 1:
        notes.append(f"{n_models_on_test} different checkpoints have been evaluated on this test split: selection bias is possible")
    evaluation = {
        "evidence_level": level, "why": why + notes, "criteria_passed": acc["passed"], "acceptance": acc, "split": split, "n_test": len(ids),
        "split_manifest": str(manifest), "split_manifest_sha256": man["file_sha256"], "checkpoint_sha256": info["sha256"], "tta": tta, "overlap": overlap,
        "hierarchy": hierarchy, "min_component_mm3": min_component_mm3, "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "criteria_note": DEFAULT_CRITERIA["note"], "external_validation": "none available - not evaluated on external data",
        "summary": {r: {"dice": round(summary[r]["metrics"]["dice"]["mean"], 4),
                        "dice_ci95": [round(summary[r]["metrics"]["dice"]["ci_low"], 4), round(summary[r]["metrics"]["dice"]["ci_high"], 4)],
                        "hd95_median_mm": round(summary[r]["metrics"]["hd95_raw"]["median"], 2) if summary[r]["metrics"]["hd95_raw"]["n"] else None,
                        "missed_rate": round(summary[r]["missed_rate"], 4)} for r in summary},
    }
    result = {"evaluation": evaluation, "metrics": summary, "size_strata_cm3": strata, "calibration": calib.result() if calib else None,
              "total_minutes": round((time.time() - t_all) / 60, 1)}
    (out / "summary.json").write_text(json.dumps(result, indent=1, default=float))
    worst = df[(df.dice < 0.5) | (df.pred_empty & ~df.gt_empty)].sort_values("dice")
    worst.to_csv(out / "failures.csv", index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot([df[df.region == r].dice.dropna() for r in REGIONS], labels=REGIONS, showmeans=True)
        ax.set(ylabel="Dice", title=f"Per-patient Dice ({split}, n={len(ids)})", ylim=(0, 1))
        fig.tight_layout()
        fig.savefig(out / "dice_boxplot.png", dpi=120)
        plt.close(fig)
    except Exception:
        pass
    if update_checkpoint_card:
        update_card(checkpoint, evaluation=evaluation)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out", default="runs/eval")
    ap.add_argument("--tta", action="store_true", help="4-pass flip TTA (4x slower)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--roi", type=int, default=128, help="sliding-window cube size (model was trained with 128)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-hierarchy", action="store_true")
    ap.add_argument("--assert-trained-on-manifest", action="store_true")
    ap.add_argument("--no-card-update", action="store_true")
    a = ap.parse_args(argv)
    r = evaluate(a.checkpoint, a.data_root, a.manifest, a.split, a.out, "flips4" if a.tta else "none", a.limit, a.device, hierarchy=not a.no_hierarchy,
                 attest_trained_on_manifest=a.assert_trained_on_manifest, update_checkpoint_card=not a.no_card_update, roi=(a.roi,) * 3)
    print(json.dumps(r["evaluation"], indent=1))


if __name__ == "__main__":
    main()

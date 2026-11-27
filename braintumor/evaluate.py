"""Held-out evaluation with the DEPLOYED pipeline (same preprocessing / inference / post-processing as the app).

    python -m braintumor.evaluate --checkpoint weights/segresnet_best.pth --data-root <BraTS> --manifest splits/legacy_seed42.json \
        --split test --out runs/eval_segresnet [--tta] [--assert-trained-on-manifest]

Splits: train | val | test | test_legacy | external.  ``val`` is for model selection; ``test`` / ``external`` are final evaluations;
``test_legacy`` is the notebook's test set, which was already used to compare models and is therefore never an untouched benchmark.

Outputs: per_patient_metrics.csv (one row per patient x region; segmentation, boundary AND physical-measurement errors), summary.json
(means, patient-level bootstrap CIs, per-dataset breakdown, size strata, lesion-wise detection, calibration, acceptance checks, evidence level),
failures.csv, dice_boxplot.png. The checkpoint's ``*.card.json`` records one evaluation per split plus the strongest evidence level; the GUI shows it.
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
from .datasets import load_registry, regions_from_labels
from .geometry import voxel_sizes
from .inference import InferenceConfig
from .io import load_label_map, load_modalities
from .measure import measurement_errors
from .metrics import (DEFAULT_CRITERIA, EVIDENCE_LEVELS, Calibration, check_acceptance, classify_evidence, evaluate_case, size_strata, summarize_metrics)
from .model import load_checkpoint, read_card, update_card
from .pipeline import AnalysisConfig, require_valid, segment_volume
from .preprocess import prepare
from .splits import load_manifest

SIZE_EDGES_CM3 = {"WT": (0, 20, 60, float("inf")), "TC": (0, 5, 20, float("inf")), "ET": (0, 1, 5, float("inf"))}   # research assumptions
SPLITS = ("train", "val", "test", "test_legacy", "external")
FINAL_SPLITS = ("test", "test_legacy", "external")
METRICS = ("dice", "iou", "precision", "recall", "specificity", "f1", "hd95", "assd", "sdice_1mm", "sdice_2mm", "vol_error_pct",
           "m_vol_abs_err_cm3", "m_vol_rel_err_pct", "m_vol_abs_rel_err_pct", "m_max_diam_err_mm", "m_max_diam_abs_err_mm", "m_length_err_mm", "m_width_err_mm",
           "m_height_err_mm", "m_axial_diam_err_mm", "m_surface_area_err_cm2", "m_surface_area_abs_err_cm2", "m_centroid_dist_mm")


def _registry_path(manifest_path: str) -> Path:
    return Path(manifest_path).with_suffix(".test_registry.json")


def _register_use(manifest_path: str, split: str, ckpt_sha: str, n_cases: int) -> tuple[int, int]:
    """Record which checkpoints touched a held-out split. Returns (distinct checkpoints seen in THIS registry, historical models compared before)."""
    p = _registry_path(manifest_path)
    reg = json.loads(p.read_text()) if p.exists() else {"entries": []}
    reg.setdefault("entries", []).append({"split": split, "checkpoint_sha256": ckpt_sha, "n_cases": n_cases,
                                          "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")})
    p.write_text(json.dumps(reg, indent=1))
    distinct = len({e["checkpoint_sha256"] for e in reg["entries"] if e.get("split") == split})
    hist = sum(h.get("n_models", 1) for h in reg.get("historical", {}).get(split, []))
    return distinct, hist


def strongest(evaluations: dict) -> dict | None:
    """The evaluation with the highest evidence level (ties: most recent)."""
    best = None
    for ev in evaluations.values():
        rank = EVIDENCE_LEVELS.index(ev["evidence_level"]) if ev.get("evidence_level") in EVIDENCE_LEVELS else -1
        if best is None or (rank, ev.get("utc", "")) >= (best[0], best[1].get("utc", "")):
            best = (rank, ev)
    return best[1] if best else None


def evaluate(checkpoint: str, data_root: str = "", manifest: str = "", split: str = "test", out_dir: str = "runs/eval", tta: str = "none", limit: int = 0,
             device: str = "auto", overlap: float = 0.5, hierarchy: bool = True, min_component_mm3: float = 0.0, n_boot: int = 2000, seed: int = 0,
             calibration: bool = True, attest_trained_on_manifest: bool = False, resume: bool = True, update_checkpoint_card: bool = True,
             roi: tuple = (128, 128, 128), data_roots: dict | None = None, measure_errors: bool = True) -> dict:
    from .data import discover_many, select
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    model, info = load_checkpoint(checkpoint, device="cpu" if device == "cpu" else device if device != "auto" else "cpu")
    man = load_manifest(manifest)
    if split not in man["splits"]:
        raise KeyError(f"manifest has no {split!r} split (available: {sorted(man['splits'])})")
    ids = man["splits"][split][: limit or None]
    if not ids:
        raise ValueError(f"split {split!r} is empty")
    cases, _ = discover_many(data_roots or {"brats2020": data_root}, load_registry())
    files = select(cases, ids)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = AnalysisConfig(infer=InferenceConfig(roi=tuple(roi), overlap=overlap, tta=tta, device=device), hierarchy=hierarchy, min_component_mm3=min_component_mm3)
    csv_p = out / "per_patient_metrics.csv"
    done = pd.read_csv(csv_p) if (resume and csv_p.exists()) else pd.DataFrame()
    rows = done.to_dict("records") if len(done) else []
    done_ids = set(done.patient_id) if len(done) else set()
    calib = Calibration() if calibration else None
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
        gt_work = {r: prep.grid.to_work(m.astype(np.uint8), order=0).astype(bool)
                   for r, m in regions_from_labels(labels, case["ncr"], case["ed"], case["et"]).items()}
        res = evaluate_case(masks, gt_work, voxel_sizes(prep.work_affine))
        for r in REGIONS:
            row = {"patient_id": case["id"], "dataset": case["dataset"], "region": r, **res[r]}
            if measure_errors:
                row.update(measurement_errors(masks[r], gt_work[r], prep.work_affine))
            row["seconds"] = round(time.time() - t0, 1)
            rows.append(row)
        if calib is not None:
            calib.update(pred.probs, {r: gt_work[r][prep.crop] for r in REGIONS}, stride=2)
        pd.DataFrame(rows).to_csv(csv_p, index=False)
        print(f"  [{k}/{len(files)}] {case['id']}  Dice " + " ".join(f"{r}={res[r]['dice']:.3f}" for r in REGIONS) + f"  ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    recs = df.to_dict("records")
    summary = summarize_metrics(recs, metrics=METRICS, seed=seed, n_boot=n_boot)
    by_dataset = {ds: summarize_metrics(df[df.dataset == ds].to_dict("records"), metrics=("dice", "hd95", "iou"), seed=seed, n_boot=n_boot)
                  for ds in sorted(df.dataset.unique())} if "dataset" in df else {}
    strata = {r: size_strata(recs, r, SIZE_EDGES_CM3[r]) for r in REGIONS}
    acc = check_acceptance(summary)

    # --- evidence level (research validation, not clinical) ---
    final = split in FINAL_SPLITS
    distinct, hist_models = _register_use(manifest, split, info["sha256"], len(ids)) if final else (0, 0)
    card = info.get("card", {})
    trained_here = bool(card.get("manifest_sha256")) and card["manifest_sha256"] == man["file_sha256"]
    trained_on = set(card.get("datasets", []))
    ext_ds = {d for d in df.dataset.unique()} if split == "external" else set()
    if split == "external":
        independent = not (ext_ds & trained_on) and (trained_here or attest_trained_on_manifest or bool(trained_on))
    else:
        independent = (trained_here or attest_trained_on_manifest)
    untouched = final and split != "test_legacy" and independent and distinct <= 1 and hist_models <= 1
    level, why = classify_evidence(heldout_eval=final, patient_level_split=True, test_untouched=untouched, consistent_pipeline=True, ci_reported=True,
                                   criteria_passed=acc["passed"], external_eval_passed=(split == "external" and acc["passed"] and untouched))
    notes = []
    if not final:
        notes.append(f"evaluated on '{split}' (used for model selection), not a final held-out split")
    if split == "test_legacy":
        notes.append("test_legacy was used in the notebook to compare models: not an untouched benchmark")
    if hist_models > 1:
        notes.append(f"this split was already used to compare {hist_models} models before (see {_registry_path(manifest).name}): selection bias is possible")
    if final and not independent:
        notes.append("training provenance not recorded: cannot verify these patients were unseen (re-run with --assert-trained-on-manifest only if you KNOW "
                     "the checkpoint was trained/selected on this manifest's train/val patients only)")
    elif final and attest_trained_on_manifest and not trained_here:
        notes.append("independence from the training data is ATTESTED BY THE USER, not verified from the checkpoint")
    if distinct > 1:
        notes.append(f"{distinct} different checkpoints have been evaluated on this split in this registry: selection bias is possible")
    if split == "external":
        notes.append("external = a patient cohort that was never trained on; no tuning was done on it")
    evaluation = {
        "evidence_level": level, "why": why + notes, "criteria_passed": acc["passed"], "acceptance": acc, "split": split, "n_test": len(ids),
        "split_manifest": str(manifest), "split_manifest_sha256": man["file_sha256"], "checkpoint_sha256": info["sha256"], "tta": tta, "overlap": overlap,
        "hierarchy": hierarchy, "min_component_mm3": min_component_mm3, "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "criteria_note": DEFAULT_CRITERIA["note"], "datasets_evaluated": sorted(df.dataset.unique().tolist()) if "dataset" in df else [],
        "external_validation": "evaluated on an external cohort" if split == "external" else "none available - not evaluated on external data",
        "summary": {r: {"dice": round(summary[r]["metrics"]["dice"]["mean"], 4),
                        "dice_ci95": [round(summary[r]["metrics"]["dice"]["ci_low"], 4), round(summary[r]["metrics"]["dice"]["ci_high"], 4)],
                        "hd95_median_mm": round(summary[r]["metrics"]["hd95_raw"]["median"], 2) if summary[r]["metrics"]["hd95_raw"]["n"] else None,
                        "missed_rate": round(summary[r]["missed_rate"], 4)} for r in summary},
    }
    result = {"evaluation": evaluation, "metrics": summary, "by_dataset": by_dataset, "size_strata_cm3": strata, "calibration": calib.result() if calib else None,
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
        evals = read_card(checkpoint).get("evaluations", {})
        evals[split] = evaluation
        update_card(checkpoint, evaluations=evals, evaluation=strongest(evals))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", default="")
    ap.add_argument("--data-roots", nargs="*", metavar="NAME=PATH")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test", choices=list(SPLITS))
    ap.add_argument("--out", default="runs/eval")
    ap.add_argument("--tta", action="store_true", help="4-pass flip TTA (4x slower)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--roi", type=int, default=128, help="sliding-window cube size (model was trained with 128)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-hierarchy", action="store_true")
    ap.add_argument("--no-measure-errors", action="store_true")
    ap.add_argument("--assert-trained-on-manifest", action="store_true")
    ap.add_argument("--no-card-update", action="store_true")
    a = ap.parse_args(argv)
    roots = dict(s.split("=", 1) for s in a.data_roots) if a.data_roots else None
    r = evaluate(a.checkpoint, a.data_root, a.manifest, a.split, a.out, "flips4" if a.tta else "none", a.limit, a.device, hierarchy=not a.no_hierarchy,
                 attest_trained_on_manifest=a.assert_trained_on_manifest, update_checkpoint_card=not a.no_card_update, roi=(a.roi,) * 3, data_roots=roots,
                 measure_errors=not a.no_measure_errors)
    print(json.dumps(r["evaluation"], indent=1))


if __name__ == "__main__":
    main()

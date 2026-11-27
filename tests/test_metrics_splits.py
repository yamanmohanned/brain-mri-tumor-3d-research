"""Evaluation metrics, statistics, calibration, acceptance logic and patient-level splitting."""
import json

import numpy as np
import pytest
from scipy.spatial import cKDTree

from braintumor.metrics import (Calibration, bootstrap_ci, brats_labels_to_regions, check_acceptance, classify_evidence, evaluate_case,
                                lesion_stats, region_metrics, size_strata, summarize_metrics, surface_distances, DEFAULT_CRITERIA)
from braintumor.splits import (SplitError, brats2020_ids, kfold_splits, legacy_split, load_manifest, save_manifest, stratified_split,
                               verify_split)

SP = (1.0, 1.0, 1.0)


def cube(shape=(40, 40, 40), lo=10, hi=20, shift=(0, 0, 0)):
    m = np.zeros(shape, bool)
    m[lo + shift[0]:hi + shift[0], lo + shift[1]:hi + shift[1], lo + shift[2]:hi + shift[2]] = True
    return m


def test_identical_masks_are_perfect():
    m = region_metrics(cube(), cube(), SP, 373.0)
    assert m["dice"] == m["iou"] == m["precision"] == m["recall"] == 1.0
    assert m["hd95"] == 0.0 and m["assd"] == 0.0 and m["sdice_1mm"] == 1.0


def test_known_overlap_values():
    a, b = cube(), cube(shift=(5, 0, 0))                     # half-overlapping 10-voxel cubes
    m = region_metrics(a, b, SP, 373.0)
    assert m["dice"] == pytest.approx(0.5) and m["iou"] == pytest.approx(1 / 3)
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(0.5)
    assert m["specificity"] == pytest.approx(1 - 500 / (40 ** 3 - 1000))
    assert m["f1"] == m["dice"]
    assert m["vol_error_pct"] == pytest.approx(0.0)


@pytest.mark.parametrize("spacing", [(1, 1, 1), (2.0, 1.0, 0.5)])
def test_surface_distances_match_independent_kdtree(spacing):
    a, b = cube(), cube(lo=12, hi=23, shift=(1, -2, 0))
    d_ab, d_ba = surface_distances(a, b, spacing)
    from braintumor.metrics import _surface
    pa = np.argwhere(_surface(a)) * spacing
    pb = np.argwhere(_surface(b)) * spacing
    ref_ab = cKDTree(pb).query(pa)[0]
    ref_ba = cKDTree(pa).query(pb)[0]
    assert np.allclose(np.sort(d_ab), np.sort(ref_ab), atol=1e-6)
    assert np.allclose(np.sort(d_ba), np.sort(ref_ba), atol=1e-6)


def test_hd95_scales_with_physical_spacing():
    a, b = cube(), cube(shift=(4, 0, 0))
    h1 = region_metrics(a, b, (1, 1, 1), 999)["hd95"]
    h2 = region_metrics(a, b, (2, 1, 1), 999)["hd95"]
    assert h2 > h1 * 1.5          # distances along the stretched axis double


def test_empty_case_conventions_do_not_hide_failures():
    z = np.zeros((20, 20, 20), bool)
    both = region_metrics(z, z, SP, 373.0)
    assert both["dice"] == 1.0 and both["hd95"] == 0.0 and np.isnan(both["recall"])
    missed = region_metrics(z, cube((20, 20, 20), 5, 10), SP, 373.0)       # prediction empty, tumour present
    assert missed["dice"] == 0.0 and missed["hd95"] == 373.0 and np.isnan(missed["hd95_raw"]) and missed["pred_empty"]
    fp = region_metrics(cube((20, 20, 20), 5, 10), z, SP, 373.0)           # tumour absent, something predicted
    assert fp["dice"] == 0.0 and fp["hd95"] == 373.0 and fp["gt_empty"] and np.isnan(fp["vol_error_pct"])


def test_lesion_wise_detection():
    gt = np.zeros((30, 30, 30), bool)
    gt[2:6, 2:6, 2:6] = True
    gt[20:26, 20:26, 20:26] = True
    pred = np.zeros_like(gt)
    pred[3:6, 3:6, 3:6] = True          # finds lesion 1
    pred[12:14, 2:4, 2:4] = True        # a false-positive lesion
    s = lesion_stats(pred, gt)
    assert s == {"n_gt_lesions": 2, "n_gt_detected": 1, "n_pred_lesions": 2, "n_fp_lesions": 1}


def test_brats_label_conversion_matches_monai_definition():
    lab = np.zeros((1, 1, 4), int)
    lab[0, 0] = [1, 2, 4, 0]
    r = brats_labels_to_regions(lab)
    assert r["TC"][0, 0].tolist() == [True, False, True, False]
    assert r["WT"][0, 0].tolist() == [True, True, True, False]
    assert r["ET"][0, 0].tolist() == [False, False, True, False]


def test_bootstrap_ci_properties():
    rng = np.random.default_rng(0)
    x = rng.normal(0.8, 0.1, 60)
    ci = bootstrap_ci(x, seed=1)
    assert ci["ci_low"] < ci["mean"] < ci["ci_high"] and ci == bootstrap_ci(x, seed=1)     # reproducible
    assert (ci["ci_high"] - ci["ci_low"]) < 0.1
    const = bootstrap_ci([0.7] * 10)
    assert const["ci_low"] == const["ci_high"] == pytest.approx(0.7)
    nan = bootstrap_ci([np.nan, 0.5, 0.7])
    assert nan["n"] == 2                                                                    # undefined values excluded and counted
    assert bootstrap_ci([np.nan])["n"] == 0


def _rows(n=30, seed=0, miss_every=None):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        for r in ("TC", "WT", "ET"):
            gt = cube()
            pred = cube(shift=tuple(rng.integers(-1, 2, 3)))
            if miss_every and i % miss_every == 0 and r == "ET":
                pred = np.zeros_like(gt)
            m = region_metrics(pred, gt, SP, 373.0)
            rows.append({"patient_id": f"p{i}", "region": r, **m})
    return rows


def test_summary_and_acceptance_flow():
    rows = _rows()
    s = summarize_metrics(rows, n_boot=300)
    assert set(s) == {"TC", "WT", "ET"} and s["WT"]["n_patients"] == 30 and s["WT"]["missed_rate"] == 0
    assert s["WT"]["metrics"]["dice"]["mean"] > 0.7
    acc = check_acceptance(s)
    assert isinstance(acc["passed"], bool) and len(acc["checks"]) == 3 + 3 + 3 + 1
    bad = summarize_metrics(_rows(miss_every=3), n_boot=300)
    assert bad["ET"]["missed_rate"] > 0.3 and not check_acceptance(bad)["passed"]
    assert not any(c["pass"] for c in check_acceptance(bad)["checks"] if "ET missed" in c["check"])


def test_size_strata_small_lesion_analysis():
    rows = _rows(6)
    out = size_strata(rows, "WT", (0, 0.5, 5, float("inf")))
    assert len(out) == 1 and out[0]["n"] == 6 and out[0]["range_cm3"] == [0.5, 5]     # every phantom tumour is 1 cm3; empty bins are omitted


def test_calibration_distinguishes_calibrated_from_overconfident():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, (3, 20, 20, 20)).astype(np.float32)
    y = {r: rng.uniform(0, 1, (20, 20, 20)) < p[i] for i, r in enumerate(("TC", "WT", "ET"))}
    good = Calibration()
    good.update(p, y)
    over = Calibration()
    over.update(np.where(p > 0.5, 0.99, 0.01).astype(np.float32), y)
    assert good.result()["WT"]["ece"] < 0.05 < over.result()["WT"]["ece"]
    assert 0 <= good.result()["WT"]["brier"] <= 1


def test_evidence_levels_follow_the_evidence():
    kw = dict(heldout_eval=True, patient_level_split=True, test_untouched=True, consistent_pipeline=True, ci_reported=True)
    assert classify_evidence(**{**kw, "heldout_eval": False}, criteria_passed=None)[0] == "Not validated"
    assert classify_evidence(**kw, criteria_passed=False)[0] == "Experimental / preliminary"
    assert classify_evidence(**{**kw, "ci_reported": False}, criteria_passed=True)[0] == "Experimental / preliminary"
    assert classify_evidence(**kw, criteria_passed=True)[0] == "Validated on an internal held-out dataset"
    assert classify_evidence(**kw, criteria_passed=True, external_eval_passed=True)[0] == "Independently evaluated on external data"
    assert "RESEARCH ASSUMPTIONS" in DEFAULT_CRITERIA["note"]


# ----------------------------------------------------------------------------- splits

def test_legacy_split_reproduces_notebook_sizes_and_known_member():
    ids = brats2020_ids()
    assert len(ids) == 368
    s = legacy_split(ids)
    assert {k: len(v) for k, v in s.items()} == {"train": 257, "val": 55, "test": 56}   # printed by the notebook
    assert "BraTS20_Training_283" in s["test"]                                         # the notebook's unseen demo patient
    assert s == legacy_split(ids[::-1])                                                # independent of input order


def test_verify_split_detects_leakage_and_duplicate_groups():
    verify_split({"train": ["a", "b"], "val": ["c"], "test": ["d"]})
    with pytest.raises(SplitError, match="LEAKAGE"):
        verify_split({"train": ["a", "b"], "val": ["b"], "test": ["d"]})
    with pytest.raises(SplitError, match="duplicate patient ids"):
        verify_split({"train": ["a", "a"], "val": [], "test": []})
    with pytest.raises(SplitError, match="near-duplicate"):
        verify_split({"train": ["a"], "val": ["b"], "test": ["c"]}, duplicate_groups=[["a", "b"]])


def test_manifest_roundtrip_and_tamper_detection(tmp_path):
    s = stratified_split([f"p{i:03d}" for i in range(100)], seed=7)
    p = tmp_path / "m.json"
    man = save_manifest(p, s, method="test", seed=7, dataset="synthetic")
    assert load_manifest(p)["splits"] == s and man["counts"] == {"train": 70, "val": 15, "test": 15}
    raw = json.loads(p.read_text())
    raw["splits"]["train"].append("p_new")
    raw["splits"]["test"].pop()
    p.write_text(json.dumps(raw))
    with pytest.raises(SplitError, match="hash"):
        load_manifest(p)


def test_stratified_split_balances_strata_and_is_deterministic():
    ids = [f"p{i:03d}" for i in range(120)]
    strata = {i: n % 3 for n, i in enumerate(ids)}
    s1, s2 = stratified_split(ids, strata, seed=3), stratified_split(ids, strata, seed=3)
    assert s1 == s2 and s1 != stratified_split(ids, strata, seed=4)
    for part in s1.values():
        counts = np.bincount([strata[i] for i in part], minlength=3)
        assert counts.max() - counts.min() <= 2


def test_kfold_is_disjoint_and_covers_everyone():
    ids = [f"p{i:03d}" for i in range(53)]
    folds = kfold_splits(ids, k=5, seed=1)
    all_val = sum((f["val"] for f in folds), [])
    assert sorted(all_val) == sorted(ids)
    for f in folds:
        verify_split(f)

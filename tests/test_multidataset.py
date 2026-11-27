"""Multi-dataset registry, label mapping, cross-dataset duplicates, combined split, evaluation extensions, cloud-session training behaviour.

Synthetic phantoms only: these tests verify CODE behaviour (leakage rules, conservative evidence labels, resume), not model accuracy.
"""
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from braintumor.audit import audit_dataset, cross_duplicate_pairs, save_audit
from braintumor.data import discover_cases, discover_many
from braintumor.datasets import DatasetError, check_label_mapping, get_spec, load_registry, regions_from_labels
from braintumor.measure import measurement_errors
from braintumor.metrics import brats_labels_to_regions
from braintumor.splits import SplitError, build_combined_split, load_manifest, save_manifest, stratified_split, verify_split
from tests.phantom import make_brats_like_dataset

FILES = {"t1": ["*_t1.nii*"], "t1ce": ["*_t1ce.nii*"], "t2": ["*_t2.nii*"], "flair": ["*_flair.nii*"], "seg": ["*_seg.nii*"]}


# ----------------------------------------------------------------------------- registry and labels

def test_shipped_registry_gates_unverified_datasets():
    reg = load_registry()
    assert reg["brats2020"].verified and reg["brats2020"].label_ints == (1, 2, 4) and reg["brats2020"].role == "pool"
    for name in ("brats2021", "ucsf_pdgm", "upenn_gbm"):
        with pytest.raises(DatasetError):
            reg[name].require_usable()
    assert reg["ucsf_pdgm"].role == "external" and reg["ucsf_pdgm"].labels is None          # TCIA page gives no integer labels: must not be assumed
    with pytest.raises(DatasetError, match="NOT VERIFIED|not VERIFIED"):
        discover_cases("E:/does/not/matter", reg["brats2021"], require_usable=True)


def test_region_conversion_equals_monai_and_supports_other_conventions():
    pytest.importorskip("monai")
    from monai.transforms import ConvertToMultiChannelBasedOnBratsClasses
    from braintumor.data import _regions_transform
    lab = np.random.default_rng(0).choice([0, 1, 2, 4], size=(1, 9, 9, 9)).astype(np.float32)
    ref = np.asarray(ConvertToMultiChannelBasedOnBratsClasses()(torch.from_numpy(lab)))
    mine = np.asarray(_regions_transform()(keys="label")({"label": torch.from_numpy(lab), "ncr": 1, "ed": 2, "et": 4})["label"])
    assert np.array_equal(ref, mine)
    lab3 = np.where(lab == 4, 3, lab)                                                        # BraTS-2023-style ET=3
    r3 = regions_from_labels(lab3[0], 1, 2, 3)
    r4 = brats_labels_to_regions(lab[0])
    assert all(np.array_equal(r3[k], r4[k]) for k in r4)
    # the WRONG mapping silently changes the regions - which is why mappings are declared per dataset and verified
    assert not np.array_equal(regions_from_labels(lab3[0], 1, 2, 4)["ET"], r4["ET"])


def test_check_label_mapping_reports_conflicts():
    spec = get_spec("brats2020")
    assert check_label_mapping(spec, {0, 1, 2, 4})["consistent"]
    bad = check_label_mapping(spec, {0, 1, 2, 3})
    assert not bad["consistent"] and bad["undeclared_values"] == [3]
    assert not check_label_mapping(get_spec("ucsf_pdgm"), {0, 1, 2, 4})["consistent"]


# ----------------------------------------------------------------------------- fixtures: three phantom "datasets"

def _spec_entry(glob, role, labels=None):
    return {"case_glob": glob, "files": FILES, "labels": labels or {"NCR": 1, "ED": 2, "ET": 4}, "role": role, "verified": True, "description": "phantom"}


def _clone_case(src_root: Path, src_id: str, dst_root: Path, dst_id: str):
    shutil.copytree(src_root / src_id, dst_root / dst_id)
    for f in (dst_root / dst_id).iterdir():
        f.rename(f.with_name(f.name.replace(src_id, dst_id)))


@pytest.fixture
def world(tmp_path, monkeypatch):
    reg = {"poolA": _spec_entry("A_*", "pool"), "poolB": _spec_entry("B_*", "pool"), "extC": _spec_entry("C_*", "external"),
           "brats2020": yaml.safe_load(open(Path(__file__).resolve().parents[1] / "configs" / "datasets.yaml"))["brats2020"]}
    (tmp_path / "reg.yaml").write_text(yaml.safe_dump(reg))
    monkeypatch.setenv("BRAINTUMOR_REGISTRY", str(tmp_path / "reg.yaml"))
    roots = {}
    for name, prefix, n, seed in (("poolA", "A_", 12, 1), ("poolB", "B_", 8, 2), ("extC", "C_", 8, 3)):
        roots[name] = tmp_path / name
        make_brats_like_dataset(roots[name], n=n, shape=(36, 40, 32), seed=seed, prefix=prefix)
    _clone_case(roots["poolA"], "A_001", roots["poolB"], "B_099")      # the same patient appears in two pool datasets
    _clone_case(roots["poolA"], "A_002", roots["extC"], "C_099")       # ... and in the external cohort
    audits = {}
    for name in roots:
        a = audit_dataset(roots[name], get_spec(name), workers=2)
        save_audit(a, tmp_path / f"audit_{name}")
        audits[name] = tmp_path / f"audit_{name}"
    ids_a = sorted(r["id"] for r in audit_dataset(roots["poolA"], get_spec("poolA"), workers=2)["cases"])
    legacy = stratified_split(ids_a, seed=5)
    save_manifest(tmp_path / "legacy.json", legacy, method="test", seed=5, dataset="poolA")
    return {"tmp": tmp_path, "roots": roots, "audits": audits, "legacy": legacy}


def _cli_split(w, out, extra=()):
    from braintumor.__main__ import main
    args = ["split-combined", *sum((["--audit", f"{n}={d}"] for n, d in w["audits"].items()), []), "--legacy", str(w["tmp"] / "legacy.json"),
            "--out", str(out), "--dup-corr", "0.9999", *extra]
    main(args)
    return load_manifest(out)


def test_audit_persists_fingerprints_and_finds_cross_dataset_duplicates(world):
    from braintumor.audit import load_audit_bundle
    bundles = {n: load_audit_bundle(d) for n, d in world["audits"].items()}
    pairs = cross_duplicate_pairs(bundles, 0.9999)
    got = {frozenset(p["pair"]) for p in pairs}
    assert frozenset({"A_001", "B_099"}) in got and frozenset({"A_002", "C_099"}) in got
    assert len(got) == 2 and all(p["cross_dataset"] for p in pairs)                       # different phantoms are NOT flagged
    assert all(p["reason"] in ("fingerprint", "identical file hash") for p in pairs)


def test_combined_split_keeps_legacy_groups_duplicates_and_protects_external(world, capsys):
    man = _cli_split(world, world["tmp"] / "combined.json")
    sp, audit = man["splits"], man["split_audit"]
    legacy = world["legacy"]
    # legacy assignments are preserved; the legacy test patients are labelled test_legacy (they are not an untouched benchmark)
    assert set(legacy["train"]) <= set(sp["train"]) and set(legacy["val"]) <= set(sp["val"]) and set(legacy["test"]) == set(sp["test_legacy"])
    # the duplicate of A_001 stays in the same split as A_001, never separated
    where = {i: k for k, v in sp.items() for i in v}
    assert where["B_099"] == where["A_001"]
    # an external patient that is also a pool patient is removed from the external cohort
    assert "C_099" not in where and any(d["id"] == "C_099" for d in audit["external_removed_for_overlap"])
    assert set(sp["external"]) <= {f"C_{i:03d}" for i in range(1, 9)} and len(sp["external"]) == 8
    # zero overlap across ALL splits, and the manifest re-verifies from disk
    assert audit["patient_overlap_between_splits"] == 0
    verify_split(sp)
    allids = sum(sp.values(), [])
    assert len(allids) == len(set(allids)) == 12 + 9 + 8                                  # A=12, B=8+clone, external C=8+clone minus the overlap
    assert sum(audit["per_split"][k]["n"] for k in sp) == len(allids)
    assert audit["per_split"]["external"]["by_dataset"] == {"extC": 8}
    assert set(audit["per_split"]["train"]["by_dataset"]) <= {"poolA", "poolB"}


def test_combined_split_is_deterministic_and_seed_sensitive(world):
    a = _cli_split(world, world["tmp"] / "c1.json", ["--seed", "7"])["splits"]
    b = _cli_split(world, world["tmp"] / "c2.json", ["--seed", "7"])["splits"]
    c = _cli_split(world, world["tmp"] / "c3.json", ["--seed", "8"])["splits"]
    assert a == b and a["test"] != c["test"] or a["val"] != c["val"] or a["train"] != c["train"]


def test_label_convention_mismatch_blocks_split_building(world):
    from braintumor.__main__ import main
    for f in (world["roots"]["poolB"]).glob("*/*_seg.nii.gz"):                           # a dataset whose ET is coded 3 but declared 4
        img = nib.load(f)
        d = np.asarray(img.dataobj).copy()
        d[d == 4] = 3
        nib.save(nib.Nifti1Image(d.astype(np.int16), img.affine), f)
    a = audit_dataset(world["roots"]["poolB"], get_spec("poolB"), workers=2)
    assert not a["summary"]["label_mapping_check"]["consistent"] and 3 in a["summary"]["label_mapping_check"]["undeclared_values"]
    save_audit(a, world["audits"]["poolB"])
    with pytest.raises(SystemExit, match="label mapping inconsistent"):
        _cli_split(world, world["tmp"] / "never.json")
    assert not (world["tmp"] / "never.json").exists()


def test_permissive_duplicate_threshold_is_refused():
    cases = [{"id": f"p{i}", "dataset": "d", "size_stratum": 0, "problems": [], "gt_WT_cm3": 10.0} for i in range(30)]
    pairs = [{"pair": [f"p{i}", f"p{i + 1}"]} for i in range(29)]                         # everything chained into one group
    with pytest.raises(SplitError, match="too permissive"):
        build_combined_split(cases, pairs=pairs)


def test_build_combined_split_properties_on_records():
    rng = np.random.default_rng(0)
    cases = [{"id": f"A{i:03d}", "dataset": "A", "size_stratum": int(i % 3), "problems": [], "gt_WT_cm3": float(rng.uniform(5, 150))} for i in range(90)] \
        + [{"id": f"B{i:03d}", "dataset": "B", "size_stratum": int(i % 3), "problems": [], "gt_WT_cm3": float(rng.uniform(5, 150))} for i in range(60)] \
        + [{"id": f"X{i:03d}", "dataset": "X", "size_stratum": 0, "problems": [], "gt_WT_cm3": 30.0} for i in range(20)] \
        + [{"id": "BAD", "dataset": "B", "size_stratum": 0, "problems": ["unreadable: truncated"], "gt_WT_cm3": 1.0}]
    legacy = {f"A{i:03d}": ("train" if i < 60 else "val" if i < 75 else "test") for i in range(90)}
    sp, info = build_combined_split(cases, seed=3, legacy=legacy, external_datasets=["X"])
    assert len(sp["test_legacy"]) == 15 and len(sp["external"]) == 20 and "BAD" not in sum(sp.values(), [])
    assert [r["id"] for r in info["rejected"]] == ["BAD"]
    nb = len(sp["train"]) - 60 + len(sp["val"]) - 15 + len(sp["test"])
    assert nb == 60                                                                          # every new pool patient placed exactly once
    new_test = [i for i in sp["test"] if i.startswith("B")]
    assert 7 <= len(new_test) <= 11                                                          # ~15 % of 60, stratified
    assert not set(sp["external"]) & set(sum((v for k, v in sp.items() if k != "external"), []))
    assert any("not an untouched benchmark" in n for n in info["notes"])


# ----------------------------------------------------------------------------- measurement errors, comparison, evaluation rules

def test_measurement_errors_against_known_geometry():
    from tests.phantom import ellipsoid
    aff = np.diag([1.0, 1.0, 1.0, 1.0])
    gt = ellipsoid((60, 60, 60), (1, 1, 1), (30, 30, 30), (14, 11, 9))
    zero = measurement_errors(gt, gt, aff)
    assert all(abs(v) < 1e-9 for v in zero.values())
    shifted = np.roll(gt, 3, axis=0)
    e = measurement_errors(shifted, gt, aff)
    assert e["m_centroid_dist_mm"] == pytest.approx(3.0, abs=0.01) and abs(e["m_vol_err_cm3"]) < 1e-9
    from scipy import ndimage
    big = ndimage.binary_dilation(gt)
    e = measurement_errors(big, gt, aff)
    assert e["m_vol_err_cm3"] > 0 and 1.0 < e["m_max_diam_err_mm"] < 3.5 and e["m_surface_area_err_cm2"] > 0 and e["m_vol_abs_rel_err_pct"] > 5
    assert measurement_errors(np.zeros_like(gt), gt, aff) == {}


def _fake_eval_dir(path, dice_shift, conditions, n=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        base = rng.uniform(0.6, 0.9)
        for r in ("TC", "WT", "ET"):
            rows.append({"patient_id": f"p{i}", "region": r, "dice": min(base + dice_shift + rng.normal(0, 0.01), 1), "hd95_raw": 5 - 10 * dice_shift,
                         "gt_empty": False, "pred_empty": False})
    path.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(path / "per_patient_metrics.csv", index=False)
    (path / "summary.json").write_text(json.dumps({"evaluation": conditions}))


def test_paired_comparison_detects_improvement_and_flags_unlike_conditions(tmp_path):
    from braintumor.compare import compare
    cond = {"split": "val", "split_manifest_sha256": "abc", "tta": "none", "overlap": 0.5, "hierarchy": True, "min_component_mm3": 0.0}
    _fake_eval_dir(tmp_path / "A", 0.0, cond)
    _fake_eval_dir(tmp_path / "B", 0.05, cond)
    r = compare(tmp_path / "A", tmp_path / "B", n_boot=300)
    d = r["regions"]["WT"]["dice"]
    assert r["identical_evaluation_conditions"] and r["n_common_patients"] == 30
    assert d["mean_diff_B_minus_A"] == pytest.approx(0.05, abs=0.01) and d["ci_excludes_zero"] and d["wilcoxon_p"] < 0.001
    assert r["regions"]["WT"]["hd95_raw"]["mean_diff_B_minus_A"] < 0                          # lower is better: B improved
    _fake_eval_dir(tmp_path / "C", 0.05, {**cond, "tta": "flips4"})
    assert compare(tmp_path / "A", tmp_path / "C", n_boot=100)["warning"].startswith("evaluation conditions differ")


@pytest.fixture
def random_ckpt(tmp_path):
    from braintumor.model import build_model, save_checkpoint
    torch.manual_seed(0)
    return save_checkpoint(tmp_path / "w" / "m.pth", build_model("segresnet"), "segresnet", {"note": "random weights"})


def test_evaluate_external_and_legacy_test_are_never_labelled_validated(world, random_ckpt):
    from braintumor.evaluate import evaluate
    from braintumor.model import read_card
    man_p = world["tmp"] / "combined.json"
    _cli_split(world, man_p)
    # pretend the notebook's historical comparison is recorded for the legacy manifest of this experiment
    reg_p = man_p.with_suffix(".test_registry.json")
    reg_p.write_text(json.dumps({"entries": [], "historical": {"test_legacy": [{"description": "notebook comparison", "n_models": 2}]}}))
    roots = {n: str(r) for n, r in world["roots"].items()}
    kw = dict(device="cpu", roi=(32, 32, 32), n_boot=100, data_roots=roots, limit=2, measure_errors=True, attest_trained_on_manifest=True)
    r = evaluate(str(random_ckpt), manifest=str(man_p), split="test_legacy", out_dir=str(world["tmp"] / "ev_leg"), **kw)
    assert r["evaluation"]["evidence_level"] in ("Not validated", "Experimental / preliminary")
    assert any("not an untouched benchmark" in w for w in r["evaluation"]["why"])
    ext = evaluate(str(random_ckpt), manifest=str(man_p), split="external", out_dir=str(world["tmp"] / "ev_ext"), **kw)
    assert ext["evaluation"]["external_validation"].startswith("evaluated on an external cohort") and ext["evaluation"]["evidence_level"] != "Independently evaluated on external data"
    card = read_card(random_ckpt)
    assert set(card["evaluations"]) == {"test_legacy", "external"} and card["evaluation"]["evidence_level"] in ("Experimental / preliminary", "Not validated")
    df = pd.read_csv(world["tmp"] / "ev_ext" / "per_patient_metrics.csv")
    assert {"dataset", "m_vol_abs_err_cm3", "m_max_diam_abs_err_mm", "m_surface_area_abs_err_cm2", "m_centroid_dist_mm"} <= set(df.columns)
    assert set(df.dataset) == {"extC"} and "by_dataset" in ext and "extC" in ext["by_dataset"]


def test_historical_registry_marks_the_shipped_legacy_test_split_as_not_untouched():
    reg = json.loads((Path(__file__).resolve().parents[1] / "splits" / "legacy_seed42.test_registry.json").read_text())
    h = reg["historical"]["test"][0]
    assert h["n_models"] == 2 and "NOT an untouched" in h["consequence"]


# ----------------------------------------------------------------------------- training objectives and cloud-session behaviour

def test_loss_options_and_the_absent_region_gradient_claim():
    from braintumor.train import CombinedLoss
    torch.manual_seed(0)
    target = torch.zeros(1, 3, 12, 12, 12)
    target[:, :2, 3:9, 3:9, 3:9] = 1                                                       # ET (channel 2) absent from the patch
    grads = {}
    for kind in ("dice", "dicece", "dicefocal", "gdice"):
        logits = torch.randn(1, 3, 12, 12, 12, requires_grad=True)
        loss = CombinedLoss(kind)(logits, target)
        loss.backward()
        assert torch.isfinite(loss)
        grads[kind] = float(logits.grad[:, 2].abs().sum())
    assert grads["dice"] == 0.0                                                            # the notebook's loss never penalises false-positive ET here
    assert all(grads[k] > 0 for k in ("dicece", "dicefocal", "gdice"))
    with pytest.raises(ValueError):
        CombinedLoss("nope")


def test_time_budget_pauses_cleanly_and_resume_finishes(tmp_path):
    from braintumor import train as T
    ids = make_brats_like_dataset(tmp_path / "data", n=10, shape=(40, 44, 36), seed=9)
    man = tmp_path / "m.json"
    save_manifest(man, stratified_split(ids, seed=1, fractions=(0.6, 0.2, 0.2)), method="t", seed=1, dataset="phantom")
    cfg = T.TrainConfig(data_root=str(tmp_path / "data"), manifest=str(man), out_dir=str(tmp_path / "run"), cache_dir=str(tmp_path / "localcache"), epochs=3,
                        warmup_epochs=1, batch_size=2, accum_steps=1, roi=(32, 32, 32), val_interval=1, num_workers=0, cache=True, allow_cpu=True, device="cpu",
                        loss="dicefocal", max_hours=1e-9)
    s1 = T.run(cfg)
    assert s1["paused_for_time_budget"] and not s1["finished"] and s1["last_epoch"] == 1
    assert (tmp_path / "run" / "last.pth").exists() and (tmp_path / "localcache" / "train").exists() and not (tmp_path / "run" / "cache").exists()
    import dataclasses
    s2 = T.run(dataclasses.replace(cfg, max_hours=0.0), resume=True)
    assert s2["finished"] and s2["last_epoch"] == 3
    lines = [json.loads(x) for x in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert [x["epoch"] for x in lines] == [1, 2, 3] and all(x["val"] is not None for x in lines)
    assert (tmp_path / "run" / "versions").exists() and list((tmp_path / "run" / "versions").glob("best_e*.pth"))
    env = json.loads((tmp_path / "run" / "environment.json").read_text())
    assert env["config_sha256"] and "hardware" in env and "git_commit" in env
    # gradient accumulation keeps the effective batch (2 x 2 = 4 samples per optimiser step) and still trains
    s3 = T.run(dataclasses.replace(cfg, out_dir=str(tmp_path / "run_acc"), batch_size=2, accum_steps=2, epochs=1, max_hours=0.0))
    assert s3["finished"] and s3["last_epoch"] == 1

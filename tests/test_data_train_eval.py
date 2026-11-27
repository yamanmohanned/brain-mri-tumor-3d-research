"""Dataset audit, patient-level training pipeline and held-out evaluation (tiny synthetic BraTS-structured dataset).

These runs only verify the CODE PATHS (the loop runs, checkpoints/resume/early-stop/leakage guards behave, evidence labels are
conservative). The networks here are trained for a few CPU steps on phantoms: no performance claim is made or implied.
"""
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from braintumor.audit import audit_dataset, legacy_split_consistency
from braintumor.data import discover_cases, locate_root, select
from braintumor.model import load_checkpoint, read_card
from braintumor.splits import SplitError, load_manifest, save_manifest, stratified_split
from tests.phantom import make_brats_like_dataset, make_phantom, save_phantom


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("brats") / "BraTS2020_TrainingData" / "MICCAI_BraTS2020_TrainingData"
    ids = make_brats_like_dataset(root, n=14, shape=(48, 52, 40), seed=3)
    return root, ids


def test_nested_kaggle_layout_is_unwrapped(dataset):
    root, ids = dataset
    top = root.parent.parent                       # .../BraTS2020_TrainingData's parent
    assert locate_root(top) == root and locate_root(root.parent) == root
    cases, skipped = discover_cases(top)
    assert sorted(cases) == sorted(ids) and not skipped
    assert [Path(f).name for f in cases[ids[0]]["image"]] == [f"{ids[0]}_{m}.nii.gz" for m in ("t1", "t1ce", "t2", "flair")]


def test_audit_detects_injected_defects(tmp_path):
    ids = make_brats_like_dataset(tmp_path, n=8, shape=(48, 52, 40), seed=5)
    # 1) exact duplicate patient stored under another id
    shutil.copytree(tmp_path / ids[0], tmp_path / "BraTS20_Training_099")
    for f in (tmp_path / "BraTS20_Training_099").iterdir():
        f.rename(f.with_name(f.name.replace(ids[0], "BraTS20_Training_099")))
    # 2) segmentation on a shifted grid
    ph = make_phantom(shape=(48, 52, 40))
    aff = ph["affine"].copy()
    aff[0, 3] += 6
    seg_p = next((tmp_path / ids[1]).glob("*_seg.nii.gz"))
    nib.save(nib.Nifti1Image(np.asarray(nib.load(seg_p).dataobj), aff), seg_p)
    # 3) truncated (corrupt) image
    t2 = next((tmp_path / ids[2]).glob("*_t2.nii.gz"))
    t2.write_bytes(t2.read_bytes()[:200])
    # 4) incomplete case (no seg)
    next((tmp_path / ids[3]).glob("*_seg.nii.gz")).unlink()
    # 5) invalid label value
    seg_p = next((tmp_path / ids[4]).glob("*_seg.nii.gz"))
    s = np.asarray(nib.load(seg_p).dataobj).copy()
    s[0, 0, 0] = 7
    nib.save(nib.Nifti1Image(s.astype(np.int16), nib.load(seg_p).affine), seg_p)

    a = audit_dataset(tmp_path, workers=2)
    sm = a["summary"]
    probs = sm["problem_cases"]
    assert any("mis-aligned" in p for p in probs[ids[1]])
    assert any("unreadable" in p for p in probs[ids[2]])
    assert [s["id"] for s in sm["skipped_incomplete"]] == [ids[3]]
    assert any("unexpected label" in p for p in probs[ids[4]])
    assert any({ids[0], "BraTS20_Training_099"} == set(g) for g in sm["exact_duplicate_groups"])
    assert sm["n_complete"] == 8 and sm["gt_volume_cm3"]["WT"]["max"] > 0 and sm["n_without_ET"] >= 1
    assert sm["shapes"][0][0] == (48, 52, 40) and sm["orientations"][0][0] == "RAS"
    assert len(sm["wt_tertile_edges_cm3"]) == 2


def test_legacy_split_consistency_check_logic():
    audit = {"cases": [{"id": "a", "gt_WT_cm3": 5.0}, {"id": "b", "gt_WT_cm3": 150.0}, {"id": "c", "gt_WT_cm3": 46.9}, {"id": "d", "gt_WT_cm3": 80.0}]}
    ok = legacy_split_consistency(audit, ["a", "b", "c", "d"], first_ids=(), chosen="c")
    assert ok["consistent"]
    bad = legacy_split_consistency(audit, ["d", "a", "c"], chosen="c")        # an earlier patient is inside 40-100 -> the notebook would have picked it
    assert not bad["consistent"]


# ----------------------------------------------------------------------------- training

@pytest.fixture(scope="module")
def trained(dataset, tmp_path_factory):
    from braintumor import train as T
    root, ids = dataset
    work = tmp_path_factory.mktemp("run")
    splits = stratified_split(ids, seed=1, fractions=(0.6, 0.2, 0.2))
    man_p = work / "manifest.json"
    save_manifest(man_p, splits, method="test", seed=1, dataset="synthetic phantoms")
    cfg = T.TrainConfig(data_root=str(root.parent.parent), manifest=str(man_p), out_dir=str(work / "run"), epochs=2, warmup_epochs=1, batch_size=2,
                        roi=(32, 32, 32), val_interval=1, num_workers=0, cache=False, allow_cpu=True, device="cpu", patience=5)
    seen = []
    orig = select

    import braintumor.data as D
    def spy(cases, id_list):
        seen.extend(id_list)
        return orig(cases, id_list)
    D.select = spy
    try:
        status = T.run(cfg)
    finally:
        D.select = orig
    return {"cfg": cfg, "status": status, "splits": splits, "manifest": man_p, "root": root.parent.parent, "seen": seen, "work": work}


@pytest.mark.slow
def test_training_loop_checkpoints_and_never_touches_test_patients(trained):
    run = Path(trained["cfg"].out_dir)
    for f in ("best.pth", "last.pth", "history.json", "history.png", "config.yaml", "environment.json", "status.json", "best.card.json"):
        assert (run / f).exists(), f
    assert set(trained["seen"]) <= set(trained["splits"]["train"]) | set(trained["splits"]["val"])
    assert not set(trained["seen"]) & set(trained["splits"]["test"])                       # no test patient is ever loaded
    assert not set(trained["splits"]["train"]) & set(trained["splits"]["val"])             # patient-level separation
    hist = json.loads((run / "history.json").read_text())
    assert hist["epoch_train"] == [1, 2] and len(hist["val_mean"]) == 2 and all(0 <= v <= 1 for v in hist["val_mean"])
    model, info = load_checkpoint(run / "best.pth")
    assert info["format"] == "braintumor_v1" and info["card"]["manifest_sha256"] == load_manifest(trained["manifest"])["file_sha256"]
    assert info["card"]["evaluation"] is None and "NOT a test result" in info["card"]["note"]


@pytest.mark.slow
def test_training_guards_refuse_cpu_overwrite_and_resume(trained):
    from braintumor import train as T
    import dataclasses
    cfg = trained["cfg"]
    with pytest.raises(FileExistsError):
        T.run(cfg)                                                                          # existing run is never overwritten
    with pytest.raises(SystemExit, match="Refusing to train on CPU"):
        T.run(dataclasses.replace(cfg, out_dir=str(trained["work"] / "other"), allow_cpu=False))
    more = T.run(dataclasses.replace(cfg, epochs=3), resume=True)                           # resume continues at epoch 3
    hist = json.loads((Path(cfg.out_dir) / "history.json").read_text())
    assert hist["epoch_train"] == [1, 2, 3] and more["last_epoch"] == 3
    est = T.run(dataclasses.replace(cfg, out_dir=str(trained["work"] / "bench")), benchmark=True)["estimate"]
    assert est["seconds_per_step"] > 0 and est["estimated_total_hours"] > 0


@pytest.mark.slow
def test_evaluation_pipeline_outputs_and_conservative_evidence_label(trained):
    from braintumor.evaluate import evaluate
    ckpt = str(Path(trained["cfg"].out_dir) / "best.pth")
    out = Path(trained["work"]) / "eval"
    r = evaluate(ckpt, str(trained["root"]), str(trained["manifest"]), split="test", out_dir=str(out), device="cpu", roi=(32, 32, 32), n_boot=200)
    df = pd.read_csv(out / "per_patient_metrics.csv")
    assert set(df.region) == {"TC", "WT", "ET"} and set(df.patient_id) == set(trained["splits"]["test"])
    for col in ("dice", "iou", "precision", "recall", "specificity", "f1", "hd95", "assd", "sdice_1mm", "gt_empty", "pred_empty", "n_gt_lesions"):
        assert col in df.columns
    s = json.loads((out / "summary.json").read_text())
    assert set(s["metrics"]) == {"TC", "WT", "ET"} and s["calibration"]["WT"]["ece"] >= 0 and "ci_low" in s["metrics"]["WT"]["metrics"]["dice"]
    ev = r["evaluation"]
    assert ev["evidence_level"] == "Experimental / preliminary" and not ev["criteria_passed"]           # 2 CPU epochs on phantoms must not pass criteria
    assert "pre-specified acceptance criteria not met" in ev["why"]
    assert ev["external_validation"].startswith("none")
    assert (out / "failures.csv").exists() and (out / "dice_boxplot.png").exists()
    card = read_card(ckpt)
    assert card["evaluation"]["evidence_level"] == ev["evidence_level"] and card["evaluation"]["n_test"] == len(trained["splits"]["test"])
    reg = json.loads(Path(trained["manifest"]).with_suffix(".test_registry.json").read_text())
    assert len(reg["entries"]) == 1                                                                      # test-set use is recorded
    # evaluating on val is labelled as such, never as held-out validation
    r2 = evaluate(ckpt, str(trained["root"]), str(trained["manifest"]), split="val", out_dir=str(out / "val"), device="cpu", roi=(32, 32, 32), n_boot=100,
                  update_checkpoint_card=False)
    assert r2["evaluation"]["evidence_level"] == "Not validated"


def test_default_yaml_config_is_valid_and_consistent_with_code_defaults():
    import dataclasses
    from braintumor.train import TrainConfig
    cfg = TrainConfig.load(str(Path(__file__).resolve().parents[1] / "configs" / "train.yaml"))
    default = TrainConfig()
    for f in dataclasses.fields(TrainConfig):
        if f.name in ("data_root", "out_dir", "allow_cpu", "device", "max_train_cases", "max_val_cases"):
            continue
        assert getattr(cfg, f.name) == getattr(default, f.name), f.name
    assert cfg.arch in ("segresnet", "unet") and cfg.epochs > 30


def test_shipped_legacy_manifest_is_valid_and_matches_notebook_facts():
    man = load_manifest(Path(__file__).resolve().parents[1] / "splits" / "legacy_seed42.json")
    assert man["counts"] == {"train": 257, "val": 55, "test": 56} and "BraTS20_Training_283" in man["splits"]["test"]

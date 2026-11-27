"""Command-line entry points (in-process): split, audit, headless analyze (annotation mode)."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from braintumor.__main__ import main
from braintumor.splits import load_manifest
from tests.phantom import make_brats_like_dataset, make_phantom, save_phantom


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    r = tmp_path_factory.mktemp("cli") / "BraTS2020_TrainingData" / "MICCAI_BraTS2020_TrainingData"
    make_brats_like_dataset(r, n=12, shape=(40, 44, 36), seed=11)
    return r.parent.parent


def test_split_legacy_and_stratified_manifests(root, tmp_path, capsys):
    main(["split", "--data-root", str(root), "--method", "legacy", "--out", str(tmp_path / "legacy.json")])
    legacy = load_manifest(tmp_path / "legacy.json")
    assert legacy["counts"] == {"train": 8, "val": 1, "test": 3} and legacy["seed"] == 42        # int(.7*12), int(.15*12)
    main(["split", "--data-root", str(root), "--method", "stratified", "--seed", "3", "--out", str(tmp_path / "strat.json"), "--workers", "2"])
    strat = load_manifest(tmp_path / "strat.json")
    assert sum(strat["counts"].values()) == 12 and strat["seed"] == 3 and strat["unit_of_split"].startswith("patient")
    main(["split", "--data-root", str(root), "--method", "stratified", "--seed", "3", "--out", str(tmp_path / "strat2.json"), "--workers", "2"])
    assert load_manifest(tmp_path / "strat2.json")["splits"] == strat["splits"]                   # reproducible


def test_audit_command_writes_summary_and_csv(root, tmp_path, capsys):
    main(["audit", "--data-root", str(root), "--out", str(tmp_path / "audit"), "--workers", "2"])
    s = json.loads((tmp_path / "audit" / "audit_summary.json").read_text())
    assert s["n_complete"] == 12 and s["n_with_problems"] == 0
    assert (tmp_path / "audit" / "audit_cases.csv").exists()


def test_headless_analyze_measures_an_annotation(tmp_path, capsys):
    ph = make_phantom(affine_kind="las", wt_radii=(12, 10, 8))
    f = save_phantom(ph, tmp_path / "in", "CASE_7")
    main(["analyze", "--t1", str(f["t1"]), "--t1ce", str(f["t1ce"]), "--t2", str(f["t2"]), "--flair", str(f["flair"]), "--seg", str(f["seg"]),
          "--out", str(tmp_path / "out"), "--mesh-format", "ply"])
    out = tmp_path / "out"
    assert (out / "CASE_7_seg_annot.nii.gz").exists() and (out / "CASE_7_WT.ply").exists() and (out / "CASE_7_report.html").exists()
    img = nib.load(out / "CASE_7_seg_annot.nii.gz")
    assert np.allclose(img.affine, ph["affine"]) and np.array_equal(np.asarray(img.dataobj), ph["seg"])      # lossless on the original grid
    rep = json.loads((out / "CASE_7_report.json").read_text())
    assert rep["provenance"]["result_type"] == "external annotation"
    with pytest.raises(FileExistsError):
        main(["analyze", "--t1", str(f["t1"]), "--t1ce", str(f["t1ce"]), "--t2", str(f["t2"]), "--flair", str(f["flair"]), "--seg", str(f["seg"]),
              "--out", str(out)])
    with pytest.raises(SystemExit):
        main(["analyze", "--t1", str(f["t1"]), "--t1ce", str(f["t1ce"]), "--t2", str(f["t2"]), "--flair", str(f["flair"]), "--out", str(tmp_path / "x")])

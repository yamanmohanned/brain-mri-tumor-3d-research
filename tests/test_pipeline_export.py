"""Full workflow (load -> preprocess -> segment -> post-process -> mesh -> measure -> report -> export) on synthetic phantoms.

The stand-in network thresholds phantom channels: it validates *plumbing and geometry* (orientation, resampling, spatial
alignment, units, export), NOT segmentation accuracy - a real model needs real weights and real data.
"""
import csv
import json

import nibabel as nib
import numpy as np
import pytest
import torch

from braintumor import MODALITIES, REGIONS
from braintumor.export import export_all, labels_from_masks
from braintumor.geometry import apply_affine
from braintumor.inference import InferenceConfig
from braintumor.io import InputError, MRVolume, load_modalities
from braintumor.metrics import brats_labels_to_regions, region_metrics
from braintumor.pipeline import AnalysisConfig, analyze
from braintumor.report import build_report, render_html
from tests.phantom import ThresholdNet, make_phantom, save_phantom


CFG = AnalysisConfig(infer=InferenceConfig(roi=(64, 64, 64), overlap=0.25, device="cpu"))


def _run(ph, with_axial=False):
    vol = MRVolume(ph["data"], ph["affine"], MODALITIES)
    res = analyze(vol, ThresholdNet(ph["data"]), {"path": "fake.pth", "sha256": "0" * 64, "arch": "stand-in", "card": {}}, CFG, case_id="phantom")
    return vol, res


def _truth_regions(ph):
    return brats_labels_to_regions(ph["seg"])


@pytest.mark.parametrize("kind", ["ras", "las", "lps", "permuted", "oblique"])
def test_pipeline_is_spatially_exact_for_every_orientation(kind, tmp_path):
    ph = make_phantom(affine_kind=kind)
    vol, res = _run(ph)
    truth_w = {r: res.prep.grid.to_work(t.astype(np.uint8), order=0).astype(bool) for r, t in _truth_regions(ph).items()}
    for r in REGIONS:
        m = region_metrics(res.masks[r], truth_w[r], (1, 1, 1), 373.0)
        assert m["dice"] > 0.96, (kind, r, m["dice"])
    # world-space alignment of the predicted tumour with the true one (mm)
    c_pred = apply_affine(res.work_affine, np.argwhere(res.masks["WT"]).mean(0)[None])[0]
    c_true = apply_affine(ph["affine"], np.argwhere(ph["seg"] > 0).mean(0)[None])[0]
    assert np.linalg.norm(c_pred - c_true) < 0.5
    assert abs(res.measurements["WT"]["volume_mm3"] - ph["truth"]["WT_voxels_volume_mm3"]) / ph["truth"]["WT_voxels_volume_mm3"] < 0.03

    # exported mask is on the ORIGINAL grid with the ORIGINAL affine and re-loads onto the scan
    out = export_all(res, tmp_path / "out")
    img = nib.load(out["mask"][0])
    assert img.shape == ph["seg"].shape and np.allclose(img.affine, ph["affine"])
    lab = np.asarray(img.dataobj)
    assert set(np.unique(lab)) <= {0, 1, 2, 4}
    for r, t in _truth_regions(ph).items():
        assert region_metrics(brats_labels_to_regions(lab)[r], t, (1, 1, 1), 373.0)["dice"] > 0.96


def test_anisotropic_scan_is_resampled_measured_in_mm_and_exported_on_original_grid(tmp_path):
    ph = make_phantom(shape=(90, 100, 32), spacing=(0.8, 0.8, 2.0), wt_radii=(16, 12, 12), center_frac=(0.5, 0.5, 0.5))
    vol, res = _run(ph)
    assert res.prep.grid.resampled and np.allclose(np.linalg.norm(res.work_affine[:3, :3], axis=0), 1.0)
    tv = ph["truth"]["WT_volume_mm3"]
    assert abs(res.measurements["WT"]["volume_mm3"] - tv) / tv < 0.08
    out = export_all(res, tmp_path, items=("mask",))
    img = nib.load(out["mask"][0])
    assert img.shape == ph["seg"].shape and np.allclose(img.affine, ph["affine"])
    pred = brats_labels_to_regions(np.asarray(img.dataobj))["WT"]
    assert region_metrics(pred, ph["seg"] > 0, ph["spacing"], 400.0)["dice"] > 0.90
    rep = build_report(res)
    assert any("not 1 mm isotropic" in w for w in rep["8_quality_and_uncertainty"]["warnings"])


def test_input_that_is_not_analysable_stops_instead_of_producing_numbers():
    ph = make_phantom()
    bad = ph["data"].copy()
    bad[2] = 7.0                                              # constant T2
    with pytest.raises(InputError, match="constant"):
        analyze(MRVolume(bad, ph["affine"], MODALITIES), ThresholdNet(ph["data"]), None, CFG)
    bad = ph["data"].copy()
    bad[1] = 0.0                                              # empty T1ce
    with pytest.raises(InputError, match="empty"):
        analyze(MRVolume(bad, ph["affine"], MODALITIES), ThresholdNet(ph["data"]), None, CFG)


def test_external_mask_mode_reports_annotation_not_prediction():
    ph = make_phantom(affine_kind="las")
    vol = MRVolume(ph["data"], ph["affine"], MODALITIES)
    res = analyze(vol, labels=ph["seg"].astype(np.int16), label_source="synthetic ground truth", case_id="gt")
    assert res.source == "external_mask" and res.uncertainty is None
    assert abs(res.measurements["WT"]["volume_mm3"] - ph["truth"]["WT_voxels_volume_mm3"]) < 1e-6
    rep = build_report(res)
    assert rep["provenance"]["result_type"] == "external annotation" and "not a model prediction" in rep["3_model"]["kind"]
    assert "Not applicable" in rep["7_validation"]["evidence_level"]
    with pytest.raises(ValueError):
        analyze(vol)                                                 # neither model nor labels


def test_report_has_all_sections_disclaimer_and_honest_validation_status():
    ph = make_phantom()
    _, res = _run(ph)
    rep = build_report(res)
    for k in ("1_input", "2_modalities", "3_model", "4_segmentation", "5_measurements", "6_reconstruction", "7_validation",
              "8_quality_and_uncertainty", "9_limitations", "10_interpretation"):
        assert k in rep
    assert "NOT A MEDICAL DEVICE" in rep["disclaimer"].upper() or "not a medical device" in rep["disclaimer"].lower()
    assert rep["7_validation"]["evidence_level"] == "Not validated"          # no evaluation card attached
    text = rep["10_interpretation"].lower()
    assert "malignan" in text and "do not establish" in text                    # explicitly no diagnosis
    assert rep["8_quality_and_uncertainty"]["prediction_uncertainty"]["WT"]["volume_range_cm3"][0] <= rep["5_measurements"]["values"]["WT"]["volume_cm3"] + 1e-6
    res.model_info["card"] = {"evaluation": {"evidence_level": "Experimental / preliminary", "why": ["pre-specified acceptance criteria not met"],
                                             "criteria_passed": False, "n_test": 56, "summary": {"WT": {"dice": 0.89, "hd95_median_mm": 4.0}}}}
    rep2 = build_report(res)
    assert rep2["7_validation"]["evidence_level"] == "Experimental / preliminary" and rep2["7_validation"]["n_test_patients"] == 56


def test_html_report_escapes_untrusted_text():
    ph = make_phantom()
    vol = MRVolume(ph["data"], ph["affine"], MODALITIES)
    res = analyze(vol, ThresholdNet(ph["data"]), None, CFG, case_id="<script>alert(1)</script>")
    html_text = render_html(build_report(res))
    assert "<script>alert(1)</script>" not in html_text and "&lt;script&gt;" in html_text


def test_export_outputs_are_consistent_and_never_overwrite(tmp_path):
    import trimesh
    ph = make_phantom(affine_kind="las")
    _, res = _run(ph, with_axial=True)
    out = export_all(res, tmp_path, items=("mask", "meshes", "report", "csv", "images"), mesh_format="ply")
    assert {Path_.suffix for Path_ in map(__import__("pathlib").Path, out["meshes"])} == {".ply"}
    wt = trimesh.load([p for p in out["meshes"] if p.endswith("_WT.ply")][0], process=False)
    assert abs(wt.volume / 1000 - res.measurements["WT"]["mesh_volume_cm3"]) < 1e-3
    rep = json.load(open(out["report"][0]))
    assert rep["5_measurements"]["values"]["WT"]["volume_cm3"] == round(res.measurements["WT"]["volume_cm3"], 1)
    rows = list(csv.DictReader(open(out["csv"][0])))
    assert [r["region"] for r in rows] == list(REGIONS) and float(rows[1]["volume_cm3"]) == pytest.approx(res.measurements["WT"]["volume_cm3"])
    assert (tmp_path / "phantom_slices.png").stat().st_size > 5000 and "<html" in open(out["report"][1], encoding="utf-8").read()[:50].lower() + "<html"
    with pytest.raises(FileExistsError):
        export_all(res, tmp_path, items=("mask",))
    export_all(res, tmp_path, items=("mask",), overwrite=True)


def test_file_based_end_to_end_with_modality_files(tmp_path):
    ph = make_phantom(affine_kind="lps")
    f = save_phantom(ph, tmp_path / "case", "case1")
    vol = load_modalities({m: f[m] for m in MODALITIES})
    res = analyze(vol, ThresholdNet(vol.data), None, CFG, case_id="case1")
    assert res.measurements["WT"]["n_components"] == 1
    assert labels_from_masks(res.masks).dtype == np.uint8

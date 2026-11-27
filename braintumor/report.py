"""Structured, honest per-analysis report (JSON + HTML). Nothing here asserts diagnosis, grade or malignancy."""
from __future__ import annotations

import datetime as _dt
import html
import platform

import numpy as np

from . import MODALITIES, REGION_NAMES, RESEARCH_DISCLAIMER, __version__
from .measure import METHODS, round_for_display
from .pipeline import AnalysisResult, mesh_summary

STATIC_LIMITATIONS = [
    "Research prototype: not a medical device, not clinically validated, not for diagnosis, grading or treatment decisions.",
    "The model was trained on BraTS-style adult glioma MRI (skull-stripped, co-registered, 1 mm isotropic; T1, T1ce, T2, FLAIR). "
    "Behaviour on other tumour types, paediatric, post-operative or differently processed scans is unknown.",
    "A segmentation is a model prediction (or a supplied annotation), not confirmed anatomy; boundaries carry uncertainty of at least about one voxel.",
    "Measurements are made on a 1 mm working grid; lengths are meaningful to roughly +/-1 mm at best and volumes to the boundary-sensitivity range shown, "
    "before any segmentation error is considered.",
    "The 'oedema' label includes non-enhancing signal change that is not necessarily tumour infiltration; the label does not identify tumour type or grade.",
    "Reported validation metrics (if any) describe the model's average behaviour on a research dataset, not the correctness of this particular case.",
]


def _fmt_meas(m: dict) -> dict:
    return round_for_display(m)


def build_report(res: AnalysisResult) -> dict:
    now = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    is_pred = res.source == "model"
    meas = {r: _fmt_meas(m) for r, m in res.measurements.items()}
    wt = meas["WT"]
    card = (res.model_info or {}).get("card", {}) or {}
    ev = card.get("evaluation")

    # --- 3: model / provenance
    if is_pred:
        mi = res.model_info or {}
        model_sec = {"kind": "Model prediction", "checkpoint": mi.get("path"), "sha256": mi.get("sha256"), "architecture": mi.get("arch"),
                     "parameters": mi.get("n_params"), "checkpoint_format": mi.get("format"), "checkpoint_meta": mi.get("meta"),
                     "inference_config": res.config.get("infer")}
    else:
        model_sec = {"kind": "Externally supplied segmentation (not a model prediction)", "source": res.source_detail,
                     "note": "Measurements and the 3D model describe the supplied annotation. No model was run."}

    # --- 7: validation evidence
    if is_pred and ev:
        validation = {"evidence_level": ev.get("evidence_level"), "caveats": ev.get("why", []), "acceptance_criteria_passed": ev.get("criteria_passed"),
                      "n_test_patients": ev.get("n_test"), "split_manifest_sha256": ev.get("split_manifest_sha256"), "summary": ev.get("summary"),
                      "scope": "Population-level performance on a held-out research dataset; says nothing about this scan.",
                      "criteria_note": ev.get("criteria_note")}
    elif is_pred:
        validation = {"evidence_level": "Not validated", "caveats": ["No evaluation record is attached to this checkpoint (no *.card.json)."],
                      "scope": "No held-out performance figures are available for this checkpoint."}
    else:
        validation = {"evidence_level": "Not applicable (no model used)", "caveats": []}

    # --- 8: quality & uncertainty
    warnings = [i.message for i in res.qc if i.severity == "warning"]
    unc = res.uncertainty
    if unc:
        for r in ("WT", "ET"):
            lo, hi = unc[r]["volume_range_cm3"]
            v = res.measurements[r].get("volume_cm3", 0.0)
            if v > 0 and (hi - lo) / max(v, 1e-9) > 0.5:
                warnings.append(f"{r}: predicted volume is highly threshold-sensitive ({lo:.1f}-{hi:.1f} cm3 between probability cut-offs 0.8 and 0.2).")
    if res.measurements["WT"].get("n_components", 0) > 1:
        warnings.append(f"Whole tumour consists of {res.measurements['WT']['n_components']} separate components (multifocal, or fragmented prediction).")
    if not res.measurements["WT"].get("present"):
        warnings.append("No whole-tumour region was found.")
    mw = [r for r, m in res.measurements.items() if m.get("present") and m.get("mesh_watertight") is False]
    if mw:
        warnings.append(f"Surface mesh not watertight for {mw}: surface area / mesh volume are approximate.")

    # --- 10: interpretation (templated, descriptive only)
    who = "The model predicts" if is_pred else "The supplied annotation contains"
    if wt.get("present"):
        lo, hi = wt["volume_cm3_boundary_range"]
        et = res.measurements["ET"]
        interp = (f"{who} a whole-tumour region of about {wt['volume_cm3']} cm3 (a +/-1-voxel boundary shift would give {lo}-{hi} cm3), "
                  f"with a longest 3D dimension of about {wt['max_diameter_mm']} mm (orthogonal L x W x H about "
                  f"{wt['length_mm']} x {wt['width_mm']} x {wt['height_mm']} mm), in {wt['n_components']} connected component(s). "
                  + "An enhancing-tumour region is "
                  + (f"present ({round(et['volume_cm3'], 1)} cm3)." if et.get("present") else "not predicted/present.")
                  + " These are geometric descriptions of the segmentation only; they do not establish tumour type, grade or malignancy.")
    else:
        interp = f"{who} no whole-tumour region. This does not exclude disease: the model may have missed a lesion or the input may be unsuitable."
    if is_pred:
        interp += (" Treat the result as a research output to be reviewed against the images by a qualified specialist; "
                   f"the model's evidence level is: {validation['evidence_level']}.")

    return {
        "title": "Brain MRI tumour segmentation - research analysis report", "generated_utc": now, "case_id": res.case_id,
        "disclaimer": RESEARCH_DISCLAIMER,
        "provenance": {"software_version": __version__, "python": platform.python_version(), "result_type": "model prediction" if is_pred else "external annotation",
                       "timings_s": res.timings},
        "1_input": {**res.volume, "working_grid": {"shape": list(res.prep.work_shape), "spacing_mm": 1.0, "orientation": "RAS",
                                                   "resampled": res.prep.grid.resampled},
                    "foreground_crop_voxels": [s.stop - s.start for s in res.prep.crop]},
        "2_modalities": {"detected": list(res.volume["modalities"]), "required_by_model": list(MODALITIES),
                         "all_present": set(MODALITIES) <= set(res.volume["modalities"])},
        "3_model": model_sec,
        "4_segmentation": {"label": "Model prediction" if is_pred else "Annotation (not a model output)",
                           "regions": {r: {"name": REGION_NAMES[r], "present": res.measurements[r]["present"],
                                           "n_voxels": res.measurements[r]["n_voxels"]} for r in res.measurements},
                           "postprocessing": res.postprocess_log},
        "5_measurements": {"values": meas, "methods": METHODS, "units": "mm / mm2 / cm3 in scanner RAS+ physical space",
                           "precision_note": "Rounded to 1 mm and 0.1 cm3; finer values would overstate the data's resolution."},
        "6_reconstruction": {"method": "marching cubes at iso-level 0.5 on the binary mask, vertices mapped to scanner RAS+ mm via the affine; "
                                       "no smoothing in the measurement mesh", "meshes": mesh_summary(res),
                             "anatomical_reference": "brain shell from T1 foreground (display only)" if res.brain_mesh is not None else None},
        "7_validation": validation,
        "8_quality_and_uncertainty": {"warnings": warnings, "qc": [{"severity": i.severity, "message": i.message} for i in res.qc],
                                      "prediction_uncertainty": unc,
                                      "uncertainty_note": "Voxel probabilities are not calibrated confidence values; ranges show sensitivity to the decision threshold."
                                      if unc else None},
        "9_limitations": STATIC_LIMITATIONS,
        "10_interpretation": interp,
    }


# ----------------------------------------------------------------------------- HTML

_CSS = """body{font-family:Segoe UI,Arial,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#1b2733;line-height:1.45}
h1{font-size:22px;margin-bottom:2px}h2{font-size:16px;border-bottom:2px solid #1f5f8b;padding-bottom:3px;margin-top:26px;color:#1f5f8b}
.warn{background:#fff3cd;border-left:4px solid #e0a800;padding:8px 12px;margin:6px 0}.disc{background:#fdecea;border-left:4px solid #c62828;padding:8px 12px;font-weight:600}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #cfd8dc;padding:4px 8px;text-align:left;vertical-align:top}th{background:#eef3f7}
code{background:#eef3f7;padding:1px 4px;border-radius:3px}small{color:#5b6b78}"""


def _tbl(rows, head=None):
    h = "".join(f"<th>{html.escape(str(c))}</th>" for c in head) if head else ""
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table>{('<tr>' + h + '</tr>') if head else ''}{body}</table>"


def render_html(rep: dict, compact: bool = False) -> str:
    esc = html.escape
    m = rep["5_measurements"]["values"]
    rows = []
    for label, key, unit in (("Volume", "volume_cm3", "cm3"), ("Volume, +/-1 voxel boundary range", "volume_cm3_boundary_range", "cm3"),
                             ("Max 3D diameter (Feret)", "max_diameter_mm", "mm"), ("Length L", "length_mm", "mm"), ("Width W", "width_mm", "mm"),
                             ("Height H", "height_mm", "mm"), ("Max axial in-plane diameter", "axial_max_diameter_mm", "mm"),
                             ("Axis-aligned bounding box (R-L, A-P, S-I)", "aabb_extent_mm", "mm"), ("Surface area", "surface_area_cm2", "cm2"),
                             ("Centroid (scanner RAS)", "centroid_world_mm", "mm"), ("Connected components", "n_components", "")):
        rows.append([label + (f" [{unit}]" if unit else "")] + [(m[r].get(key, "-") if m[r].get("present") else "absent") for r in ("WT", "TC", "ET")])
    v = rep["7_validation"]
    val_rows = []
    for r, s in (v.get("summary") or {}).items():
        val_rows.append([r, s.get("dice"), s.get("hd95_median_mm")])
    parts = [f"<h1>{esc(rep['title'])}</h1><small>Case {esc(str(rep['case_id']))} - generated {rep['generated_utc']} - {esc(rep['provenance']['result_type'])}</small>",
             f"<p class='disc'>{esc(rep['disclaimer'])}</p>"]
    inp = rep["1_input"]
    parts += ["<h2>1-2. Input and modalities</h2>", _tbl([["Grid", inp["shape"]], ["Voxel size (mm)", inp["spacing_mm"]], ["Orientation", inp["orientation"]],
                                                         ["Working grid", f"{inp['working_grid']['shape']} RAS 1 mm (resampled: {inp['working_grid']['resampled']})"],
                                                         ["Modalities", ", ".join(rep["2_modalities"]["detected"])]])]
    mod = rep["3_model"]
    parts += ["<h2>3. Model / source</h2>", _tbl([[k, v_] for k, v_ in mod.items() if v_ not in (None, {})])]
    seg = rep["4_segmentation"]
    parts += [f"<h2>4. Segmentation ({esc(seg['label'])})</h2>",
              _tbl([[r, d["name"], "yes" if d["present"] else "no", d["n_voxels"]] for r, d in seg["regions"].items()], ["Region", "Definition", "Present", "Voxels"])]
    parts += ["<h2>5. Physical measurements</h2>", _tbl(rows, ["Measure", "Whole tumour", "Tumour core", "Enhancing"]),
              f"<p><small>{esc(rep['5_measurements']['precision_note'])}</small></p>",
              ("<h3>How each value is computed</h3>" + _tbl([[k, v_] for k, v_ in rep["5_measurements"]["methods"].items()])) if compact else
              "<details><summary>How each value is computed</summary>" + _tbl([[k, v_] for k, v_ in rep["5_measurements"]["methods"].items()]) + "</details>"]
    rec = rep["6_reconstruction"]
    parts += ["<h2>6. 3D reconstruction</h2>", f"<p>{esc(rec['method'])}</p>",
              _tbl([[r, d.get("n_vertices"), d.get("n_faces"), d.get("watertight", "-")] for r, d in rec["meshes"].items()], ["Region", "Vertices", "Faces", "Watertight"])]
    parts += ["<h2>7. Model validation evidence</h2>", f"<p><b>Evidence level:</b> {esc(str(v.get('evidence_level')))}</p>"]
    for c in v.get("caveats", []):
        parts.append(f"<div class='warn'>{esc(c)}</div>")
    if val_rows:
        parts.append(_tbl(val_rows, ["Region", "Mean Dice (held-out)", "Median HD95 mm"]))
    if v.get("scope"):
        parts.append(f"<p><small>{esc(v['scope'])}</small></p>")
    q = rep["8_quality_and_uncertainty"]
    parts.append("<h2>8. Quality warnings and uncertainty</h2>")
    parts += [f"<div class='warn'>{esc(w)}</div>" for w in q["warnings"]] or ["<p>No warnings.</p>"]
    if q.get("prediction_uncertainty"):
        parts.append(_tbl([[r, d["volume_range_cm3"], d["uncertain_band_pct"], d["mean_prob_in_mask"]] for r, d in q["prediction_uncertainty"].items()],
                          ["Region", "Volume range (cm3) at p>=0.8 / p>=0.2", "% uncertain band", "Mean probability in mask"]))
        parts.append(f"<p><small>{esc(q['uncertainty_note'])}</small></p>")
    parts += ["<h2>9. Technical limitations</h2><ul>" + "".join(f"<li>{esc(x)}</li>" for x in rep["9_limitations"]) + "</ul>",
              f"<h2>10. Research interpretation</h2><p>{esc(rep['10_interpretation'])}</p>"]
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{esc(rep['title'])}</title><style>{_CSS}</style></head><body>{''.join(parts)}</body></html>"


def measurements_rows(res: AnalysisResult) -> list[dict]:
    """Flat rows (one per region) for CSV export."""
    rows = []
    for r, m in res.measurements.items():
        row = {"case_id": res.case_id, "region": r, "result_type": res.source}
        for k, v in m.items():
            if isinstance(v, (list, tuple)) and v and not isinstance(v[0], (list, tuple)) and len(v) <= 3:
                for i, x in enumerate(v):
                    row[f"{k}_{i}"] = x
            elif not isinstance(v, (list, dict, tuple)):
                row[k] = v
        rows.append(row)
    return rows

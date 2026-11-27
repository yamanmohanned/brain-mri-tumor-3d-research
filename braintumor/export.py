"""Export masks (NIfTI on the ORIGINAL scan grid), meshes, reports and images."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from . import REGIONS
from .mesh import export_mesh, smooth_mesh
from .pipeline import AnalysisResult
from .report import build_report, measurements_rows, render_html
from .viz import save_slices_png

ALL_ITEMS = ("mask", "meshes", "report", "csv", "images")


def labels_from_masks(masks: dict) -> np.ndarray:
    """BraTS-convention label volume: 2 = oedema (WT only), 1 = core without ET, 4 = enhancing tumour."""
    lab = np.zeros(masks["WT"].shape, np.uint8)
    lab[masks["WT"]] = 2
    lab[masks["TC"]] = 1
    lab[masks["ET"]] = 4
    return lab


def mask_on_original_grid(res: AnalysisResult) -> np.ndarray:
    """Label volume resampled (nearest neighbour; exact reorientation when no resampling was needed) to the scan's own grid."""
    return res.prep.grid.to_orig(labels_from_masks(res.masks), order=0).astype(np.uint8)


def save_nifti_mask(res: AnalysisResult, path: Path) -> Path:
    lab = mask_on_original_grid(res)
    img = nib.Nifti1Image(lab, res.prep.grid.orig_affine)
    img.set_qform(res.prep.grid.orig_affine, code=2)
    img.set_sform(res.prep.grid.orig_affine, code=2)
    img.header.set_data_dtype(np.uint8)
    nib.save(img, str(path))
    return path


def export_all(res: AnalysisResult, out_dir: str | Path, items=ALL_ITEMS, mesh_format: str = "stl", smooth_iterations: int = 0,
               overwrite: bool = False, extra_images: dict | None = None) -> dict:
    """Write the selected outputs. Never overwrites existing files unless ``overwrite``; returns {item: [paths]}."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cid = res.case_id
    written: dict[str, list[str]] = {}

    def target(name: str) -> Path:
        p = out / name
        if p.exists() and not overwrite:
            raise FileExistsError(f"{p} exists (choose another folder or enable overwrite)")
        return p

    if "mask" in items:
        written["mask"] = [str(save_nifti_mask(res, target(f"{cid}_seg_{'pred' if res.source == 'model' else 'annot'}.nii.gz")))]
    if "meshes" in items:
        written["meshes"] = []
        notes = {}
        for r in REGIONS:
            m = res.meshes.get(r)
            if m is None:
                continue
            if smooth_iterations:
                m, rep = smooth_mesh(m, smooth_iterations)
                notes[r] = rep
            written["meshes"].append(str(export_mesh(m, target(f"{cid}_{r}.{mesh_format}"))))
        if notes:
            (out / f"{cid}_mesh_smoothing.json").write_text(json.dumps(notes, indent=2))
        (out / f"{cid}_mesh_readme.txt").write_text(
            "Mesh vertices are in scanner RAS+ millimetres (x=right, y=anterior, z=superior) taken from the scan affine. "
            "STL/OBJ/PLY carry no coordinate-system tag; import into viewers accordingly (e.g. 3D Slicer expects LPS for some formats).\n"
            + ("Meshes were smoothed (Taubin, volume rescaled); see the smoothing json for displacement.\n" if smooth_iterations
               else "Meshes are unsmoothed marching-cubes surfaces of the binary mask.\n"))
    rep = None
    if "report" in items or "csv" in items:
        rep = build_report(res)
    if "report" in items:
        jp, hp = target(f"{cid}_report.json"), target(f"{cid}_report.html")
        jp.write_text(json.dumps(rep, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
        hp.write_text(render_html(rep), encoding="utf-8")
        written["report"] = [str(jp), str(hp)]
    if "csv" in items:
        rows = measurements_rows(res)
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("case_id", "region", "result_type"), k))
        p = target(f"{cid}_measurements.csv")
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        written["csv"] = [str(p)]
    if "images" in items:
        p = target(f"{cid}_slices.png")
        bg = res.prep.raw_work[res.volume["modalities"].index("flair")]
        save_slices_png(p, bg, res.masks, title=f"{cid} - FLAIR", opacity=0.45)
        written["images"] = [str(p)]
        for name, png in (extra_images or {}).items():
            q = target(f"{cid}_{name}.png")
            q.write_bytes(png)
            written["images"].append(str(q))
    return written

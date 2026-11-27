"""Surface reconstruction: marching cubes in physical (scanner RAS+, mm) coordinates.

* The *measurement* mesh is the raw marching-cubes surface at the 0.5 iso-level of the binary mask: no
  smoothing, so it is a faithful (staircase-free but unsmoothed) rendering of the segmentation.
* Optional Taubin smoothing (volume-preserving, no shrinkage) is for display/printing only; its volume
  drift is reported so nothing is hidden.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage
from skimage.measure import marching_cubes

from .geometry import apply_affine


def mask_to_mesh(mask: np.ndarray, affine: np.ndarray, step_size: int = 1) -> trimesh.Trimesh | None:
    """Binary mask -> triangle mesh with vertices in world (scanner RAS+) mm and outward-facing normals."""
    if mask is None or not mask.any():
        return None
    padded = np.pad(mask.astype(np.float32), 1)
    try:
        verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=(1.0, 1.0, 1.0), step_size=step_size, allow_degenerate=False)
    except (ValueError, RuntimeError):
        return None
    verts = verts - 1.0                                      # undo the 1-voxel pad (index coordinates)
    world = apply_affine(affine, verts)
    if np.linalg.det(np.asarray(affine)[:3, :3]) < 0:        # a mirrored affine inverts triangle winding
        faces = faces[:, ::-1]
    mesh = trimesh.Trimesh(vertices=world, faces=faces, process=False)
    if mesh.volume < 0:                                      # safety net: enforce outward normals
        mesh.invert()
    return mesh


def smooth_mesh(mesh: trimesh.Trimesh, iterations: int = 10, lamb: float = 0.5, nu: float = -0.53,
                preserve_volume: bool = True) -> tuple[trimesh.Trimesh, dict]:
    """Taubin smoothing on a copy (display / printing only - never used for measurements).

    Taubin smoothing is not strictly volume preserving; with ``preserve_volume`` the result is rescaled about
    its centroid so the enclosed volume equals the original. The report states the correction that was needed
    and how far vertices moved, so the geometry change is explicit.
    """
    m = mesh.copy()
    m.merge_vertices()
    v0 = float(m.volume)
    before = m.vertices.copy()
    trimesh.smoothing.filter_taubin(m, lamb=lamb, nu=nu, iterations=iterations)
    v1 = float(m.volume)
    if preserve_volume and v1 > 0:
        c = m.centroid
        m.vertices = c + (m.vertices - c) * (v0 / v1) ** (1.0 / 3.0)
    disp = np.linalg.norm(m.vertices - before, axis=1)
    return m, {"iterations": iterations, "volume_before_mm3": v0, "volume_after_taubin_mm3": v1, "volume_after_mm3": float(m.volume),
               "volume_drift_before_correction_pct": 100.0 * abs(v1 - v0) / max(v0, 1e-9),
               "volume_drift_pct": 100.0 * abs(float(m.volume) - v0) / max(v0, 1e-9),
               "mean_vertex_shift_mm": float(disp.mean()), "max_vertex_shift_mm": float(disp.max())}


def brain_shell_mask(t1_or_any: np.ndarray) -> np.ndarray:
    """Anatomical-reference mask: largest connected foreground component of an intensity volume, holes filled."""
    fg = t1_or_any > 0
    lab, n = ndimage.label(fg)
    if n == 0:
        return fg
    keep = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    return ndimage.binary_fill_holes(lab == keep)


def mesh_report(mesh: trimesh.Trimesh | None) -> dict:
    if mesh is None:
        return {"n_vertices": 0, "n_faces": 0}
    return {"n_vertices": int(len(mesh.vertices)), "n_faces": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight), "volume_mm3": float(mesh.volume),
            "surface_area_mm2": float(mesh.area)}


def export_mesh(mesh: trimesh.Trimesh, path: str | Path) -> Path:
    """Write STL / OBJ / PLY (format from suffix). Vertices are scanner RAS+ millimetres."""
    path = Path(path)
    if path.suffix.lower() not in (".stl", ".obj", ".ply"):
        raise ValueError(f"Unsupported mesh format {path.suffix!r} (use .stl, .obj or .ply)")
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))
    return path

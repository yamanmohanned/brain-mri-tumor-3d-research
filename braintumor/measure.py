"""Physical tumour measurements from a segmentation mask + its affine (millimetres, never voxel counts)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

from .geometry import apply_affine, voxel_volume_mm3
from .postprocess import STRUCT26

#: How each reported number is computed (shown in the GUI and in reports).
METHODS = {
    "volume": "Voxel count x physical voxel volume |det(affine)| (mm3, cm3). Subject to boundary/partial-volume error.",
    "volume_boundary_sensitivity": "Volume after eroding / dilating the mask by one voxel (6-connectivity): a +/-1 voxel boundary error range, "
                                   "not a statistical confidence interval.",
    "surface_area": "Area of the marching-cubes surface (iso-level 0.5) in mm2. Marching cubes slightly over/under-estimates curved area.",
    "max_diameter": "Maximum 3D Feret diameter: largest distance between any two surface vertices (convex-hull vertices of the "
                    "marching-cubes mesh), in true scanner millimetres. This is the real maximum geometric dimension, not a bounding-box side.",
    "orthogonal_dims": "Length L = max 3D diameter; Width W = largest diameter of the shape projected perpendicular to L; "
                       "Height H = extent along the axis orthogonal to both. Data-driven orthogonal triple, not a minimum-volume box.",
    "aabb": "Axis-aligned bounding box in scanner RAS axes (x=R-L, y=A-P, z=S-I), from mesh vertex extremes. Depends on head position; "
            "differs from the true diameters for obliquely oriented tumours.",
    "axial_max_diameter": "Longest 2D Feret diameter within any single axial slice of the working grid (RANO-style in-plane long axis).",
    "centroid": "Mean position of mask voxel centres, in voxel indices of the working grid and in scanner RAS+ mm.",
    "components": "26-connected components of the mask; counts all components (none removed unless a minimum size was configured).",
}


def _hull_vertices(points: np.ndarray) -> np.ndarray:
    pts = np.unique(np.round(points, 4), axis=0)
    if len(pts) < 4:
        return pts
    try:
        return pts[ConvexHull(pts).vertices]
    except Exception:  # coplanar / degenerate
        return pts


def max_feret(points: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    h = _hull_vertices(points)
    if len(h) < 2:
        return 0.0, h[0] if len(h) else np.zeros(3), h[0] if len(h) else np.zeros(3)
    d = cdist(h, h)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    return float(d[i, j]), h[i], h[j]


def orthogonal_dims(points: np.ndarray) -> dict:
    """L = max Feret; W = max extent perpendicular to L; H = extent along the third orthogonal axis."""
    h = _hull_vertices(points)
    L, p1, p2 = max_feret(h)
    if L == 0:
        return {"L_mm": 0.0, "W_mm": 0.0, "H_mm": 0.0, "axes": np.eye(3).tolist()}
    u1 = (p2 - p1) / L
    helper = np.eye(3)[np.argmin(np.abs(u1))]
    e2 = np.cross(u1, helper)
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(u1, e2)
    q = np.stack([h @ e2, h @ e3], axis=1)
    q = np.unique(np.round(q, 4), axis=0)
    if len(q) >= 3:
        try:
            q = q[ConvexHull(q).vertices]
        except Exception:
            pass
    d2 = cdist(q, q)
    a, b = np.unravel_index(np.argmax(d2), d2.shape)
    W = float(d2[a, b])
    if W > 0:
        v = q[b] - q[a]
        v /= np.linalg.norm(v)
        u2 = v[0] * e2 + v[1] * e3
    else:
        u2 = e2
    u3 = np.cross(u1, u2)
    proj = h @ u3
    return {"L_mm": float(L), "W_mm": W, "H_mm": float(proj.max() - proj.min()), "axes": [u1.tolist(), u2.tolist(), u3.tolist()]}


def max_axial_diameter(mask: np.ndarray, affine: np.ndarray) -> tuple[float, int]:
    """Longest in-plane Feret diameter over axial slices (array axis 2 of the RAS working grid). Pixels are treated as
    squares (corner points) so a 1-voxel lesion has the physical size of a voxel, not zero."""
    best, best_k = 0.0, -1
    corners = np.array([[dx, dy] for dx in (-.5, .5) for dy in (-.5, .5)])
    for k in np.flatnonzero(mask.any(axis=(0, 1))):
        sl = mask[:, :, k]
        edge = sl & ~ndimage.binary_erosion(sl, border_value=0)
        ij = np.argwhere(edge)
        pts = (ij[:, None, :] + corners[None]).reshape(-1, 2)
        pts3 = np.column_stack([pts, np.full(len(pts), k, float)])
        d, _, _ = max_feret(apply_affine(affine, pts3))
        if d > best:
            best, best_k = d, int(k)
    return best, best_k


def boundary_sensitivity(mask: np.ndarray, voxel_mm3: float) -> list[float]:
    """Volumes (cm3) after 1-voxel erosion and dilation."""
    er = ndimage.binary_erosion(mask)
    di = ndimage.binary_dilation(mask)
    return [round(er.sum() * voxel_mm3 / 1000, 3), round(di.sum() * voxel_mm3 / 1000, 3)]


def components(mask: np.ndarray, voxel_mm3: float, top: int = 10) -> dict:
    lab, n = ndimage.label(mask, structure=STRUCT26)
    sizes = np.sort(np.bincount(lab.ravel())[1:])[::-1] if n else np.array([], int)
    vols = sizes * voxel_mm3 / 1000
    return {"n_components": int(n), "component_volumes_cm3": [round(float(v), 3) for v in vols[:top]],
            "largest_component_fraction": round(float(sizes[0] / sizes.sum()), 3) if n else 0.0}


def measure_region(mask: np.ndarray, affine: np.ndarray, mesh=None, with_axial: bool = True) -> dict:
    """All physical measurements for one binary region. Empty masks return ``{'present': False}``."""
    n = int(mask.sum())
    if n == 0:
        return {"present": False, "volume_mm3": 0.0, "volume_cm3": 0.0, "n_voxels": 0}
    vv = voxel_volume_mm3(affine)
    idx = np.argwhere(mask)
    out = {
        "present": True, "n_voxels": n, "voxel_volume_mm3": vv,
        "volume_mm3": n * vv, "volume_cm3": n * vv / 1000.0,
        "volume_cm3_boundary_range": boundary_sensitivity(mask, vv),
        "centroid_voxel": idx.mean(0).tolist(),
        "centroid_world_mm": apply_affine(affine, idx.mean(0)[None])[0].tolist(),
    }
    out.update(components(mask, vv))
    if mesh is not None and len(mesh.vertices):
        v = mesh.vertices
        out["surface_area_mm2"] = float(mesh.area)
        out["mesh_volume_cm3"] = float(mesh.volume) / 1000.0
        out["mesh_watertight"] = bool(mesh.is_watertight)
        lo, hi = v.min(0), v.max(0)
        out["aabb_min_mm"], out["aabb_max_mm"], out["aabb_extent_mm"] = lo.tolist(), hi.tolist(), (hi - lo).tolist()
        d, p1, p2 = max_feret(v)
        out["max_diameter_mm"] = d
        out["max_diameter_endpoints_mm"] = [p1.tolist(), p2.tolist()]
        od = orthogonal_dims(v)
        out.update({"length_mm": od["L_mm"], "width_mm": od["W_mm"], "height_mm": od["H_mm"], "orthogonal_axes": od["axes"]})
    if with_axial:
        d, k = max_axial_diameter(mask, affine)
        out["axial_max_diameter_mm"], out["axial_slice_index"] = d, k
    return out


def measure_all(masks: dict, affine: np.ndarray, meshes: dict | None = None, with_axial: bool = True) -> dict:
    return {r: measure_region(m, affine, (meshes or {}).get(r), with_axial) for r, m in masks.items()}


def round_for_display(m: dict) -> dict:
    """Round to a precision the data can support: lengths to 1 mm (voxel size), volumes to 0.1 cm3."""
    if not m.get("present"):
        return m
    r = dict(m)
    for k in ("max_diameter_mm", "length_mm", "width_mm", "height_mm", "axial_max_diameter_mm"):
        if k in r:
            r[k] = round(r[k])
    for k in ("volume_cm3", "mesh_volume_cm3"):
        if k in r:
            r[k] = round(r[k], 1)
    if "volume_cm3_boundary_range" in r:
        r["volume_cm3_boundary_range"] = [round(v, 1) for v in r["volume_cm3_boundary_range"]]
    for k in ("aabb_extent_mm",):
        if k in r:
            r[k] = [round(x) for x in r[k]]
    for k in ("centroid_world_mm",):
        if k in r:
            r[k] = [round(x, 1) for x in r[k]]
    if "surface_area_mm2" in r:
        r["surface_area_cm2"] = round(r["surface_area_mm2"] / 100.0, 1)
    return r


def measurement_errors(pred: np.ndarray, gt: np.ndarray, affine: np.ndarray) -> dict:
    """Predicted-vs-ground-truth physical measurement differences for ONE region (signed = prediction - truth). {} unless both are non-empty."""
    from .mesh import mask_to_mesh
    mp = measure_region(pred, affine, mask_to_mesh(pred, affine), with_axial=True)
    mg = measure_region(gt, affine, mask_to_mesh(gt, affine), with_axial=True)
    if not (mp.get("present") and mg.get("present")):
        return {}
    d = {
        "m_vol_err_cm3": mp["volume_cm3"] - mg["volume_cm3"], "m_vol_rel_err_pct": 100.0 * (mp["volume_cm3"] - mg["volume_cm3"]) / mg["volume_cm3"],
        "m_max_diam_err_mm": mp["max_diameter_mm"] - mg["max_diameter_mm"], "m_length_err_mm": mp["length_mm"] - mg["length_mm"],
        "m_width_err_mm": mp["width_mm"] - mg["width_mm"], "m_height_err_mm": mp["height_mm"] - mg["height_mm"],
        "m_axial_diam_err_mm": mp["axial_max_diameter_mm"] - mg["axial_max_diameter_mm"],
        "m_surface_area_err_cm2": (mp["surface_area_mm2"] - mg["surface_area_mm2"]) / 100.0,
        "m_centroid_dist_mm": float(np.linalg.norm(np.array(mp["centroid_world_mm"]) - np.array(mg["centroid_world_mm"]))),
    }
    for k in ("m_vol_err_cm3", "m_max_diam_err_mm", "m_surface_area_err_cm2"):
        d[k.replace("_err_", "_abs_err_")] = abs(d[k])
    d["m_vol_abs_rel_err_pct"] = abs(d["m_vol_rel_err_pct"])
    return d

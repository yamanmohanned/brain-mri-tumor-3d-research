"""Measurements and meshes verified against analytic ellipsoid phantoms (synthetic fixtures)."""
import numpy as np
import pytest

from braintumor.geometry import apply_affine, make_working_grid, voxel_volume_mm3
from braintumor.measure import max_feret, measure_region, orthogonal_dims, round_for_display
from braintumor.mesh import export_mesh, mask_to_mesh, mesh_report, smooth_mesh
from braintumor.postprocess import enforce_hierarchy, postprocess, remove_small_components, uncertainty_summary
from tests.phantom import ellipsoid, make_phantom

RADII = (14.0, 11.0, 9.0)


def _measure(kind="ras", spacing=(1, 1, 1), shape=(72, 80, 64), **kw):
    ph = make_phantom(shape=shape, spacing=spacing, affine_kind=kind, wt_radii=RADII, **kw)
    mask = ph["seg"] > 0
    mesh = mask_to_mesh(mask, ph["affine"])
    return ph, mask, mesh, measure_region(mask, ph["affine"], mesh)


@pytest.mark.parametrize("kind", ["ras", "las", "lps", "permuted", "oblique"])
def test_volume_and_diameters_match_analytic_truth_for_any_orientation(kind):
    ph, mask, mesh, m = _measure(kind)
    t = ph["truth"]
    assert m["volume_mm3"] == pytest.approx(t["WT_voxels_volume_mm3"])
    assert abs(m["volume_mm3"] - t["WT_volume_mm3"]) / t["WT_volume_mm3"] < 0.03        # discretisation error only
    assert abs(m["max_diameter_mm"] - t["WT_max_diameter_mm"]) < 1.6                   # marching-cubes surface adds <= ~1 voxel
    assert abs(m["length_mm"] - 28) < 1.6 and abs(m["width_mm"] - 22) < 1.6 and abs(m["height_mm"] - 18) < 1.6
    assert m["length_mm"] >= m["width_mm"] >= m["height_mm"] * 0.95
    assert abs(m["axial_max_diameter_mm"] - 28) < 1.6                                   # tumour centred on an axial slice
    assert m["n_components"] == 1
    assert m["mesh_watertight"]
    assert abs(m["mesh_volume_cm3"] * 1000 - m["volume_mm3"]) / m["volume_mm3"] < 0.04  # mesh and voxel volumes agree


def test_anisotropic_voxels_use_physical_not_voxel_units():
    ph, mask, mesh, m = _measure(spacing=(0.8, 0.8, 2.0), shape=(90, 100, 32), center_frac=(0.5, 0.5, 0.5))
    assert m["voxel_volume_mm3"] == pytest.approx(0.8 * 0.8 * 2.0)
    assert abs(m["volume_mm3"] - ph["truth"]["WT_volume_mm3"]) / ph["truth"]["WT_volume_mm3"] < 0.06
    assert abs(m["max_diameter_mm"] - 28) < 2.6                                          # through-plane sampling is 2 mm
    # voxel-count-as-mm would be wildly wrong: make sure we did not do that
    assert abs(m["aabb_extent_mm"][2] - 18) < 3.0


def test_centroid_world_position_is_affine_transformed_array_centroid():
    ph, mask, mesh, m = _measure("oblique")
    expected = apply_affine(ph["affine"], np.argwhere(mask).mean(0)[None])[0]
    assert np.allclose(m["centroid_world_mm"], expected, atol=1e-6)


def test_aabb_differs_from_true_diameters_when_oblique():
    _, _, _, ras = _measure("ras")
    _, _, _, obl = _measure("oblique")
    assert max(ras["aabb_extent_mm"]) == pytest.approx(ras["max_diameter_mm"], abs=1.0)  # aligned: box side == long axis
    assert max(obl["aabb_extent_mm"]) < obl["max_diameter_mm"] * 1.01                    # sanity
    assert obl["max_diameter_mm"] == pytest.approx(ras["max_diameter_mm"], abs=1.0)      # diameter is rotation invariant
    assert not np.allclose(obl["aabb_extent_mm"], ras["aabb_extent_mm"], atol=1.0)       # box is not


def test_boundary_range_brackets_the_volume():
    _, _, _, m = _measure()
    lo, hi = m["volume_cm3_boundary_range"]
    assert lo < m["volume_cm3"] < hi


def test_multifocal_components_counted_and_empty_mask_handled():
    ph = make_phantom(wt_radii=RADII, second_blob=True)
    m = measure_region(ph["seg"] > 0, ph["affine"], None, with_axial=False)
    assert m["n_components"] == 2 and m["component_volumes_cm3"][0] > m["component_volumes_cm3"][1]
    assert m["largest_component_fraction"] < 1.0
    assert measure_region(np.zeros((8, 8, 8), bool), np.eye(4)) == {"present": False, "volume_mm3": 0.0, "volume_cm3": 0.0, "n_voxels": 0}


def test_feret_and_orthogonal_dims_on_known_points():
    box = np.array([[x, y, z] for x in (0, 10) for y in (0, 4) for z in (0, 2)], float)
    d, p, q = max_feret(box)
    assert d == pytest.approx(np.sqrt(100 + 16 + 4))
    od = orthogonal_dims(box)
    assert od["L_mm"] == pytest.approx(np.sqrt(120))
    assert od["W_mm"] >= od["H_mm"] > 0


def test_display_rounding_does_not_invent_precision():
    _, _, _, m = _measure()
    r = round_for_display(m)
    assert float(r["max_diameter_mm"]).is_integer()
    assert round(r["volume_cm3"], 1) == r["volume_cm3"]


# ----------------------------------------------------------------------------- meshes

@pytest.mark.parametrize("kind", ["ras", "las", "lps", "oblique"])
def test_mesh_has_outward_normals_and_world_coordinates(kind):
    ph = make_phantom(affine_kind=kind, wt_radii=RADII)
    mask = ph["seg"] > 0
    mesh = mask_to_mesh(mask, ph["affine"])
    assert mesh.volume > 0 and mesh.is_watertight
    world_c = apply_affine(ph["affine"], np.argwhere(mask).mean(0)[None])[0]
    assert np.linalg.norm(mesh.centroid - world_c) < 1.0                    # mesh lives in scanner space, not index space
    assert len(mesh.split(only_watertight=False)) == 1


def test_empty_mask_gives_no_mesh():
    assert mask_to_mesh(np.zeros((10, 10, 10), bool), np.eye(4)) is None


def test_smoothing_is_volume_preserving_and_reports_drift():
    ph = make_phantom(wt_radii=RADII)
    mesh = mask_to_mesh(ph["seg"] > 0, ph["affine"])
    sm, rep = smooth_mesh(mesh, iterations=10)
    assert rep["volume_drift_before_correction_pct"] > 0           # the uncorrected drift is reported, not hidden
    assert rep["volume_drift_pct"] < 0.1                           # after centroid rescale the volume is preserved
    assert rep["max_vertex_shift_mm"] < 1.5                        # smoothing stays within ~1 voxel of the raw surface
    assert sm.volume > 0 and len(sm.faces) == len(mesh.faces)
    raw, rep2 = smooth_mesh(mesh, iterations=10, preserve_volume=False)
    assert rep2["volume_drift_pct"] == pytest.approx(rep2["volume_drift_before_correction_pct"])


@pytest.mark.parametrize("suffix", [".stl", ".obj", ".ply"])
def test_mesh_export_roundtrip_preserves_geometry(tmp_path, suffix):
    import trimesh
    ph = make_phantom(wt_radii=RADII, affine_kind="las")
    mesh = mask_to_mesh(ph["seg"] > 0, ph["affine"])
    p = export_mesh(mesh, tmp_path / f"t{suffix}")
    back = trimesh.load(str(p), process=False)
    assert abs(back.volume - mesh.volume) / mesh.volume < 1e-3
    assert np.allclose(back.bounds, mesh.bounds, atol=1e-3)
    with pytest.raises(ValueError):
        export_mesh(mesh, tmp_path / "x.xyz")
    assert mesh_report(mesh)["watertight"]


# ----------------------------------------------------------------------------- postprocessing

def test_hierarchy_only_adds_and_default_postprocess_removes_nothing():
    z = np.zeros((10, 10, 10), bool)
    tc, wt, et = z.copy(), z.copy(), z.copy()
    et[2:4, 2:4, 2:4] = True        # ET outside TC/WT -> hierarchy must add, not delete
    wt[6:8, 6:8, 6:8] = True
    masks = {"TC": tc, "WT": wt, "ET": et}
    out, log = postprocess(masks, voxel_mm3=1.0)
    assert out["ET"].sum() == 8 and out["TC"].sum() == 8 and out["WT"].sum() == 16
    assert (out["TC"] | ~out["ET"]).all() and (out["WT"] | ~out["TC"]).all()
    assert all(v["voxels"] == 0 for v in log["components_removed"].values())


def test_small_component_removal_is_opt_in_and_logged():
    m = np.zeros((20, 20, 20), bool)
    m[2:10, 2:10, 2:10] = True
    m[15, 15, 15] = True
    kept, rm = remove_small_components(m, 0)
    assert kept.sum() == m.sum() and rm == []
    kept, rm = remove_small_components(m, 5)
    assert kept.sum() == m.sum() - 1 and rm == [1]


def test_uncertainty_summary_threshold_sensitivity():
    rng = np.random.default_rng(0)
    probs = rng.uniform(0, 1, (3, 10, 10, 10)).astype(np.float32)
    masks = {r: probs[i] >= 0.5 for i, r in enumerate(("TC", "WT", "ET"))}
    u = uncertainty_summary(probs, masks, 1.0)
    for r in ("TC", "WT", "ET"):
        lo, hi = u[r]["volume_range_cm3"]
        assert lo <= masks[r].sum() / 1000 <= hi
        assert 0 < u[r]["uncertain_band_pct"] < 100

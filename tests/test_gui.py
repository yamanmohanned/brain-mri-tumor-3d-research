"""GUI: display geometry, critical workflows and error paths (needs a desktop session; uses synthetic phantoms + a stand-in network).

The stand-in network validates the *application plumbing*; it says nothing about segmentation accuracy.
"""
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from braintumor import MODALITIES
from braintumor.model import build_model, save_checkpoint, update_card
from braintumor.viz import PLANE_AXIS, PLANES, cursor_pixel, extract_plane, pixel_to_index, set_cursor
from tests.phantom import ThresholdNet, make_affine, make_phantom, save_phantom

pytestmark = pytest.mark.gui


# ----------------------------------------------------------------------------- display geometry (no widgets)

@pytest.mark.parametrize("plane", PLANES)
def test_click_pixel_roundtrip_hits_the_clicked_voxel(plane):
    shape = (31, 37, 23)
    vol = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    rng = np.random.default_rng(0)
    for _ in range(40):
        cur = [int(rng.integers(0, s)) for s in shape]
        r, c = cursor_pixel(plane, shape, cur)
        img = extract_plane(vol, plane, cur[PLANE_AXIS[plane]])
        assert img[r, c] == vol[tuple(cur)]                      # the crosshair sits on exactly the voxel it represents
        assert set_cursor([0, 0, 0], plane, r, c, shape)[PLANE_AXIS[plane]] == 0
        back = set_cursor(cur, plane, r, c, shape)
        assert back == cur                                       # clicking the crosshair pixel is the identity


def test_display_is_radiological_right_on_screen_left():
    vol = np.zeros((20, 20, 20), np.float32)
    vol[18:, :, :] = 1                                           # strong +x side == patient RIGHT in the RAS working grid
    img = extract_plane(vol, "axial", 10)
    assert img[:, :2].mean() == 1 and img[:, 2:].mean() == 0      # right side of the patient is on the LEFT of the screen
    vol = np.zeros((20, 20, 20), np.float32)
    vol[:, 18:, :] = 1                                           # +y == anterior -> top of the screen
    a = extract_plane(vol, "axial", 10)
    assert a[:2].mean() == 1 and a[2:].mean() == 0
    vol = np.zeros((20, 20, 20), np.float32)
    vol[:, :, 18:] = 1                                           # +z == superior -> top for every plane
    for pl in ("coronal", "sagittal"):                           # the axial plane has no superior-inferior extent
        a = extract_plane(vol, pl, 10)
        assert a[:2].mean() == 1 and a[2:].mean() == 0


@pytest.mark.parametrize("kind", ["ras", "oblique"])
def test_slice_plane_texture_aligns_with_world(kind):
    """A marker voxel must appear on screen where its WORLD position projects (guards the pyvista top-down texture convention)."""
    pv = pytest.importorskip("pyvista")
    import vtk
    from braintumor.gui.widgets import plane_quad_texture
    shape = (30, 40, 20)
    vol = np.zeros(shape, np.float32)
    mk = (22, 9, 7)
    vol[mk[0] - 1:mk[0] + 2, mk[1] - 1:mk[1] + 2, mk[2] - 1:mk[2] + 2] = 1.0
    aff = make_affine(kind, shape, (1, 1, 1))
    world_marker = aff[:3, :3] @ np.array(mk, float) + aff[:3, 3]
    for plane in PLANES:
        quad, rgb = plane_quad_texture(vol, aff, plane, mk[PLANE_AXIS[plane]], (0.0, 1.0))
        pl = pv.Plotter(off_screen=True, window_size=(420, 420))
        pl.set_background("black")                                 # pyvista defaults to white
        pl.add_mesh(quad, texture=pv.Texture(rgb), lighting=False)
        pts = np.asarray(quad.points)
        center = pts.mean(0)
        n = np.cross(pts[1] - pts[0], pts[3] - pts[0])
        n /= np.linalg.norm(n)
        pl.camera.parallel_projection = True
        pl.camera.focal_point = center
        pl.camera.position = center + n * 200
        pl.camera.up = (pts[3] - pts[0]) / np.linalg.norm(pts[3] - pts[0])
        pl.camera.parallel_scale = 28
        shot = pl.screenshot(return_img=True)
        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToWorld()
        coord.SetValue(*world_marker)
        x, y = coord.GetComputedDoubleDisplayValue(pl.renderer)
        pl.close()
        rows, cols = np.nonzero(shot.sum(2) > 500)
        assert len(rows) > 20, plane
        got = np.array([cols.mean(), shot.shape[0] - rows.mean()])   # (x, y) with y measured from the bottom, like VTK display coords
        assert np.linalg.norm(got - np.array([x, y])) < 4.0, (plane, got, (x, y))


# ----------------------------------------------------------------------------- application workflows

@pytest.fixture
def app_env(qtbot, tmp_path, monkeypatch):
    from braintumor.gui import app as A
    from braintumor.gui.style import QSS
    qtbot.addWidget  # (pytest-qt fixture presence check)
    A.QApplication.instance().setStyleSheet(QSS)
    msgs = []
    monkeypatch.setattr(A, "err_box", lambda parent, title, msg, details=None: msgs.append((title, msg)))
    monkeypatch.setattr(A.os, "startfile", lambda p: msgs.append(("startfile", str(p))), raising=False)
    state = A.AppState(weights_dir=tmp_path / "weights", history_path=tmp_path / "history.json")
    ph = make_phantom(shape=(72, 80, 64), wt_radii=(14, 11, 9))
    files = save_phantom(ph, tmp_path / "case", "PH_001")
    win = A.MainWindow(state)
    qtbot.addWidget(win)
    win.show()
    qtbot.wait(300)
    yield {"A": A, "win": win, "state": state, "ph": ph, "files": files, "msgs": msgs, "tmp": tmp_path}
    win.close()


def _load_via_upload(env, with_seg=False):
    up = env["win"].pg["upload"]
    for m in MODALITIES:
        up.edits[m].setText(str(env["files"][m]))
    up.seg_edit.setText(str(env["files"]["seg"]) if with_seg else "")
    up.validate()
    return up


def _inject_model(env):
    s = env["state"]
    s.model = ThresholdNet(env["ph"]["data"])
    s.model_info = {"path": "standin.pth", "sha256": "0" * 64, "arch": "stand-in", "n_params": 0, "format": "test", "meta": {}, "card": {}}
    s.cfg.infer.roi, s.cfg.infer.device = (64, 64, 64), "cpu"
    s.model_changed.emit()


def test_starts_without_model_and_gates_pages_until_data_exists(app_env):
    win = app_env["win"]
    assert "No model loaded" in win.pg["dashboard"].c_model.body.text()
    en = [bool(win.nav.item(i).flags() & Qt_enabled()) for i in range(6)]
    assert en == [True, True, False, False, False, False]
    assert not win.pg["analysis"].btn_model.isEnabled()
    up = _load_via_upload(app_env)
    assert up.btn_next.isEnabled() and app_env["state"].volume is not None
    assert bool(win.nav.item(2).flags() & Qt_enabled()) and not bool(win.nav.item(3).flags() & Qt_enabled())
    assert win.pg["analysis"].btn_model.isEnabled() is False      # still no model
    assert win.pg["analysis"].btn_annot.isEnabled() is False      # and no annotation supplied


def Qt_enabled():
    from PyQt6.QtCore import Qt
    return Qt.ItemFlag.ItemIsEnabled


def test_upload_reports_clear_errors_and_keeps_state_clean(app_env):
    up = app_env["win"].pg["upload"]
    for m in ("t1", "t1ce", "t2"):
        up.edits[m].setText(str(app_env["files"][m]))
    up.validate()                                                 # FLAIR missing
    assert not up.btn_next.isEnabled() and app_env["state"].volume is None
    assert "FLAIR" in app_env["msgs"][-1][1] and "ERROR" in up.issues.item(0).text()
    up.edits["flair"].setText(str(app_env["tmp"] / "nope.nii.gz"))
    up.validate()
    assert "not found" in app_env["msgs"][-1][1].lower()


def test_full_workflow_model_run_3d_results_export(app_env, qtbot):
    win, s = app_env["win"], app_env["state"]
    _load_via_upload(app_env)
    _inject_model(env := app_env)
    an = win.pg["analysis"]
    assert an.btn_model.isEnabled()
    win.show_page("analysis")
    with qtbot.waitSignal(s.result_changed, timeout=180_000, raising=False) as blocker:
        an.run_model()
    assert blocker.signal_triggered, f"analysis did not finish: status={an.status.text()!r} messages={app_env['msgs']}"
    res = s.result
    assert res.source == "model" and res.measurements["WT"]["present"]
    truth = app_env["ph"]["truth"]["WT_voxels_volume_mm3"]
    assert abs(res.measurements["WT"]["volume_mm3"] - truth) / truth < 0.05
    assert all(bool(win.nav.item(i).flags() & Qt_enabled()) for i in range(6))
    assert "MODEL PREDICTION" in an.banner.text() and an.ortho.masks is not None
    # slice interaction: clicking the crosshair pixel keeps the voxel; sliders and views stay in sync
    cur = list(an.ortho.cursor)
    r, c = cursor_pixel("axial", an.ortho.vol.shape, cur)
    an.ortho.views["axial"].clicked.emit(r, c)
    assert an.ortho.cursor == cur and an.ortho.sliders["axial"].value() == cur[2]
    an.ortho.views["coronal"].scrolled.emit(1)
    assert an.ortho.cursor[1] == cur[1] + 1 and s.cursor == an.ortho.cursor
    # 3D page
    win.show_page("3d")
    qtbot.wait(600)
    v3 = win.pg["3d"]
    if v3.view.available:
        assert {"WT", "TC", "ET", "feret"} <= set(v3.view.actors)
        v3.c_planes.setChecked(True)
        assert {"plane_axial", "plane_coronal", "plane_sagittal"} <= set(v3.view.actors)
        v3.chk["WT"].setChecked(False)
        assert not v3.view.actors["WT"].GetVisibility()
        assert len(v3.view.screenshot_png()) > 3000
    wt_cell = v3.table.item(0, 0).text()
    assert float(wt_cell) == pytest.approx(res.measurements["WT"]["volume_cm3"], abs=0.06)
    # Results page content
    win.show_page("results")
    html = win.pg["results"].text.toPlainText()
    for needle in ("NOT A MEDICAL DEVICE", "Model validation evidence", "Not validated", "do not establish tumour type"):
        assert needle.lower() in html.lower(), needle
    # Export
    ex = win.pg["export"]
    ex.out.setText(str(app_env["tmp"] / "exports"))
    ex.export()
    out = app_env["tmp"] / "exports" / "PH_001"
    names = sorted(p.name for p in out.iterdir())
    assert any(n.endswith("_seg_pred.nii.gz") for n in names) and any(n.endswith("_WT.stl") for n in names)
    assert "PH_001_report.json" in names and "PH_001_report.html" in names and "PH_001_measurements.csv" in names and "PH_001_slices.png" in names
    if v3.view.available:
        assert "PH_001_3d_view.png" in names
    assert json.loads((out / "PH_001_report.json").read_text())["provenance"]["result_type"] == "model prediction"
    hist = json.loads((app_env["tmp"] / "history.json").read_text())
    assert hist[-1]["case"] == "PH_001" and hist[-1]["evidence"] == "Not validated"


def test_annotation_mode_labels_result_as_annotation(app_env, qtbot):
    win, s = app_env["win"], app_env["state"]
    _load_via_upload(app_env, with_seg=True)
    an = win.pg["analysis"]
    assert an.btn_annot.isEnabled() and not an.btn_model.isEnabled()
    with qtbot.waitSignal(s.result_changed, timeout=60_000, raising=False) as blocker:
        an.run_annotation()
    assert blocker.signal_triggered, f"annotation analysis did not finish: status={an.status.text()!r} messages={app_env['msgs']}"
    assert s.result.source == "external_mask" and "SUPPLIED ANNOTATION" in an.banner.text()
    assert "annotation" in json.dumps(__import__("braintumor.report", fromlist=["x"]).build_report(s.result)["provenance"])


def test_cancel_stops_the_analysis_cleanly(app_env, qtbot):
    import time
    import torch
    win, s = app_env["win"], app_env["state"]
    _load_via_upload(app_env)

    class Slow(torch.nn.Module):
        def forward(self, x):
            time.sleep(0.4)
            return torch.zeros(x.shape[0], 3, *x.shape[2:])
    s.model, s.model_info = Slow(), {"path": "slow", "sha256": "0" * 64, "arch": "slow", "n_params": 0, "format": "t", "meta": {}, "card": {}}
    s.cfg.infer.roi, s.cfg.infer.device, s.cfg.infer.overlap = (32, 32, 32), "cpu", 0.0
    s.model_changed.emit()
    an = win.pg["analysis"]
    an.run_model()
    qtbot.waitUntil(lambda: an.worker is not None and an.worker.isRunning() and an.btn_cancel.isEnabled(), timeout=10_000)
    qtbot.wait(900)
    with qtbot.waitSignal(an.worker.failed, timeout=30_000):
        an.btn_cancel.click()
    qtbot.wait(300)
    assert s.result is None and any("cancel" in m[1].lower() for m in app_env["msgs"] if m[0] == "Analysis stopped")


def test_dashboard_shows_checkpoint_and_its_evidence_level(qtbot, tmp_path, monkeypatch):
    from braintumor.gui import app as A
    monkeypatch.setattr(A, "err_box", lambda *a, **k: None)
    wdir = tmp_path / "weights"
    ck = save_checkpoint(wdir / "m.pth", build_model("segresnet"), "segresnet", {"note": "random weights for a UI test"})
    state = A.AppState(weights_dir=wdir, history_path=tmp_path / "h.json")
    win = A.MainWindow(state)
    qtbot.addWidget(win)
    win.show()
    qtbot.waitUntil(lambda: state.model is not None, timeout=20_000)             # exactly one checkpoint -> auto-loaded
    t = win.pg["dashboard"].c_model.body.text()
    assert "m.pth" in t and "Not validated" in t and "segresnet" in t
    update_card(ck, evaluation={"evidence_level": "Experimental / preliminary", "why": ["pre-specified acceptance criteria not met"], "n_test": 56, "split": "test",
                                "summary": {"WT": {"dice": 0.89, "dice_ci95": [0.86, 0.91], "hd95_median_mm": 4.1, "missed_rate": 0.0}}})
    state.load_model(str(ck))
    t = win.pg["dashboard"].c_model.body.text()
    assert "Experimental / preliminary" in t and "acceptance criteria not met" in t and "0.89" in t
    assert "Experimental / preliminary" in win.evid.text()
    win.close()

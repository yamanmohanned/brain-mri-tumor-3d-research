"""Reusable widgets: pixel-accurate slice view, three-plane viewer, and the 3D (VTK) viewport."""
from __future__ import annotations

import io

import numpy as np
from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PyQt6.QtWidgets import QLabel, QSlider, QVBoxLayout, QWidget

from ..viz import PLANE_AXIS, PLANES, REGION_COLORS, SIDE_LABELS, cursor_pixel, extract_plane, intensity_window, set_cursor, to_rgba_overlay, window


class SliceView(QWidget):
    """One 2D plane. Emits (row, col) image-pixel coordinates on click/drag; wheel steps the slice."""

    clicked = pyqtSignal(int, int)
    scrolled = pyqtSignal(int)

    def __init__(self, plane: str):
        super().__init__()
        self.plane = plane
        self._img: QImage | None = None
        self._buf = None
        self._cross = None
        self._geom = (0.0, 0.0, 1.0)
        self.setMinimumSize(200, 200)
        self.setMouseTracking(False)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_image(self, rgb: np.ndarray | None, cross=None) -> None:
        if rgb is None:
            self._img, self._buf = None, None
        else:
            self._buf = np.ascontiguousarray(rgb, dtype=np.uint8)
            h, w = self._buf.shape[:2]
            self._img = QImage(self._buf.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self._cross = cross
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#05080c"))
        if self._img is None:
            p.setPen(QColor("#6b7c8c"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "no image")
            return
        iw, ih = self._img.width(), self._img.height()
        s = min(self.width() / iw, self.height() / ih)
        x0, y0 = (self.width() - iw * s) / 2, (self.height() - ih * s) / 2
        self._geom = (x0, y0, s)
        p.drawImage(QRectF(x0, y0, iw * s, ih * s), self._img)
        if self._cross is not None:
            r, c = self._cross
            cx, cy = x0 + (c + 0.5) * s, y0 + (r + 0.5) * s
            p.setPen(QPen(QColor(255, 213, 74, 170), 1))
            p.drawLine(int(x0), int(cy), int(x0 + iw * s), int(cy))
            p.drawLine(int(cx), int(y0), int(cx), int(y0 + ih * s))
        left, right, top, bottom = SIDE_LABELS[self.plane]
        p.setPen(QColor("#ffd54a"))
        p.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        p.drawText(int(x0) + 6, int(self.height() / 2) + 5, left)
        p.drawText(int(x0 + iw * s) - 16, int(self.height() / 2) + 5, right)
        p.drawText(int(self.width() / 2) - 4, int(y0) + 16, top)
        p.drawText(int(self.width() / 2) - 4, int(y0 + ih * s) - 6, bottom)

    def _emit(self, e):
        if self._img is None:
            return
        x0, y0, s = self._geom
        col = int(np.clip((e.position().x() - x0) / s, 0, self._img.width() - 1))
        row = int(np.clip((e.position().y() - y0) / s, 0, self._img.height() - 1))
        self.clicked.emit(row, col)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._emit(e)

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.MouseButton.LeftButton:
            self._emit(e)

    def wheelEvent(self, e):
        self.scrolled.emit(1 if e.angleDelta().y() > 0 else -1)


class OrthoViewer(QWidget):
    """Axial / coronal / sagittal views with synchronised crosshair, per-plane sliders and segmentation overlay."""

    cursor_changed = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.vol: np.ndarray | None = None
        self.masks: dict | None = None
        self.cursor = [0, 0, 0]
        self.win = (0.0, 1.0)
        self.opacity = 0.45
        self.visible = ("WT", "TC", "ET")
        self.views, self.sliders, self.titles = {}, {}, {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        from PyQt6.QtWidgets import QGridLayout
        grid = QGridLayout()
        grid.setSpacing(8)
        lay.addLayout(grid)
        for i, plane in enumerate(PLANES):
            v = SliceView(plane)
            v.clicked.connect(lambda r, c, pl=plane: self._on_click(pl, r, c))
            v.scrolled.connect(lambda d, pl=plane: self._step(pl, d))
            sl = QSlider(Qt.Orientation.Horizontal)
            sl.valueChanged.connect(lambda val, pl=plane: self._on_slider(pl, val))
            t = QLabel(plane.capitalize())
            t.setStyleSheet("font-weight:600;")
            self.views[plane], self.sliders[plane], self.titles[plane] = v, sl, t
            col = QVBoxLayout()
            col.addWidget(t)
            col.addWidget(v, 1)
            col.addWidget(sl)
            w = QWidget()
            w.setLayout(col)
            grid.addWidget(w, 0, i)

    # ---- data
    def set_data(self, vol: np.ndarray | None, masks: dict | None = None, keep_cursor: bool = False) -> None:
        self.vol, self.masks = vol, masks
        if vol is None:
            for v in self.views.values():
                v.set_image(None)
            return
        self.win = intensity_window(vol)
        for plane, sl in self.sliders.items():
            sl.blockSignals(True)
            sl.setRange(0, vol.shape[PLANE_AXIS[plane]] - 1)
            sl.blockSignals(False)
        if not keep_cursor or any(c >= s for c, s in zip(self.cursor, vol.shape)):
            from ..viz import default_cursor
            self.cursor = default_cursor(masks, vol.shape)
        self.refresh()

    def set_overlay(self, visible=None, opacity=None) -> None:
        if visible is not None:
            self.visible = tuple(visible)
        if opacity is not None:
            self.opacity = opacity
        self.refresh()

    def set_cursor(self, cursor, emit: bool = True) -> None:
        self.cursor = [int(c) for c in cursor]
        self.refresh()
        if emit:
            self.cursor_changed.emit(self.cursor)

    # ---- interaction
    def _on_click(self, plane, row, col):
        if self.vol is not None:
            self.set_cursor(set_cursor(self.cursor, plane, row, col, self.vol.shape))

    def _on_slider(self, plane, val):
        c = list(self.cursor)
        c[PLANE_AXIS[plane]] = val
        self.set_cursor(c)

    def _step(self, plane, d):
        if self.vol is None:
            return
        ax = PLANE_AXIS[plane]
        c = list(self.cursor)
        c[ax] = int(np.clip(c[ax] + d, 0, self.vol.shape[ax] - 1))
        self.set_cursor(c)

    # ---- drawing
    def refresh(self) -> None:
        if self.vol is None:
            return
        lo, hi = self.win
        for plane, view in self.views.items():
            ax = PLANE_AXIS[plane]
            idx = self.cursor[ax]
            g = window(extract_plane(self.vol, plane, idx), lo, hi)
            rgb = np.repeat(g[..., None], 3, axis=2)
            if self.masks:
                ov = to_rgba_overlay({r: extract_plane(m, plane, idx).astype(bool) for r, m in self.masks.items()}, self.opacity, self.visible)
                a = ov[..., 3:4]
                rgb = rgb * (1 - a) + ov[..., :3] * a
            view.set_image((rgb * 255).astype(np.uint8), cursor_pixel(plane, self.vol.shape, self.cursor))
            self.titles[plane].setText(f"{plane.capitalize()}  -  slice {idx + 1}/{self.vol.shape[ax]}")
            self.sliders[plane].blockSignals(True)
            self.sliders[plane].setValue(idx)
            self.sliders[plane].blockSignals(False)


def plane_quad_texture(vol: np.ndarray, affine: np.ndarray, plane: str, idx: int, win=(0.0, 1.0)):
    """Textured quad for one MRI slice in WORLD (scanner RAS) coordinates.

    pyvista treats row 0 of a numpy texture as the TOP of the image (v = 1), so rows are flipped here so that
    array index b maps to texture v = b / nb. Verified by tests/test_gui.py::test_slice_plane_texture_aligns_with_world.
    """
    import pyvista as pv
    ax = PLANE_AXIS[plane]
    idx = int(np.clip(idx, 0, vol.shape[ax] - 1))
    a, b = [i for i in range(3) if i != ax]
    na, nb = vol.shape[a], vol.shape[b]
    corners = []
    for ia, ib in ((-.5, -.5), (na - .5, -.5), (na - .5, nb - .5), (-.5, nb - .5)):
        c = [0.0, 0.0, 0.0]
        c[a], c[b], c[ax] = ia, ib, float(idx)
        corners.append(c)
    world = np.asarray(corners) @ np.asarray(affine)[:3, :3].T + np.asarray(affine)[:3, 3]
    quad = pv.PolyData(world, np.array([4, 0, 1, 2, 3]))
    quad.active_texture_coordinates = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    sl = np.take(vol, idx, axis=ax)                                  # dims (a, b)
    img = (window(sl.T, *win) * 255).astype(np.uint8)[::-1]          # rows run along b, flipped for pyvista's top-down convention
    return quad, np.repeat(img[..., None], 3, axis=2)


def _to_pv(mesh):
    import pyvista as pv
    f = np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]).astype(np.int64).ravel()
    return pv.PolyData(np.asarray(mesh.vertices, float), f)


class Viewer3D(QWidget):
    """Interactive 3D viewport (rotate: left-drag, pan: middle/shift-drag, zoom: wheel). Falls back to a message if OpenGL/VTK is unavailable."""

    DEFAULT_OPACITY = {"WT": 0.25, "TC": 0.60, "ET": 1.0}

    def __init__(self):
        super().__init__()
        self.plotter = None
        self.error = None
        self.res = None
        self.actors: dict = {}
        self._plane_state = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        try:
            from pyvistaqt import QtInteractor
            self.plotter = QtInteractor(self)
            self.plotter.set_background("#0b1520")
            lay.addWidget(self.plotter.interactor)
        except Exception as exc:  # no OpenGL / VTK problem
            self.error = f"{type(exc).__name__}: {exc}"
            lbl = QLabel(f"3D viewport unavailable ({self.error}).\nMeasurements and exports still work.")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(lbl)

    @property
    def available(self) -> bool:
        return self.plotter is not None

    def clear(self) -> None:
        self.res, self.actors = None, {}
        if self.plotter:
            self.plotter.clear()
            self.plotter.render()

    def show_result(self, res, smooth_iterations: int = 0) -> None:
        self.res = res
        if not self.plotter:
            return
        self.plotter.clear()
        self.actors = {}
        from ..mesh import smooth_mesh
        for r in ("WT", "TC", "ET"):
            m = res.meshes.get(r)
            if m is None:
                continue
            if smooth_iterations:
                m, _ = smooth_mesh(m, smooth_iterations)
            self.actors[r] = self.plotter.add_mesh(_to_pv(m), color=REGION_COLORS[r], opacity=self.DEFAULT_OPACITY[r], smooth_shading=True,
                                                   name=r, specular=0.3)
        if res.brain_mesh is not None:
            self.actors["brain"] = self.plotter.add_mesh(_to_pv(res.brain_mesh), color="#c9d3dc", opacity=0.10, smooth_shading=True, name="brain")
        self.plotter.add_axes(xlabel="R", ylabel="A", zlabel="S")
        self._add_feret()
        self.plotter.view_isometric()
        self.plotter.reset_camera()
        self.plotter.render()

    def _add_feret(self) -> None:
        import pyvista as pv
        wt = self.res.measurements.get("WT", {}) if self.res else {}
        if not self.plotter or not wt.get("present") or "max_diameter_endpoints_mm" not in wt:
            return
        p1, p2 = (np.array(p) for p in wt["max_diameter_endpoints_mm"])
        self.actors["feret"] = self.plotter.add_mesh(pv.Line(p1, p2), color="white", line_width=3, name="feret")
        self.actors["feret_label"] = self.plotter.add_point_labels([(p1 + p2) / 2], [f"{wt['max_diameter_mm']:.0f} mm"], font_size=14, text_color="white",
                                                                   shape_opacity=0.3, name="feret_label", always_visible=True)

    # ---- controls
    def set_visible(self, key: str, on: bool) -> None:
        a = self.actors.get(key)
        if a is not None:
            a.SetVisibility(on)
            if key == "feret" and "feret_label" in self.actors:
                self.actors["feret_label"].SetVisibility(on)
            self.plotter.render()

    def set_opacity(self, key: str, value: float) -> None:
        a = self.actors.get(key)
        if a is not None:
            a.GetProperty().SetOpacity(value)
            self.plotter.render()

    def set_planes(self, on: bool, vol: np.ndarray | None = None, cursor=None, win=(0, 1)) -> None:
        """Textured MRI slice planes at the crosshair position (reference for tumour location)."""
        if not self.plotter or self.res is None:
            return
        for pl in PLANES:
            self.plotter.remove_actor(f"plane_{pl}", render=False)
            self.actors.pop(f"plane_{pl}", None)
        if on and vol is not None:
            import pyvista as pv
            for pl in PLANES:
                quad, rgb = plane_quad_texture(vol, self.res.work_affine, pl, cursor[PLANE_AXIS[pl]], win)
                self.actors[f"plane_{pl}"] = self.plotter.add_mesh(quad, texture=pv.Texture(rgb), name=f"plane_{pl}", lighting=False, opacity=0.85)
        self.plotter.render()

    def reset_camera(self) -> None:
        if self.plotter:
            self.plotter.view_isometric()
            self.plotter.reset_camera()
            self.plotter.render()

    def screenshot_png(self) -> bytes | None:
        if not self.plotter:
            return None
        from PIL import Image
        img = self.plotter.screenshot(return_img=True)
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, "PNG")
        return buf.getvalue()

    def shutdown(self) -> None:
        if self.plotter:
            try:
                self.plotter.close()
            except Exception:
                pass

"""Main application window and pages."""
from __future__ import annotations

import ctypes
import datetime as _dt
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QInputDialog,
                             QDoubleSpinBox, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSlider,
                             QSpinBox, QStackedWidget, QStatusBar, QTableWidget, QTableWidgetItem, QTabWidget, QTextBrowser, QVBoxLayout, QWidget)

from .. import MODALITIES, REGION_NAMES, RESEARCH_DISCLAIMER, __version__
from ..export import ALL_ITEMS, export_all
from ..geometry import apply_affine, make_working_grid
from ..inference import InferenceConfig
from ..io import AmbiguousSeriesError, InputError, load_dicom_series, load_label_map, load_modalities, load_modalities_dicom, assess_input, guess_modality, scan_dicom_folder
from ..model import CheckpointError, find_checkpoints, load_checkpoint
from ..pipeline import AnalysisConfig
from ..report import build_report, render_html
from .style import QSS, evidence_style
from .widgets import OrthoViewer, Viewer3D
from .workers import AnalysisWorker

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOD_LABEL = {"t1": "T1", "t1ce": "T1ce (contrast)", "t2": "T2", "flair": "FLAIR"}


# ----------------------------------------------------------------------------- state

def hardware_summary() -> dict:
    import torch
    cpu = platform.processor()
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            cpu = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except Exception:
        pass
    ram_tot = ram_free = None
    try:
        class MS(ctypes.Structure):
            _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong), ("tv", ctypes.c_ulonglong),
                        ("av", ctypes.c_ulonglong), ("tx", ctypes.c_ulonglong), ("ax", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]
        ms = MS()
        ms.l = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        ram_tot, ram_free = ms.tp / 2 ** 30, ms.ap / 2 ** 30
    except Exception:
        pass
    cuda = torch.cuda.is_available()
    return {"cpu": cpu, "cores": os.cpu_count(), "ram_total_gb": ram_tot, "ram_free_gb": ram_free, "torch": torch.__version__, "cuda": cuda,
            "gpu": torch.cuda.get_device_name(0) if cuda else None, "threads": torch.get_num_threads()}


class AppState(QObject):
    volume_changed = pyqtSignal()
    model_changed = pyqtSignal()
    result_changed = pyqtSignal()
    cursor_changed = pyqtSignal(list)

    def __init__(self, weights_dir: Path | None = None, history_path: Path | None = None):
        super().__init__()
        self.weights_dir = weights_dir or PROJECT_ROOT / "weights"
        self.history_path = history_path or PROJECT_ROOT / "runs" / "history.json"
        self.volume = None
        self.qc: list = []
        self.labels = None
        self.label_source = ""
        self.grid = None
        self.display = None
        self.model = None
        self.model_info = None
        self.result = None
        self.cfg = AnalysisConfig(infer=InferenceConfig(device="auto"))
        self.case_id = "case"
        self.cursor = [0, 0, 0]

    def load_volume(self, vol, labels=None, label_source="", case_id="case") -> None:
        self.volume, self.labels, self.label_source, self.case_id = vol, labels, label_source, case_id
        self.qc = assess_input(vol)
        self.grid = make_working_grid(vol.affine, vol.shape)
        self.display = self.grid.to_work(vol.data, order=1).astype(np.float32)
        self.result = None
        self.volume_changed.emit()
        self.result_changed.emit()

    def load_model(self, path: str) -> None:
        self.model, self.model_info = load_checkpoint(path, device="cpu")
        self.model_changed.emit()

    def set_result(self, res) -> None:
        self.result = res
        self._append_history(res)
        self.result_changed.emit()

    def evidence_level(self) -> str:
        if self.model_info is None:
            return "No model loaded"
        ev = (self.model_info.get("card") or {}).get("evaluation") or {}
        return ev.get("evidence_level", "Not validated")

    def history(self) -> list:
        try:
            return json.loads(self.history_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _append_history(self, res) -> None:
        h = self.history()
        wt = res.measurements["WT"]
        h.append({"utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), "case": res.case_id, "source": res.source,
                  "wt_cm3": round(wt.get("volume_cm3", 0.0), 1), "max_diameter_mm": round(wt.get("max_diameter_mm", 0.0)),
                  "checkpoint": (res.model_info or {}).get("sha256", "")[:10], "evidence": self.evidence_level() if res.source == "model" else "n/a"})
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            self.history_path.write_text(json.dumps(h[-200:], indent=1), encoding="utf-8")
        except OSError:
            pass


# ----------------------------------------------------------------------------- helpers

class Card(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 14)
        t = QLabel(title.upper())
        t.setObjectName("cardTitle")
        self.body = QLabel()
        self.body.setWordWrap(True)
        self.body.setTextFormat(Qt.TextFormat.RichText)
        self.body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(t)
        lay.addWidget(self.body)
        self.extra = QVBoxLayout()
        lay.addLayout(self.extra)
        lay.addStretch(1)


def page_header(title: str, sub: str = "") -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 4)
    h = QLabel(title)
    h.setObjectName("h1")
    lay.addWidget(h)
    if sub:
        s = QLabel(sub)
        s.setObjectName("muted")
        s.setWordWrap(True)
        lay.addWidget(s)
    return w


def table(headers: list[str], rows: list[list], widths=None) -> QTableWidget:
    t = QTableWidget(len(rows), len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    t.horizontalHeader().setStretchLastSection(True)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            t.setItem(i, j, QTableWidgetItem(str(v)))
    t.resizeColumnsToContents()
    return t


def err_box(parent, title: str, msg: str, details: str | None = None) -> None:
    m = QMessageBox(parent)
    m.setIcon(QMessageBox.Icon.Warning)
    m.setWindowTitle(title)
    m.setText(msg)
    if details:
        m.setDetailedText(details)
    m.exec()


# ----------------------------------------------------------------------------- pages

class DashboardPage(QWidget):
    def __init__(self, state: AppState, main):
        super().__init__()
        self.setObjectName("page")
        self.state, self.main = state, main
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 16)
        lay.addWidget(page_header("Brain MRI Tumour Analysis", "Research workflow: MRI -> tumour segmentation -> physical measurement -> interactive 3D reconstruction"))
        d = QLabel(RESEARCH_DISCLAIMER)
        d.setObjectName("disclaimer")
        d.setWordWrap(True)
        lay.addWidget(d)
        grid = QGridLayout()
        grid.setSpacing(14)
        self.c_model, self.c_fmt, self.c_hw = Card("Model & validation status"), Card("Supported MRI input"), Card("Hardware status")
        grid.addWidget(self.c_model, 0, 0, 1, 2)
        grid.addWidget(self.c_fmt, 0, 2)
        grid.addWidget(self.c_hw, 1, 0, 1, 3)
        lay.addLayout(grid)
        row = QHBoxLayout()
        self.combo = QComboBox()
        self.combo.setMinimumWidth(280)
        b_load, b_browse = QPushButton("Load selected"), QPushButton("Browse for checkpoint...")
        b_load.setObjectName("primary")
        b_load.clicked.connect(lambda: self._load(self.combo.currentData()))
        b_browse.clicked.connect(self._browse)
        row.addWidget(self.combo)
        row.addWidget(b_load)
        row.addWidget(b_browse)
        row.addStretch(1)
        self.c_model.extra.addLayout(row)
        t = QLabel("RECENT ANALYSES")
        t.setObjectName("cardTitle")
        lay.addWidget(t)
        self.hist = QTableWidget()
        lay.addWidget(self.hist, 1)
        self.c_fmt.body.setText("<b>NIfTI</b> (.nii, .nii.gz): four co-registered 3D volumes - T1, T1ce, T2, FLAIR<br><br>"
                                "<b>DICOM series</b> (MR only): one series folder per modality; geometry is rebuilt from the headers; "
                                "missing slices or mixed series are rejected.<br><br>"
                                "<span style='color:#5b6b78'>Expected preprocessing: skull-stripped, co-registered, ~1 mm isotropic (BraTS style). "
                                "Other inputs are flagged as out-of-distribution.</span>")
        state.model_changed.connect(self.refresh)
        state.result_changed.connect(self.refresh)
        self.refresh()

    def _load(self, path):
        if not path:
            return
        try:
            self.state.load_model(path)
        except CheckpointError as e:
            err_box(self, "Cannot load checkpoint", str(e))

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select model checkpoint", str(self.state.weights_dir), "PyTorch checkpoint (*.pth *.pt)")
        if p:
            self._load(p)

    def refresh(self):
        s = self.state
        found = find_checkpoints(s.weights_dir)
        cur = self.combo.currentData()
        self.combo.clear()
        for p in found:
            self.combo.addItem(p.name, str(p))
        if cur:
            i = self.combo.findData(cur)
            self.combo.setCurrentIndex(max(i, 0))
        if s.model_info is None:
            self.c_model.body.setText("<b style='color:#9b1c1c'>No model loaded.</b><br>"
                                      + (f"{len(found)} checkpoint(s) found in <code>{s.weights_dir}</code>." if found else
                                         f"No checkpoint found in <code>{s.weights_dir}</code>. Train one (see README) or copy a trained <code>.pth</code> there. "
                                         "Without a model you can still measure and reconstruct an existing segmentation (annotation)."))
        else:
            mi, card = s.model_info, s.model_info.get("card") or {}
            ev = card.get("evaluation") or {}
            lvl = s.evidence_level()
            why = "".join(f"<li>{w}</li>" for w in ev.get("why", [])) or ("<li>no evaluation record (*.card.json) is attached to this checkpoint</li>" if not ev else "")
            summ = ""
            for r, v in (ev.get("summary") or {}).items():
                summ += f"{r}: Dice {v['dice']} (95% CI {v['dice_ci95'][0]}-{v['dice_ci95'][1]}), median HD95 {v['hd95_median_mm']} mm<br>"
            self.c_model.body.setText(
                f"<b>{Path(mi['path']).name}</b> &nbsp; <code>{mi['arch']}</code> &middot; {mi['n_params'] / 1e6:.2f} M parameters &middot; sha256 <code>{mi['sha256'][:12]}</code><br>"
                f"Evidence level: <span style='{evidence_style(lvl)}'>&nbsp;{lvl}&nbsp;</span><br>"
                + (f"<small>Held-out evaluation on {ev.get('n_test')} patients ({ev.get('split')} split):</small><br>{summ}" if ev else "")
                + (f"<ul style='margin-top:2px'>{why}</ul>" if why else "")
                + "<small style='color:#5b6b78'>Evidence levels describe research validation, not clinical approval.</small>")
        hw = hardware_summary()
        gpu = (f"CUDA GPU: <b>{hw['gpu']}</b>" if hw["cuda"] else "No CUDA GPU in use - inference runs on CPU (about 1-2 min per scan)")
        ram = f"{hw['ram_free_gb']:.1f} GB free of {hw['ram_total_gb']:.1f} GB" if hw["ram_total_gb"] else "n/a"
        self.c_hw.body.setText(f"CPU: <b>{hw['cpu']}</b> ({hw['cores']} logical cores, torch threads {hw['threads']})<br>RAM: {ram}<br>{gpu}<br>PyTorch {hw['torch']} &middot; app {__version__}")
        h = list(reversed(s.history()))[:12]
        self.hist.clear()
        self.hist.setColumnCount(6)
        self.hist.setRowCount(len(h))
        self.hist.setHorizontalHeaderLabels(["Time (UTC)", "Case", "Result type", "WT volume cm3", "Max diameter mm", "Model evidence"])
        for i, r in enumerate(h):
            for j, k in enumerate(["utc", "case", "source", "wt_cm3", "max_diameter_mm", "evidence"]):
                self.hist.setItem(i, j, QTableWidgetItem(str(r.get(k, ""))))
        self.hist.horizontalHeader().setStretchLastSection(True)
        self.hist.resizeColumnsToContents()
        self.hist.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.hist.verticalHeader().setVisible(False)


class UploadPage(QWidget):
    def __init__(self, state: AppState, main):
        super().__init__()
        self.setObjectName("page")
        self.state, self.main = state, main
        self.edits: dict[str, QLineEdit] = {}
        self.dicom: dict[str, tuple] = {}
        self.dicom_lbl: dict[str, QLabel] = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 16)
        lay.addWidget(page_header("MRI Upload", "Select the four co-registered modalities, validate them, review the scan information, then continue to analysis."))
        self.tabs = QTabWidget()
        nif, dic = QWidget(), QWidget()
        fl = QFormLayout(nif)
        for m in MODALITIES:
            e = QLineEdit()
            e.setPlaceholderText(f"{MOD_LABEL[m]} NIfTI file (.nii / .nii.gz)")
            b = QPushButton("Browse...")
            b.clicked.connect(lambda _, mm=m: self._pick_nifti(mm))
            row = QHBoxLayout()
            row.addWidget(e, 1)
            row.addWidget(b)
            fl.addRow(MOD_LABEL[m], row)
            self.edits[m] = e
        auto = QPushButton("Auto-fill from folder (BraTS-style names)...")
        auto.clicked.connect(self._autofill)
        fl.addRow("", auto)
        fd = QFormLayout(dic)
        for m in MODALITIES:
            l = QLabel("not selected")
            l.setObjectName("muted")
            b = QPushButton("Series folder...")
            b.clicked.connect(lambda _, mm=m: self._pick_dicom(mm))
            row = QHBoxLayout()
            row.addWidget(l, 1)
            row.addWidget(b)
            fd.addRow(MOD_LABEL[m], row)
            self.dicom_lbl[m] = l
        self.tabs.addTab(nif, "NIfTI volumes")
        self.tabs.addTab(dic, "DICOM series (MR)")
        lay.addWidget(self.tabs)
        g = QGroupBox("Existing segmentation (optional)")
        gl = QHBoxLayout(g)
        self.seg_edit = QLineEdit()
        self.seg_edit.setPlaceholderText("BraTS-style label map (labels 1, 2, 4) - e.g. ground truth. Measured and shown as ANNOTATION, never as a prediction.")
        bs = QPushButton("Browse...")
        bs.clicked.connect(self._pick_seg)
        gl.addWidget(self.seg_edit, 1)
        gl.addWidget(bs)
        lay.addWidget(g)
        row = QHBoxLayout()
        self.btn_load = QPushButton("Validate && load scan")
        self.btn_load.setObjectName("primary")
        self.btn_load.clicked.connect(self.validate)
        self.btn_next = QPushButton("Continue to analysis  >")
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(lambda: main.show_page("analysis"))
        row.addWidget(self.btn_load)
        row.addWidget(self.btn_next)
        row.addStretch(1)
        lay.addLayout(row)
        split = QHBoxLayout()
        self.info = QTableWidget(0, 2)
        self.info.setHorizontalHeaderLabels(["Scan information", "Value"])
        self.info.horizontalHeader().setStretchLastSection(True)
        self.info.verticalHeader().setVisible(False)
        self.issues = QListWidget()
        split.addWidget(self.info, 1)
        split.addWidget(self.issues, 1)
        lay.addLayout(split, 1)

    def _pick_nifti(self, m):
        p, _ = QFileDialog.getOpenFileName(self, f"Select {MOD_LABEL[m]}", "", "NIfTI (*.nii *.nii.gz)")
        if p:
            self.edits[m].setText(p)

    def _pick_seg(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select segmentation", "", "NIfTI (*.nii *.nii.gz)")
        if p:
            self.seg_edit.setText(p)

    def _autofill(self):
        d = QFileDialog.getExistingDirectory(self, "Folder containing the four modality files")
        if not d:
            return
        found = {}
        for f in sorted(Path(d).glob("*.nii*")):
            m = guess_modality(f)
            if m and m not in found:
                found[m] = str(f)
            elif f.name.lower().endswith(("_seg.nii", "_seg.nii.gz")):
                self.seg_edit.setText(str(f))
        for m, p in found.items():
            self.edits[m].setText(p)
        miss = [MOD_LABEL[m] for m in MODALITIES if m not in found]
        if miss:
            err_box(self, "Incomplete folder", "Could not identify: " + ", ".join(miss) + ". Select those files manually.")

    def _pick_dicom(self, m):
        d = QFileDialog.getExistingDirectory(self, f"DICOM folder for {MOD_LABEL[m]}")
        if not d:
            return
        try:
            series = scan_dicom_folder(d)
            if not series:
                raise InputError("No DICOM images found in that folder.")
            uid = series[0]["uid"]
            if len(series) > 1:
                names = [f"{s['description'] or '(no description)'} - {s['modality']} - {s['n_files']} files [{s['uid'][-8:]}]" for s in series]
                choice, ok = QInputDialog.getItem(self, "Select series", "Folder contains several series:", names, 0, False)
                if not ok:
                    return
                uid = series[names.index(choice)]["uid"]
            chosen = next(s for s in series if s["uid"] == uid)
            if chosen["modality"] != "MR":
                raise InputError(f"Series has Modality={chosen['modality']!r}, not MR - not accepted as an MRI volume.")
            self.dicom[m] = (d, uid)
            self.dicom_lbl[m].setText(f"{chosen['description'] or 'series'} ({chosen['n_files']} slices) - {d}")
        except InputError as e:
            err_box(self, "DICOM series rejected", str(e))

    def validate(self):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            if self.tabs.currentIndex() == 0:
                files = {m: e.text().strip() for m, e in self.edits.items()}
                vol = load_modalities(files)
                cid = Path(files["t1"]).name.split("_t1")[0].split(".")[0]
            else:
                vol = load_modalities_dicom(self.dicom)
                cid = "dicom_case"
            labels, src = None, ""
            if self.seg_edit.text().strip():
                labels, _ = load_label_map(self.seg_edit.text().strip(), like=vol)
                src = self.seg_edit.text().strip()
            self.state.load_volume(vol, labels, src, cid)
        except (InputError, AmbiguousSeriesError) as e:
            QApplication.restoreOverrideCursor()
            self.btn_next.setEnabled(False)
            self.issues.clear()
            self.issues.addItem(QListWidgetItem("ERROR: " + str(e)))
            self.issues.item(0).setForeground(Qt.GlobalColor.red)
            err_box(self, "Input rejected", str(e))
            return
        QApplication.restoreOverrideCursor()
        d = self.state.volume.describe()
        rows = [("Case id", self.state.case_id), ("Matrix", " x ".join(map(str, d["shape"]))), ("Voxel size (mm)", " x ".join(f"{s:.2f}" for s in d["spacing_mm"])),
                ("Orientation", d["orientation"] + (f"  (oblique {d['obliquity_deg']} deg)" if d["obliquity_deg"] > 1 else "")), ("Modalities", ", ".join(d["modalities"])),
                ("Intensity range (T1ce)", f"{self.state.volume.channel('t1ce').min():.0f} - {self.state.volume.channel('t1ce').max():.0f}"),
                ("Annotation supplied", "yes" if self.state.labels is not None else "no")]
        self.info.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self.info.setItem(i, 0, QTableWidgetItem(k))
            self.info.setItem(i, 1, QTableWidgetItem(str(v)))
        self.info.resizeColumnsToContents()
        self.issues.clear()
        ok = True
        for it in self.state.qc:
            li = QListWidgetItem(("ERROR: " if it.severity == "error" else "Warning: " if it.severity == "warning" else "") + it.message)
            li.setForeground({"error": Qt.GlobalColor.red, "warning": Qt.GlobalColor.darkYellow}.get(it.severity, Qt.GlobalColor.darkGray))
            self.issues.addItem(li)
            ok &= it.severity != "error"
        self.btn_next.setEnabled(ok)


class AnalysisPage(QWidget):
    def __init__(self, state: AppState, main):
        super().__init__()
        self.setObjectName("page")
        self.state, self.main = state, main
        self.worker = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 12)
        lay.addWidget(page_header("Analysis workspace", "Scroll or drag in any view to move through the volume; the crosshair is shared by all three planes."))
        bar = QHBoxLayout()
        self.mod = QComboBox()
        for m in MODALITIES:
            self.mod.addItem(MOD_LABEL[m], m)
        self.mod.setCurrentIndex(3)
        self.mod.currentIndexChanged.connect(self._bg)
        self.chk = {r: QCheckBox(r) for r in ("WT", "TC", "ET")}
        for r, c in self.chk.items():
            c.setChecked(True)
            c.setToolTip(REGION_NAMES[r])
            c.toggled.connect(self._overlay)
        self.op = QSlider(Qt.Orientation.Horizontal)
        self.op.setRange(0, 100)
        self.op.setValue(45)
        self.op.setMaximumWidth(140)
        self.op.valueChanged.connect(self._overlay)
        self.tta = QCheckBox("Flip-TTA uncertainty (4x slower)")
        self.minc = QDoubleSpinBox()
        self.minc.setRange(0, 5000)
        self.minc.setDecimals(0)
        self.minc.setSuffix(" mm3")
        self.minc.setToolTip("Optional: drop predicted components smaller than this. 0 (default) removes nothing - removal is logged in the report.")
        for w in (QLabel("Modality"), self.mod, QLabel("   Overlay"), *self.chk.values(), QLabel("opacity"), self.op, QLabel("  Ignore components <"), self.minc, self.tta):
            bar.addWidget(w)
        bar.addStretch(1)
        lay.addLayout(bar)
        self.ortho = OrthoViewer()
        self.ortho.cursor_changed.connect(self._cursor)
        lay.addWidget(self.ortho, 1)
        self.readout = QLabel(" ")
        self.readout.setObjectName("muted")
        self.readout.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.readout)
        row = QHBoxLayout()
        self.btn_model = QPushButton("Run model segmentation")
        self.btn_model.setObjectName("primary")
        self.btn_model.clicked.connect(self.run_model)
        self.btn_annot = QPushButton("Measure supplied annotation (no model)")
        self.btn_annot.clicked.connect(self.run_annotation)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setObjectName("danger")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(lambda: self.worker and self.worker.cancel())
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.status = QLabel("Load a scan on the Upload page.")
        self.status.setObjectName("muted")
        for w in (self.btn_model, self.btn_annot, self.btn_cancel):
            row.addWidget(w)
        row.addWidget(self.bar, 1)
        lay.addLayout(row)
        lay.addWidget(self.status)
        self.banner = QLabel()
        self.banner.setObjectName("disclaimer")
        self.banner.setWordWrap(True)
        self.banner.hide()
        lay.addWidget(self.banner)
        state.volume_changed.connect(self.on_volume)
        state.result_changed.connect(self.on_result)
        state.model_changed.connect(self._buttons)
        self._buttons()

    def _buttons(self):
        s, busy = self.state, self.worker is not None and self.worker.isRunning()
        have = s.volume is not None and not busy
        self.btn_model.setEnabled(have and s.model is not None)
        self.btn_model.setToolTip("" if s.model is not None else "Load a model checkpoint on the Dashboard first")
        self.btn_annot.setEnabled(have and s.labels is not None)
        self.btn_annot.setToolTip("" if s.labels is not None else "Supply a segmentation on the Upload page")
        self.btn_cancel.setEnabled(busy)

    def _bg(self):
        s = self.state
        if s.display is None:
            return
        vol = s.display[MODALITIES.index(self.mod.currentData())]
        self.ortho.set_data(vol, s.result.masks if s.result else None, keep_cursor=True)

    def _overlay(self):
        self.ortho.set_overlay([r for r, c in self.chk.items() if c.isChecked()], self.op.value() / 100)

    def on_volume(self):
        s = self.state
        self.ortho.set_data(s.display[MODALITIES.index(self.mod.currentData())], None)
        self.status.setText(f"Scan loaded: {s.case_id}. Run the model, or measure the supplied annotation.")
        self.bar.setValue(0)
        self.banner.hide()
        self._buttons()
        self._cursor(self.ortho.cursor)

    def on_result(self):
        s = self.state
        if s.result is None:
            return
        self._bg()
        what = "MODEL PREDICTION" if s.result.source == "model" else "SUPPLIED ANNOTATION (not a model output)"
        self.banner.setText(f"Overlay shows a {what}. It is not confirmed anatomy. {RESEARCH_DISCLAIMER}")
        self.banner.show()
        self._buttons()

    def _cursor(self, c):
        s = self.state
        if s.display is None:
            return
        s.cursor = list(c)
        s.cursor_changed.emit(list(c))
        w = apply_affine(s.grid.work_affine, np.array([c]))[0]
        vals = " ".join(f"{m.upper()}={s.display[i][tuple(c)]:.0f}" for i, m in enumerate(MODALITIES))
        lab = ""
        if s.result:
            lab = " | in: " + (",".join(r for r in ("WT", "TC", "ET") if s.result.masks[r][tuple(c)]) or "background")
        self.readout.setText(f"voxel (x,y,z) = {tuple(c)}   scanner RAS = ({w[0]:.1f}, {w[1]:.1f}, {w[2]:.1f}) mm   {vals}{lab}")

    def _start(self, **kw):
        s = self.state
        s.cfg.infer.tta = "flips4" if self.tta.isChecked() else "none"
        s.cfg.min_component_mm3 = float(self.minc.value())
        w = AnalysisWorker(s.volume, s.cfg, case_id=s.case_id, **kw)
        w.progress.connect(lambda f, m: (self.bar.setValue(int(f * 1000)), self.status.setText(f"Running: {m} ({f:.0%})")))
        w.done.connect(self._done)
        w.failed.connect(self._failed)
        w.finished.connect(lambda w=w: self._worker_finished(w))     # drop the reference only once the thread has really ended
        self.worker = w
        self.bar.setValue(0)
        w.start()
        self._buttons()

    def _worker_finished(self, w):
        if self.worker is w:
            self.worker = None
        w.deleteLater()
        self._buttons()

    def run_model(self):
        self._start(model=self.state.model, model_info=self.state.model_info)

    def run_annotation(self):
        self._start(labels=self.state.labels, label_source=self.state.label_source)

    def _done(self, res):
        self.bar.setValue(1000)
        t = res.timings
        self.status.setText(f"Done in {t.get('total_s', 0):.0f} s (inference {t.get('inference_s', 0):.0f} s). Open the 3D view or Results.")
        self.state.set_result(res)

    def _failed(self, msg, user_err):
        self.bar.setValue(0)
        self.status.setText("Stopped: " + msg.splitlines()[0][:160])
        self._buttons()
        err_box(self, "Analysis stopped" if user_err else "Unexpected error", msg if user_err else "An unexpected error occurred.", None if user_err else msg)


MEAS_ROWS = [("Volume (cm3)", "volume_cm3"), ("Volume +/-1 voxel (cm3)", "volume_cm3_boundary_range"), ("Max 3D diameter (mm)", "max_diameter_mm"),
             ("Length L (mm)", "length_mm"), ("Width W (mm)", "width_mm"), ("Height H (mm)", "height_mm"), ("Max axial diameter (mm)", "axial_max_diameter_mm"),
             ("Bounding box R-L, A-P, S-I (mm)", "aabb_extent_mm"), ("Surface area (cm2)", "surface_area_cm2"), ("Centroid RAS (mm)", "centroid_world_mm"),
             ("Connected components", "n_components")]


class Viewer3DPage(QWidget):
    def __init__(self, state: AppState, main):
        super().__init__()
        self.setObjectName("page")
        self.state, self.main = state, main
        lay = QHBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 12)
        left = QVBoxLayout()
        left.addWidget(page_header("3D reconstruction", "Rotate: left-drag - Pan: middle-drag or Shift+drag - Zoom: wheel. Surfaces are marching-cubes meshes in scanner mm."))
        self.view = Viewer3D()
        left.addWidget(self.view, 1)
        self.note = QLabel("")
        self.note.setObjectName("muted")
        self.note.setWordWrap(True)
        left.addWidget(self.note)
        lay.addLayout(left, 3)
        panel = QVBoxLayout()
        g = QGroupBox("Display")
        gl = QGridLayout(g)
        self.chk, self.sl = {}, {}
        for i, (key, label) in enumerate([("WT", "Whole tumour"), ("TC", "Tumour core"), ("ET", "Enhancing"), ("brain", "Brain (reference)")]):
            c = QCheckBox(label)
            c.setChecked(True)
            c.toggled.connect(lambda on, k=key: self.view.set_visible(k, on))
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(0, 100)
            s.setValue(int(100 * Viewer3D.DEFAULT_OPACITY.get(key, 0.10)))
            s.valueChanged.connect(lambda v, k=key: self.view.set_opacity(k, v / 100))
            gl.addWidget(c, i, 0)
            gl.addWidget(s, i, 1)
            self.chk[key], self.sl[key] = c, s
        self.c_feret = QCheckBox("Max-diameter line")
        self.c_feret.setChecked(True)
        self.c_feret.toggled.connect(lambda on: self.view.set_visible("feret", on))
        self.c_planes = QCheckBox("MRI slice planes at crosshair")
        self.c_planes.toggled.connect(self._planes)
        gl.addWidget(self.c_feret, 4, 0, 1, 2)
        gl.addWidget(self.c_planes, 5, 0, 1, 2)
        self.smooth = QSpinBox()
        self.smooth.setRange(0, 50)
        self.smooth.setToolTip("Display-only Taubin smoothing iterations (0 = faithful surface). Never used for measurements.")
        b_s = QPushButton("Apply")
        b_s.clicked.connect(lambda: self._rebuild())
        gl.addWidget(QLabel("Display smoothing iterations"), 6, 0)
        r = QHBoxLayout()
        r.addWidget(self.smooth)
        r.addWidget(b_s)
        gl.addLayout(r, 6, 1)
        b_r = QPushButton("Reset camera")
        b_r.clicked.connect(self.view.reset_camera)
        gl.addWidget(b_r, 7, 0, 1, 2)
        panel.addWidget(g)
        self.table = QTableWidget(len(MEAS_ROWS), 3)
        self.table.setHorizontalHeaderLabels(["WT", "TC", "ET"])
        self.table.setVerticalHeaderLabels([r[0] for r in MEAS_ROWS])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setMinimumWidth(190)
        self.table.setWordWrap(True)
        self.table.setTextElideMode(Qt.TextElideMode.ElideNone)
        panel.addWidget(self.table, 1)
        w = QWidget()
        w.setLayout(panel)
        w.setMaximumWidth(560)
        lay.addWidget(w, 1)
        state.result_changed.connect(self.on_result)
        state.cursor_changed.connect(lambda c: self.c_planes.isChecked() and self._planes())

    def _rebuild(self):
        if self.state.result is not None:
            self.view.show_result(self.state.result, self.smooth.value())
            for k, c in self.chk.items():
                self.view.set_visible(k, c.isChecked())
                self.view.set_opacity(k, self.sl[k].value() / 100)
            self._planes()
            self.note.setText("Display meshes are smoothed (volume-preserving rescale) - measurements use the unsmoothed surface." if self.smooth.value()
                              else "Faithful marching-cubes surfaces (no smoothing).")

    def _planes(self):
        s = self.state
        if s.result is None:
            return
        vol = s.display[MODALITIES.index("flair")] if s.display is not None else None
        from ..viz import intensity_window
        self.view.set_planes(self.c_planes.isChecked(), vol, s.cursor, intensity_window(vol) if vol is not None else (0, 1))

    def on_result(self):
        s = self.state
        if s.result is None:
            self.view.clear()
            return
        self._rebuild()
        from ..measure import round_for_display
        for j, r in enumerate(("WT", "TC", "ET")):
            m = round_for_display(s.result.measurements[r])
            for i, (_, k) in enumerate(MEAS_ROWS):
                v = m.get(k, "-") if m.get("present") else ("absent" if i == 0 else "")
                if isinstance(v, (list, tuple)):
                    v = " - ".join(str(x) for x in v) if k == "volume_cm3_boundary_range" else ", ".join(str(x) for x in v)
                self.table.setItem(i, j, QTableWidgetItem(str(v)))
        self.table.resizeRowsToContents()


class ResultsPage(QWidget):
    def __init__(self, state: AppState, main):
        super().__init__()
        self.setObjectName("page")
        self.state = state
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 12)
        lay.addWidget(page_header("Results and evaluation", "Predictions, measurements, validation evidence, warnings and limitations - kept separate on purpose."))
        self.badge = QLabel()
        lay.addWidget(self.badge)
        self.text = QTextBrowser()
        lay.addWidget(self.text, 1)
        state.result_changed.connect(self.refresh)
        self.refresh()

    def refresh(self):
        r = self.state.result
        if r is None:
            self.text.setHtml("<p style='color:#5b6b78'>No analysis yet.</p>")
            self.badge.setText("")
            return
        rep = build_report(r)
        lvl = rep["7_validation"]["evidence_level"]
        self.badge.setText(f"<span style='{evidence_style(lvl)}'>&nbsp;Model evidence: {lvl}&nbsp;</span>&nbsp; "
                           f"<small>{len(rep['8_quality_and_uncertainty']['warnings'])} warning(s)</small>")
        self.text.setHtml(render_html(rep, compact=True))


class ExportPage(QWidget):
    def __init__(self, state: AppState, main):
        super().__init__()
        self.setObjectName("page")
        self.state, self.main = state, main
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 12)
        lay.addWidget(page_header("Export", "Masks are written on the ORIGINAL scan grid with the original affine; meshes are in scanner RAS+ millimetres. Existing files are never overwritten silently."))
        g = QGroupBox("Contents")
        fl = QFormLayout(g)
        self.items = {"mask": QCheckBox("Segmentation mask (NIfTI, BraTS labels 1/2/4)"), "meshes": QCheckBox("3D meshes (WT, TC, ET)"),
                      "report": QCheckBox("Report (JSON + HTML)"), "csv": QCheckBox("Measurements (CSV)"), "images": QCheckBox("Images (3 planes + 3D view)")}
        for c in self.items.values():
            c.setChecked(True)
            fl.addRow(c)
        self.fmt = QComboBox()
        self.fmt.addItems(["stl", "obj", "ply"])
        self.sm = QSpinBox()
        self.sm.setRange(0, 50)
        fl.addRow("Mesh format", self.fmt)
        fl.addRow("Mesh smoothing iterations (0 = faithful)", self.sm)
        lay.addWidget(g)
        row = QHBoxLayout()
        self.out = QLineEdit(str(PROJECT_ROOT / "exports"))
        b = QPushButton("Choose folder...")
        b.clicked.connect(self._choose)
        row.addWidget(QLabel("Output folder"))
        row.addWidget(self.out, 1)
        row.addWidget(b)
        lay.addLayout(row)
        self.go = QPushButton("Export")
        self.go.setObjectName("primary")
        self.go.setMaximumWidth(160)
        self.go.clicked.connect(self.export)
        lay.addWidget(self.go)
        self.log = QTextBrowser()
        lay.addWidget(self.log, 1)
        state.result_changed.connect(lambda: self.go.setEnabled(self.state.result is not None))
        self.go.setEnabled(False)

    def _choose(self):
        d = QFileDialog.getExistingDirectory(self, "Output folder", self.out.text())
        if d:
            self.out.setText(d)

    def export(self):
        s = self.state
        sel = tuple(k for k, c in self.items.items() if c.isChecked())
        extra = {}
        if "images" in sel:
            self.main.show_page("3d")
            QApplication.processEvents()
            png = self.main.page3d.view.screenshot_png()
            if png:
                extra["3d_view"] = png
        out = Path(self.out.text()) / s.result.case_id
        for overwrite in (False, True):
            try:
                w = export_all(s.result, out, sel, self.fmt.currentText(), self.sm.value(), overwrite=overwrite, extra_images=extra)
                break
            except FileExistsError as e:
                if QMessageBox.question(self, "Files exist", f"{e}\n\nOverwrite existing files?") != QMessageBox.StandardButton.Yes:
                    self.log.append("Export cancelled (nothing overwritten).")
                    return
        self.log.append(f"<b>Exported to {out}</b>")
        for k, v in w.items():
            for p in v:
                self.log.append(f"&nbsp;&nbsp;{k}: {Path(p).name}")
        try:
            os.startfile(out)  # type: ignore[attr-defined]
        except Exception:
            pass


# ----------------------------------------------------------------------------- main window

class MainWindow(QMainWindow):
    PAGES = [("dashboard", "Dashboard"), ("upload", "MRI Upload"), ("analysis", "Analysis"), ("3d", "3D View"), ("results", "Results"), ("export", "Export")]

    def __init__(self, state: AppState | None = None):
        super().__init__()
        self.state = state or AppState()
        self.setWindowTitle("Brain MRI Tumour Analysis - research prototype")
        self.resize(1420, 880)
        root = QWidget()
        h = QHBoxLayout(root)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        side = QWidget()
        side.setObjectName("sidebar")
        side.setFixedWidth(210)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 0, 0)
        b = QLabel("Brain MRI\nTumour Analysis")
        b.setObjectName("brand")
        sb = QLabel("Research prototype - not a medical device")
        sb.setObjectName("brandSub")
        sb.setWordWrap(True)
        self.nav = QListWidget()
        self.nav.setObjectName("nav")
        for key, label in self.PAGES:
            self.nav.addItem(QListWidgetItem(label))
        self.nav.currentRowChanged.connect(self._nav)
        sl.addWidget(b)
        sl.addWidget(sb)
        sl.addWidget(self.nav, 1)
        self.evid = QLabel()
        self.evid.setWordWrap(True)
        self.evid.setStyleSheet("color:#9fb6cc;padding:10px 14px;font-size:8.5pt;")
        sl.addWidget(self.evid)
        h.addWidget(side)
        self.stack = QStackedWidget()
        self.pg = {}
        self.page3d = Viewer3DPage(self.state, self)
        for key, cls in (("dashboard", DashboardPage), ("upload", UploadPage), ("analysis", AnalysisPage), ("3d", None), ("results", ResultsPage), ("export", ExportPage)):
            w = self.page3d if key == "3d" else cls(self.state, self)
            self.pg[key] = w
            self.stack.addWidget(w)
        h.addWidget(self.stack, 1)
        self.setCentralWidget(root)
        self.setStatusBar(QStatusBar())
        self.state.volume_changed.connect(self._enable)
        self.state.result_changed.connect(self._enable)
        self.state.model_changed.connect(self._enable)
        self._enable()
        self.nav.setCurrentRow(0)
        QTimer.singleShot(0, self._auto_model)

    def _auto_model(self):
        """Load a checkpoint automatically only if exactly one candidate exists (never guess between several)."""
        cks = find_checkpoints(self.state.weights_dir)
        if len(cks) == 1:
            try:
                self.state.load_model(str(cks[0]))
            except CheckpointError as e:
                self.statusBar().showMessage(f"Checkpoint not loaded: {e}", 8000)

    def _enable(self):
        s = self.state
        enabled = {"dashboard": True, "upload": True, "analysis": s.volume is not None, "3d": s.result is not None, "results": s.result is not None,
                   "export": s.result is not None}
        for i, (k, _) in enumerate(self.PAGES):
            it = self.nav.item(i)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsEnabled if enabled[k] else it.flags() & ~Qt.ItemFlag.ItemIsEnabled)
        lvl = s.evidence_level()
        self.evid.setText(f"Model: {Path(s.model_info['path']).name if s.model_info else 'none'}<br>Evidence: <b>{lvl}</b>")
        self.statusBar().showMessage(f"Case: {s.case_id if s.volume is not None else '-'}   |   {RESEARCH_DISCLAIMER.split('.')[0]}.")

    def _nav(self, row):
        if row >= 0:
            self.stack.setCurrentIndex(row)

    def show_page(self, key: str) -> None:
        self.nav.setCurrentRow([k for k, _ in self.PAGES].index(key))

    def closeEvent(self, e):
        w = self.pg.get("analysis")
        if w is not None and w.worker is not None and w.worker.isRunning():
            w.worker.cancel()
            w.worker.wait(5000)
        self.page3d.view.shutdown()
        super().closeEvent(e)


def main(argv=None) -> int:
    app = QApplication(sys.argv[:1])
    app.setStyleSheet(QSS)
    app.setFont(QFont("Segoe UI", 10))
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

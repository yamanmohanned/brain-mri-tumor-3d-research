"""Background analysis thread (keeps the UI responsive and cancellable)."""
from __future__ import annotations

import traceback

from PyQt6.QtCore import QThread, pyqtSignal

from ..inference import Cancelled
from ..io import InputError
from ..pipeline import analyze


class AnalysisWorker(QThread):
    progress = pyqtSignal(float, str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str, bool)      # message, is_user_error

    def __init__(self, volume, cfg, model=None, model_info=None, labels=None, label_source="", case_id="case"):
        super().__init__()
        self._args = dict(vol=volume, model=model, model_info=model_info, cfg=cfg, labels=labels, label_source=label_source, case_id=case_id)
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            self.progress.emit(0.0, "preprocessing")
            res = analyze(progress=lambda f, m: self.progress.emit(f, m), cancel=lambda: self._cancel, **self._args)
            self.done.emit(res)
        except Cancelled:
            self.failed.emit("Analysis cancelled.", True)
        except (InputError, ValueError) as exc:
            self.failed.emit(str(exc), True)
        except Exception:
            self.failed.emit(traceback.format_exc(), False)

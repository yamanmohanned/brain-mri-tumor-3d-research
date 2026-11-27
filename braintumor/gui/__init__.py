"""Desktop GUI (PyQt6 + pyvistaqt). torch is imported first - see braintumor/_preload.py."""
import os

from .. import _preload  # noqa: F401  (must precede any Qt import)

os.environ.setdefault("QT_API", "pyqt6")

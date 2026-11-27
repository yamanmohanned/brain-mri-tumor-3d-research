"""Import torch before any Qt binding.

On this Windows setup, loading PyQt6 first makes ``import torch`` fail with
``WinError 1114`` (c10.dll initialisation) - the two bundle conflicting MSVC runtime DLLs.
Importing torch first is harmless. pytest loads this via ``-p braintumor._preload`` (see pyproject.toml)
because pytest-qt imports Qt while plugins are loaded; the GUI entry point imports it explicitly.
"""
try:
    import torch  # noqa: F401
except Exception:  # pragma: no cover - torch genuinely missing; callers report this themselves
    pass

"""MRI loading and validation (NIfTI and DICOM series).

Nothing here silently repairs or guesses: inputs that are not a coherent MR volume raise ``InputError``
with a message that says what is wrong.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np

from . import MODALITIES
from .geometry import axcodes, obliquity_deg, voxel_sizes


class InputError(ValueError):
    """The input cannot be used for analysis (message is meant for the user)."""


class AmbiguousSeriesError(InputError):
    """A DICOM folder holds several series; the caller must pick one."""

    def __init__(self, series: list[dict]):
        self.series = series
        listing = "; ".join(
            f"[{s['uid'][-8:]}] {s['description']!r} ({s['modality']}, {s['n_files']} files)" for s in series
        )
        super().__init__(f"Folder contains {len(series)} DICOM series - select one: {listing}")


@dataclass
class Issue:
    severity: str  # "error" | "warning" | "info"
    message: str


@dataclass
class MRVolume:
    """Co-registered multi-modality scan on one voxel grid, in the ORIGINAL file orientation."""

    data: np.ndarray                 # (C, X, Y, Z) float32
    affine: np.ndarray               # 4x4 voxel -> scanner RAS+ mm
    modalities: tuple[str, ...]
    sources: dict = field(default_factory=dict)   # modality -> file/folder
    header_notes: list = field(default_factory=list)

    @property
    def shape(self) -> tuple:
        return tuple(self.data.shape[1:])

    @property
    def spacing(self) -> np.ndarray:
        return voxel_sizes(self.affine)

    def channel(self, name: str) -> np.ndarray:
        return self.data[self.modalities.index(name)]

    def describe(self) -> dict:
        return {
            "shape": list(self.shape),
            "spacing_mm": [round(float(s), 4) for s in self.spacing],
            "orientation": axcodes(self.affine),
            "obliquity_deg": round(obliquity_deg(self.affine), 2),
            "modalities": list(self.modalities),
            "sources": {k: str(v) for k, v in self.sources.items()},
            "affine": np.asarray(self.affine, float).round(6).tolist(),
        }


# ----------------------------------------------------------------------------- NIfTI

_MOD_PATTERNS = [  # order matters: most specific first
    ("flair", r"flair|t2f"),
    ("t1ce", r"t1ce|t1c|t1gd|t1_ce|t1post"),
    ("t2", r"t2w|t2"),
    ("t1", r"t1n|t1w|t1"),
]


def guess_modality(path: str | Path) -> str | None:
    """Guess the modality from a file name using BraTS-style tokens (``..._t1ce.nii.gz``)."""
    name = Path(path).name.lower()
    for mod, pat in _MOD_PATTERNS:
        if re.search(rf"(?<![a-z0-9])({pat})(?![a-z0-9])", name):
            return mod
    return None


def load_nifti(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load one 3D NIfTI -> (float32 array, affine, notes)."""
    path = Path(path)
    if not path.is_file():
        raise InputError(f"File not found: {path}")
    try:
        img = nib.load(str(path))
    except Exception as exc:  # nibabel raises several types
        raise InputError(f"{path.name}: not a readable NIfTI file ({exc})") from exc
    shape = img.shape
    if not (len(shape) == 3 or (len(shape) == 4 and shape[3] == 1)):
        raise InputError(f"{path.name}: expected a 3D volume, found shape {shape}")
    try:
        data = np.asarray(img.get_fdata(dtype=np.float32)).reshape(shape[:3])
    except Exception as exc:
        raise InputError(f"{path.name}: data are corrupted or truncated ({exc})") from exc
    if not np.isfinite(data).all():
        raise InputError(f"{path.name}: contains NaN/Inf voxels")
    aff = np.asarray(img.affine, float)
    notes = []
    hdr = img.header
    try:
        if int(hdr["sform_code"]) == 0 and int(hdr["qform_code"]) == 0:
            notes.append(f"{path.name}: header has no sform/qform - position/orientation are NOT defined "
                         "(voxel-size-only affine assumed)")
    except (KeyError, ValueError):
        pass
    sizes = voxel_sizes(aff)
    if not np.all(np.isfinite(aff)) or abs(np.linalg.det(aff[:3, :3])) < 1e-9 or np.any(sizes <= 0):
        raise InputError(f"{path.name}: degenerate spatial transform (affine is singular)")
    if np.any(sizes > 20):
        raise InputError(f"{path.name}: implausible voxel size {sizes.round(2).tolist()} mm")
    return data, aff, notes


def load_modalities(files: dict[str, str | Path]) -> MRVolume:
    """Load the four model modalities from NIfTI files and verify they share one grid."""
    missing = [m for m in MODALITIES if m not in files or not files[m]]
    if missing:
        raise InputError(
            "Missing required modalities: " + ", ".join(m.upper() for m in missing)
            + f". The model needs all of {', '.join(m.upper() for m in MODALITIES)}."
        )
    arrays, notes = [], []
    ref_aff = ref_shape = ref_mod = None
    for mod in MODALITIES:
        data, aff, n = load_nifti(files[mod])
        notes += n
        if ref_aff is None:
            ref_aff, ref_shape, ref_mod = aff, data.shape, mod
        else:
            if data.shape != ref_shape:
                raise InputError(
                    f"{mod.upper()} shape {data.shape} differs from {ref_mod.upper()} shape {ref_shape}: modalities must be "
                    "co-registered on the same voxel grid (resample/register them first)."
                )
            if not np.allclose(aff, ref_aff, atol=1e-3):
                raise InputError(
                    f"{mod.upper()} spatial transform differs from {ref_mod.upper()} (orientation/position/spacing "
                    "mismatch): modalities must be co-registered on the same grid."
                )
        arrays.append(data)
    return MRVolume(np.stack(arrays), ref_aff, tuple(MODALITIES), {m: files[m] for m in MODALITIES}, notes)


def load_label_map(path: str | Path, like: MRVolume | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load an integer segmentation (e.g. BraTS ground truth). Verifies it matches ``like``'s grid."""
    data, aff, _ = load_nifti(path)
    if np.abs(data - np.round(data)).max() > 1e-3:
        raise InputError(f"{Path(path).name}: not an integer label map")
    lab = np.round(data).astype(np.int16)
    vals = set(np.unique(lab).tolist())
    if not vals <= {0, 1, 2, 3, 4}:
        raise InputError(f"{Path(path).name}: unexpected label values {sorted(vals)} (expected BraTS labels 0,1,2,4)")
    if like is not None and (lab.shape != like.shape or not np.allclose(aff, like.affine, atol=1e-3)):
        raise InputError("Segmentation grid (shape/affine) does not match the MRI volume.")
    return lab, aff


# ----------------------------------------------------------------------------- DICOM

def _pydicom():
    import pydicom  # imported lazily: only needed when DICOM is used
    return pydicom


def scan_dicom_folder(folder: str | Path) -> list[dict]:
    """List the image series in a folder (recursive). Files that are not DICOM are ignored and counted."""
    pd = _pydicom()
    series: dict[str, dict] = {}
    skipped = 0
    for f in sorted(Path(folder).rglob("*")):
        if not f.is_file():
            continue
        try:
            ds = pd.dcmread(str(f), stop_before_pixels=True)
            uid = str(ds.SeriesInstanceUID)
        except Exception:
            skipped += 1
            continue
        s = series.setdefault(uid, {
            "uid": uid, "description": str(getattr(ds, "SeriesDescription", "")),
            "modality": str(getattr(ds, "Modality", "?")), "files": [],
        })
        s["files"].append(str(f))
    out = list(series.values())
    for s in out:
        s["n_files"] = len(s["files"])
        s["skipped"] = skipped
    return out


def load_dicom_series(folder: str | Path, series_uid: str | None = None, min_slices: int = 10):
    """Load one MR DICOM series into (float32 [X,Y,Z], RAS affine, info).

    Raises ``InputError`` if the series is not MR, mixes geometries, has duplicate/missing slices or is too
    small to be a head volume, and ``AmbiguousSeriesError`` if the folder holds several series and
    ``series_uid`` is not given.
    """
    pd = _pydicom()
    found = scan_dicom_folder(folder)
    if not found:
        raise InputError(f"No DICOM images found in {folder}")
    if series_uid is None:
        if len(found) > 1:
            raise AmbiguousSeriesError(found)
        chosen = found[0]
    else:
        matches = [s for s in found if s["uid"] == series_uid or s["uid"].endswith(series_uid)]
        if len(matches) != 1:
            raise InputError(f"Series {series_uid} not found in folder")
        chosen = matches[0]
    if chosen["modality"] != "MR":
        raise InputError(f"Series {chosen['description']!r} has Modality={chosen['modality']!r}, not MR - "
                         "refusing to treat it as an MRI volume.")
    ds = [pd.dcmread(f) for f in chosen["files"]]
    if len(ds) < min_slices:
        raise InputError(f"Series has only {len(ds)} slices (< {min_slices}); not a 3D volume")
    for need in ("ImageOrientationPatient", "ImagePositionPatient", "PixelSpacing", "Rows", "Columns"):
        if any(need not in d for d in ds):
            raise InputError(f"DICOM slices lack required attribute {need}; cannot determine geometry")
    rows, cols = int(ds[0].Rows), int(ds[0].Columns)
    iop0 = np.array(ds[0].ImageOrientationPatient, float)
    for d in ds:
        if (int(d.Rows), int(d.Columns)) != (rows, cols):
            raise InputError("Slices have different matrix sizes")
        if not np.allclose(np.array(d.ImageOrientationPatient, float), iop0, atol=1e-3):
            raise InputError("Slices have different orientations (gantry tilt / mixed series) - cannot build a regular volume")
    row_dir, col_dir = iop0[:3], iop0[3:]
    normal = np.cross(row_dir, col_dir)
    pos = np.array([d.ImagePositionPatient for d in ds], float)
    dist = pos @ normal
    order = np.argsort(dist)
    ds, pos, dist = [ds[i] for i in order], pos[order], dist[order]
    gaps = np.diff(dist)
    if np.any(gaps < 1e-3):
        raise InputError("Duplicate slice positions in series (multiple volumes/echoes mixed?)")
    step = float(np.median(gaps))
    if np.max(np.abs(gaps - step)) > 0.05 * step:
        raise InputError(f"Non-uniform slice spacing ({gaps.min():.2f}-{gaps.max():.2f} mm): missing slices - "
                         "refusing to build a distorted volume")
    ps = np.array(ds[0].PixelSpacing, float)  # [row spacing, column spacing]
    vol = np.empty((rows, cols, len(ds)), np.float32)
    for k, d in enumerate(ds):
        vol[:, :, k] = d.pixel_array.astype(np.float32) * float(getattr(d, "RescaleSlope", 1.0)) \
            + float(getattr(d, "RescaleIntercept", 0.0))
    if not np.isfinite(vol).all():
        raise InputError("DICOM pixel data contain NaN/Inf")
    vol = vol.transpose(1, 0, 2)  # -> [column, row, slice]
    lps = np.eye(4)
    lps[:3, 0] = row_dir * ps[1]   # step along increasing column index
    lps[:3, 1] = col_dir * ps[0]   # step along increasing row index
    lps[:3, 2] = normal * step     # step along sorted slice index
    lps[:3, 3] = pos[0]
    affine = np.diag([-1.0, -1.0, 1.0, 1.0]) @ lps  # DICOM LPS+ -> RAS+
    info = {"series_uid": chosen["uid"], "description": chosen["description"], "n_slices": len(ds),
            "slice_spacing_mm": step, "modality": "MR", "files": len(chosen["files"])}
    return vol, affine, info


def load_modalities_dicom(series: dict[str, tuple]) -> MRVolume:
    """``series`` maps modality -> (folder, series_uid or None)."""
    missing = [m for m in MODALITIES if m not in series]
    if missing:
        raise InputError("Missing required modalities: " + ", ".join(m.upper() for m in missing))
    files, arrays, ref = {}, [], None
    for mod in MODALITIES:
        folder, uid = series[mod]
        data, aff, info = load_dicom_series(folder, uid)
        files[mod] = f"{folder}#{info['series_uid'][-8:]}"
        if ref is None:
            ref = (aff, data.shape, mod)
        elif data.shape != ref[1] or not np.allclose(aff, ref[0], atol=1e-2):
            raise InputError(f"{mod.upper()} series geometry differs from {ref[2].upper()}: modalities must be co-registered "
                             "on one grid (register/resample first; this tool does not guess).")
        arrays.append(data)
    return MRVolume(np.stack(arrays), ref[0], tuple(MODALITIES), files, [])


# ----------------------------------------------------------------------------- input QC

def assess_input(vol: MRVolume) -> list[Issue]:
    """Technical-suitability checks. 'error' issues must stop the analysis."""
    out: list[Issue] = []
    sp = vol.spacing
    out.append(Issue("info", f"Grid {vol.shape}, voxel {np.round(sp, 3).tolist()} mm, orientation {axcodes(vol.affine)}"))
    if obliquity_deg(vol.affine) > 1.0:
        out.append(Issue("warning", f"Scan axes are oblique ({obliquity_deg(vol.affine):.1f} deg from scanner axes); "
                                    "measurements use true physical coordinates, axis-aligned boxes are in scanner axes."))
    if np.any(np.abs(sp - 1.0) > 0.05):
        out.append(Issue("warning", f"Voxel size {np.round(sp, 2).tolist()} mm is not 1 mm isotropic: data will be resampled to "
                                    "1 mm (model was trained on 1 mm isotropic BraTS data); accuracy is not validated for this case."))
    if sp.max() > 3.0:
        out.append(Issue("warning", f"Coarse voxel dimension ({sp.max():.1f} mm): small structures cannot be resolved."))
    for i, m in enumerate(vol.modalities):
        c = vol.data[i]
        if not np.any(c):
            out.append(Issue("error", f"{m.upper()} volume is empty (all zeros)."))
        elif float(c.std()) == 0.0:
            out.append(Issue("error", f"{m.upper()} volume is constant."))
        elif c.min() < 0:
            out.append(Issue("warning", f"{m.upper()} contains negative values; model preprocessing treats only >0 as foreground."))
    fg = (vol.data > 0).any(axis=0)
    vox_cm3 = float(abs(np.linalg.det(vol.affine[:3, :3]))) / 1000.0
    fg_cm3 = float(fg.sum()) * vox_cm3
    zero_frac = float((vol.data == 0).all(axis=0).mean())
    if fg_cm3 > 2200 or zero_frac < 0.3:
        out.append(Issue("warning", f"Foreground is {fg_cm3:.0f} cm3 and {zero_frac:.0%} of voxels are exactly zero: the scan does "
                                    "not look skull-stripped/masked like the BraTS training data. Predictions are out-of-distribution "
                                    "and unreliable."))
    if 0 < fg_cm3 < 300:
        out.append(Issue("warning", f"Foreground is only {fg_cm3:.0f} cm3 - unusually small for a whole head/brain volume."))
    if min(vol.shape) < 32:
        out.append(Issue("error", f"Volume too small ({vol.shape})."))
    for n in vol.header_notes:
        out.append(Issue("warning", n))
    return out

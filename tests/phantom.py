"""SYNTHETIC test fixtures. These are NOT MRI data and must never be used as experimental evidence.

A phantom is an ellipsoidal "brain" with nested ellipsoidal "tumour" regions (BraTS label convention:
1 = core, 2 = oedema, 4 = enhancing). Because the shapes are analytic, volumes / diameters / centroids
have exact ground truth against which measurement and geometry code can be verified.
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch

MODS = ("t1", "t1ce", "t2", "flair")


def _rot_z(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def make_affine(kind: str, shape, spacing) -> np.ndarray:
    """Return a voxel->RAS affine. kinds: ras, las, lps, permuted, oblique."""
    sp = np.asarray(spacing, float)
    if kind == "ras":
        lin = np.diag(sp)
    elif kind == "las":
        lin = np.diag(sp * [-1, 1, 1])
    elif kind == "lps":
        lin = np.diag(sp * [-1, -1, 1])
    elif kind == "permuted":  # array axes = (S, R, A)
        lin = np.zeros((3, 3))
        lin[2, 0], lin[0, 1], lin[1, 2] = sp[0], sp[1], sp[2]
    elif kind == "oblique":
        lin = _rot_z(15) @ np.diag(sp)
    else:
        raise ValueError(kind)
    aff = np.eye(4)
    aff[:3, :3] = lin
    aff[:3, 3] = np.array([-30.0, -40.0, -20.0]) + lin @ np.zeros(3)
    return aff


def ellipsoid(shape, spacing, center_mm, radii_mm):
    """Boolean mask of an ellipsoid defined in array-frame physical coordinates (index * spacing)."""
    ax = [np.arange(n) * s for n, s in zip(shape, spacing)]
    x, y, z = np.meshgrid(*ax, indexing="ij")
    c, r = center_mm, radii_mm
    return ((x - c[0]) / r[0]) ** 2 + ((y - c[1]) / r[1]) ** 2 + ((z - c[2]) / r[2]) ** 2 <= 1.0


def make_phantom(shape=(72, 80, 64), spacing=(1.0, 1.0, 1.0), affine_kind="ras", wt_radii=(14.0, 11.0, 9.0),
                 center_frac=(0.55, 0.5, 0.5), n_et=True, seed=0, second_blob=False):
    """Return dict(data (4,X,Y,Z), affine, seg, truth). ``truth`` holds analytic quantities in mm / mm^3."""
    rng = np.random.default_rng(seed)
    sp = np.asarray(spacing, float)
    fov = np.array(shape) * sp
    ctr = fov / 2
    brain = ellipsoid(shape, sp, ctr, fov * 0.42)
    wt_c = fov * np.asarray(center_frac)
    wt = ellipsoid(shape, sp, wt_c, wt_radii) & brain
    tc = ellipsoid(shape, sp, wt_c, 0.6 * np.asarray(wt_radii)) & wt
    et = ellipsoid(shape, sp, wt_c + 0.15 * np.asarray(wt_radii), 0.3 * np.asarray(wt_radii)) & tc if n_et else np.zeros_like(wt)
    seg = np.zeros(shape, np.int16)
    seg[wt] = 2
    seg[tc] = 1
    seg[et] = 4
    if second_blob:  # a small separate lesion (multifocal case)
        blob = ellipsoid(shape, sp, ctr - fov * np.array([0.22, 0.2, 0.1]), (3.0, 3.0, 3.0)) & brain & (seg == 0)
        seg[blob] = 2
    data = np.zeros((4, *shape), np.float32)
    base = {"t1": 90, "t1ce": 95, "t2": 110, "flair": 100}
    inc = {"t1": (-3, -30, -30), "t1ce": (0, 10, 90), "t2": (50, 40, 35), "flair": (50, 40, 35)}   # (oedema, core, ET) increments
    for i, m in enumerate(MODS):
        v = base[m] + rng.normal(0, 4, shape)
        v[seg == 2] += inc[m][0]
        v[seg == 1] += inc[m][1]
        v[seg == 4] += inc[m][2]
        data[i] = np.where(brain, np.clip(v, 1, None), 0).astype(np.float32)
    affine = make_affine(affine_kind, shape, sp)
    voxel_mm3 = float(np.prod(sp))
    truth = {
        "WT_volume_mm3": 4 / 3 * np.pi * float(np.prod(wt_radii)),
        "WT_voxels_volume_mm3": float((seg > 0).sum()) * voxel_mm3,
        "WT_max_diameter_mm": 2 * max(wt_radii),
        "WT_centroid_array_mm": wt_c,
        "WT_extents_mm": 2 * np.asarray(wt_radii),
        "voxel_mm3": voxel_mm3,
    }
    return {"data": data, "affine": affine, "seg": seg, "truth": truth, "spacing": sp}


def save_phantom(ph: dict, folder: Path, case_id: str, with_seg=True) -> dict:
    """Write 4 modality NIfTIs (+ seg) with BraTS-style names; return the file dict."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    for i, m in enumerate(MODS):
        p = folder / f"{case_id}_{m}.nii.gz"
        nib.save(nib.Nifti1Image(ph["data"][i], ph["affine"]), str(p))
        files[m] = p
    if with_seg:
        p = folder / f"{case_id}_seg.nii.gz"
        nib.save(nib.Nifti1Image(ph["seg"].astype(np.int16), ph["affine"]), str(p))
        files["seg"] = p
    return files


def make_brats_like_dataset(root: Path, n: int = 12, shape=(48, 52, 40), seed=0) -> list[str]:
    """Tiny BraTS-structured directory (BraTS20_Training_001 ...) of phantoms with varied tumour sizes."""
    rng = np.random.default_rng(seed)
    ids = []
    for i in range(1, n + 1):
        cid = f"BraTS20_Training_{i:03d}"
        r = rng.uniform(5, 9, 3)
        ph = make_phantom(shape, (1, 1, 1), "ras", tuple(r), center_frac=tuple(rng.uniform(0.4, 0.6, 3)), n_et=bool(i % 4),
                          seed=int(rng.integers(1e6)))
        save_phantom(ph, Path(root) / cid, cid)
        ids.append(cid)
    return ids


def write_dicom_series(folder: Path, vol: np.ndarray, spacing=(1.0, 1.0, 2.0), iop=(1, 0, 0, 0, 1, 0), origin=(-60.0, -70.0, -30.0),
                       modality="MR", series_uid=None, drop_slice=None, desc="T1"):
    """Write a synthetic DICOM series (int16). vol is [col,row,slice] -> stored rows x cols per slice."""
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    series_uid = series_uid or generate_uid()
    study = generate_uid()
    iop = np.asarray(iop, float)
    normal = np.cross(iop[:3], iop[3:])
    ncol, nrow, nsl = vol.shape
    for k in range(nsl):
        if drop_slice is not None and k == drop_slice:
            continue
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(str(folder / f"s{k:03d}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.Modality = modality
        ds.SeriesInstanceUID = series_uid
        ds.StudyInstanceUID = study
        ds.SeriesDescription = desc
        ds.Rows, ds.Columns = nrow, ncol
        ds.PixelSpacing = [spacing[1], spacing[0]]  # [row spacing, column spacing]
        ds.ImageOrientationPatient = list(map(float, iop))
        ds.ImagePositionPatient = list(map(float, np.asarray(origin) + normal * spacing[2] * k))
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.RescaleSlope, ds.RescaleIntercept = 1.0, 0.0
        ds.PixelData = np.ascontiguousarray(vol[:, :, k].T).astype(np.int16).tobytes()
        ds.save_as(str(folder / f"s{k:03d}.dcm"), enforce_file_format=True)
    return series_uid


class ThresholdNet(torch.nn.Module):
    """Deterministic stand-in 'model' for the phantom: TC from T1 (dark), WT from FLAIR, ET from T1ce, in z-score space."""

    def __init__(self, raw: np.ndarray):
        super().__init__()
        nz = [raw[c][raw[c] != 0] for c in range(4)]
        mu = [float(v.mean()) for v in nz]
        sd = [float(v.std()) for v in nz]
        z = lambda c, thr: (thr - mu[c]) / sd[c]
        self.t_tc, self.t_wt, self.t_et = z(0, 90 - 16.5), z(3, 100 + 20), z(1, 95 + 50)
        self.sd = sd

    def forward(self, x):
        wt = (x[:, 3:4] - self.t_wt) * 5
        tc = torch.minimum((self.t_tc - x[:, 0:1]) * 5, wt)       # dark T1 AND tumour-level FLAIR (blurred brain rim is dark but not bright)
        et = torch.minimum((x[:, 1:2] - self.t_et) * 5, wt)
        return torch.cat([tc, wt, et], 1)

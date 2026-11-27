"""Affine / orientation / resampling helpers.

Conventions
-----------
* Arrays are indexed ``[x, y, z]`` (voxel indices); an *affine* maps voxel index -> scanner RAS+ millimetres.
* The *working grid* is the grid on which the model runs and all measurements are made: the input
  axes are permuted/flipped (lossless) so they are closest to RAS+, then resampled to isotropic
  1 mm only if the input is not already 1 mm (the model was trained on 1 mm isotropic data).
* Everything computed on the working grid can be mapped back to the original grid
  (``GridTransform.to_orig``) so exported masks stay aligned with the original scan.
"""
from __future__ import annotations

from dataclasses import dataclass

import nibabel as nib
import numpy as np
from nibabel.orientations import apply_orientation, axcodes2ornt, inv_ornt_aff, io_orientation, ornt_transform
from scipy import ndimage

_RAS = axcodes2ornt(("R", "A", "S"))


def voxel_sizes(affine: np.ndarray) -> np.ndarray:
    """Edge length (mm) of a voxel along each array axis."""
    return np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)


def voxel_volume_mm3(affine: np.ndarray) -> float:
    """Physical volume of one voxel in mm^3 (valid for oblique / anisotropic grids)."""
    return float(abs(np.linalg.det(np.asarray(affine)[:3, :3])))


def axcodes(affine: np.ndarray) -> str:
    return "".join(nib.aff2axcodes(affine))


def obliquity_deg(affine: np.ndarray) -> float:
    """Largest angle (deg) between a voxel axis and the nearest scanner axis (0 for axis-aligned scans)."""
    m = np.asarray(affine)[:3, :3]
    cols = m / np.linalg.norm(m, axis=0)
    return float(np.degrees(np.arccos(np.clip(np.abs(cols).max(axis=0), 0.0, 1.0))).max())


def apply_affine(affine: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    return pts @ np.asarray(affine)[:3, :3].T + np.asarray(affine)[:3, 3]


def resample_to_grid(data: np.ndarray, src_affine, dst_affine, dst_shape, order: int = 1, cval: float = 0.0) -> np.ndarray:
    """Resample ``data`` ([X,Y,Z] or [C,X,Y,Z]) from the ``src`` grid onto the ``dst`` grid (same world space)."""
    m = np.linalg.inv(np.asarray(src_affine, float)) @ np.asarray(dst_affine, float)
    kw = dict(matrix=m[:3, :3], offset=m[:3, 3], output_shape=tuple(int(s) for s in dst_shape),
              order=order, mode="constant", cval=cval, prefilter=order > 1)
    if data.ndim == 3:
        return ndimage.affine_transform(data, **kw)
    return np.stack([ndimage.affine_transform(c, **kw) for c in data])


@dataclass(frozen=True)
class GridTransform:
    """Bidirectional mapping between the original voxel grid and the working grid."""

    orig_affine: np.ndarray
    orig_shape: tuple
    work_affine: np.ndarray
    work_shape: tuple
    scale: tuple  # per-axis (target spacing / current spacing) applied after reorientation; 1.0 = no resampling

    @property
    def resampled(self) -> bool:
        return any(abs(s - 1.0) > 1e-9 for s in self.scale)

    @property
    def _fwd(self):
        return ornt_transform(io_orientation(self.orig_affine), _RAS)

    @property
    def _inv(self):
        return ornt_transform(_RAS, io_orientation(self.orig_affine))

    def _reoriented_affine_shape(self):
        t = self._fwd
        aff = np.asarray(self.orig_affine, float) @ inv_ornt_aff(t, self.orig_shape)
        return aff, _oriented_shape(t, self.orig_shape)

    def to_work(self, data: np.ndarray, order: int = 1) -> np.ndarray:
        """Original grid -> working grid. ``data`` is [X,Y,Z] or [C,X,Y,Z]."""
        lead = data.ndim - 3
        out = apply_orientation(np.moveaxis(data, list(range(lead)), list(range(-lead, 0))) if lead else data, self._fwd)
        if lead:
            out = np.moveaxis(out, list(range(-lead, 0)), list(range(lead)))
        if not self.resampled:
            return np.ascontiguousarray(out)
        aff, _ = self._reoriented_affine_shape()
        return resample_to_grid(out, aff, self.work_affine, self.work_shape, order=order)

    def to_orig(self, data: np.ndarray, order: int = 0) -> np.ndarray:
        """Working grid -> original grid (use ``order=0`` for label masks)."""
        lead = data.ndim - 3
        if self.resampled:
            aff, shape = self._reoriented_affine_shape()
            data = resample_to_grid(data, self.work_affine, aff, shape, order=order)
        moved = np.moveaxis(data, list(range(lead)), list(range(-lead, 0))) if lead else data
        out = apply_orientation(moved, self._inv)
        if lead:
            out = np.moveaxis(out, list(range(-lead, 0)), list(range(lead)))
        return np.ascontiguousarray(out)


def _oriented_shape(t: np.ndarray, shape) -> tuple:
    """Shape after ``apply_orientation``: input axis i becomes output axis ``t[i, 0]`` (NOT the inverse mapping -
    the two differ for 3-cycles of axes)."""
    out = [0, 0, 0]
    for i in range(3):
        out[int(t[i, 0])] = int(shape[i])
    return tuple(out)


def make_working_grid(orig_affine, orig_shape, spacing: float = 1.0, tol: float = 1e-3) -> GridTransform:
    """Build the RAS / ``spacing``-mm isotropic working grid for a scan."""
    orig_affine = np.asarray(orig_affine, float)
    orig_shape = tuple(int(s) for s in orig_shape[:3])
    t = ornt_transform(io_orientation(orig_affine), _RAS)
    aff = orig_affine @ inv_ornt_aff(t, orig_shape)
    shape = _oriented_shape(t, orig_shape)
    sizes = voxel_sizes(aff)
    scale = np.where(np.abs(sizes - spacing) < tol, 1.0, spacing / sizes)
    new_shape = tuple(int(max(1, round(s / sc))) for s, sc in zip(shape, scale))
    work_affine = aff @ np.diag([*scale, 1.0])
    return GridTransform(orig_affine, orig_shape, work_affine, new_shape, tuple(float(s) for s in scale))

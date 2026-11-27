"""Dataset discovery (via the dataset registry) and MONAI transforms for training / validation."""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np

from . import MODALITIES
from .datasets import DatasetSpec, get_spec


def _default_spec() -> DatasetSpec:
    return get_spec("brats2020")


def locate_root(path: str | Path, spec: DatasetSpec | None = None) -> Path:
    """Unwrap the nesting used by downloads (e.g. Kaggle: .../BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/...)."""
    spec = spec or _default_spec()
    probe = (spec.files.get("t1") or ["*_t1.nii*"])[0]
    p = Path(path)
    for _ in range(3):
        subs = [d for d in p.iterdir() if d.is_dir()] if p.is_dir() else []
        if any(glob.glob(str(d / probe)) for d in subs):
            return p
        nested = [d for d in subs if "TrainingData" in d.name or "BraTS" in d.name or "PDGM" in d.name]
        if len(nested) == 1:
            p = nested[0]
            continue
        break
    return p


def find_case(case_dir: Path, spec: DatasetSpec | None = None) -> tuple[dict | None, str]:
    spec = spec or _default_spec()
    files = {}
    for key in (*MODALITIES, "seg"):
        for pat in spec.files.get(key, []):
            m = sorted(glob.glob(str(case_dir / pat)))
            if m:
                files[key] = m[0]
                break
    missing = [m for m in (*MODALITIES, "seg") if m not in files]
    if missing:
        return None, "missing files: " + ", ".join(missing) + f" (found: {sorted(os.listdir(case_dir))[:8]})"
    return files, ""


def discover_cases(root: str | Path, spec: DatasetSpec | None = None, require_usable: bool = False) -> tuple[dict[str, dict], list[tuple[str, str]]]:
    """-> ({case_id: {'id','image':[t1,t1ce,t2,flair],'label','files','dataset','ncr','ed','et'}}, [(case_id, reason skipped)]).

    ``require_usable=True`` (training/evaluation) rejects datasets whose label mapping is undeclared or unverified."""
    spec = spec or _default_spec()
    if require_usable:
        spec.require_usable()
    root = locate_root(root, spec)
    cases, skipped = {}, []
    for d in sorted(p for p in root.glob(spec.case_glob) if p.is_dir()):
        files, why = find_case(d, spec)
        if files is None:
            skipped.append((d.name, why))
            continue
        ncr, ed, et = spec.label_ints if spec.labels else (1, 2, 4)
        cases[d.name] = {"id": d.name, "image": [files[m] for m in MODALITIES], "label": files["seg"], "files": files, "dataset": spec.name,
                         "ncr": ncr, "ed": ed, "et": et}
    return cases, skipped


def discover_many(roots: dict[str, str | Path], registry: dict[str, DatasetSpec]) -> tuple[dict[str, dict], dict[str, list]]:
    """Merge several datasets (training/evaluation: every dataset must be verified). Case ids must be globally unique."""
    merged, skipped = {}, {}
    for name, root in roots.items():
        cases, sk = discover_cases(root, registry[name], require_usable=True)
        clash = set(cases) & set(merged)
        if clash:
            raise ValueError(f"case ids occur in more than one dataset (e.g. {sorted(clash)[:3]}): ids must be globally unique")
        merged.update(cases)
        skipped[name] = sk
    return merged, skipped


def select(cases: dict[str, dict], ids: list[str]) -> list[dict]:
    missing = [i for i in ids if i not in cases]
    if missing:
        raise KeyError(f"{len(missing)} manifest patients not found in the data folder(s), e.g. {missing[:3]}")
    return [cases[i] for i in ids]


def _regions_transform():
    from monai.transforms import MapTransform
    import torch

    class RegionsFromLabelsd(MapTransform):
        """label -> 3 channels [TC, WT, ET] using the CASE's own label integers (d['ncr'], d['ed'], d['et'])."""

        def __call__(self, data):
            d = dict(data)
            ncr, ed, et = int(d.get("ncr", 1)), int(d.get("ed", 2)), int(d.get("et", 4))
            for key in self.key_iterator(d):
                lab = d[key]
                tc = (lab == ncr) | (lab == et)
                d[key] = torch.cat([tc, tc | (lab == ed), lab == et], 0).float()
            return d
    return RegionsFromLabelsd


def build_transforms(train: bool, roi=(128, 128, 128), flip_axes=(0,), affine_prob: float = 0.2, noise_prob: float = 0.1):
    """Training/validation transforms. Deterministic part == notebook (Orientationd RAS, CropForegroundd, NormalizeIntensityd);
    augmentation is deliberately anatomy-preserving: left-right flip only (patients are scanned head-up, so A-P / S-I flips create
    impossible anatomy), mild affine (<=10 deg, 0.9-1.1x), intensity scale/shift and light Gaussian noise."""
    import torch
    from monai.transforms import (CastToTyped, Compose, CropForegroundd, EnsureChannelFirstd, EnsureTyped, LoadImaged, NormalizeIntensityd, Orientationd,
                                  RandAffined, RandFlipd, RandGaussianNoised, RandScaleIntensityd, RandShiftIntensityd, RandSpatialCropd, SpatialPadd)
    Regions = _regions_transform()
    det = [
        LoadImaged(keys=["image", "label"], image_only=True), EnsureChannelFirstd(keys=["image", "label"]),
        Regions(keys="label"), Orientationd(keys=["image", "label"], axcodes="RAS"),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
    if not train:
        return Compose([*det, EnsureTyped(keys=["image", "label"], dtype=torch.float32)])
    cache_cast = [SpatialPadd(keys=["image", "label"], spatial_size=roi), CastToTyped(keys=["image", "label"], dtype=(np.float16, np.uint8))]
    aug = [
        RandSpatialCropd(keys=["image", "label"], roi_size=roi, random_size=False),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        *[RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=a) for a in flip_axes],
        RandAffined(keys=["image", "label"], prob=affine_prob, rotate_range=(0.17, 0.17, 0.17), scale_range=(0.1, 0.1, 0.1),
                    mode=("bilinear", "nearest"), padding_mode="zeros"),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5), RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        RandGaussianNoised(keys="image", prob=noise_prob, std=0.05),
    ]
    return Compose([*det, *cache_cast, *aug])

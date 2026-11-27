"""BraTS-style dataset discovery and MONAI transforms for training / validation."""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np

from . import MODALITIES


def locate_root(path: str | Path) -> Path:
    """Unwrap the nesting used by the Kaggle BraTS 2020 download (…/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/…)."""
    p = Path(path)
    for _ in range(3):
        subs = [d for d in p.iterdir() if d.is_dir()] if p.is_dir() else []
        if any(glob.glob(str(d / "*_t1.nii*")) for d in subs):
            return p
        nested = [d for d in subs if "TrainingData" in d.name or "BraTS" in d.name]
        if len(nested) == 1:
            p = nested[0]
            continue
        break
    return p


def find_case(case_dir: Path) -> tuple[dict | None, str]:
    files = {}
    for mod in (*MODALITIES, "seg"):
        m = sorted(glob.glob(str(case_dir / f"*_{mod}.nii*")))
        if m:
            files[mod] = m[0]
    missing = [m for m in (*MODALITIES, "seg") if m not in files]
    if missing:
        return None, "missing files: " + ", ".join(missing) + f" (found: {sorted(os.listdir(case_dir))[:8]})"
    return files, ""


def discover_cases(root: str | Path, with_seg: bool = True) -> tuple[dict[str, dict], list[tuple[str, str]]]:
    """-> ({case_id: {'image': [t1,t1ce,t2,flair], 'label': seg, 'files': {...}}}, [(case_id, reason skipped)])."""
    root = locate_root(root)
    cases, skipped = {}, []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        files, why = find_case(d)
        if files is None:
            skipped.append((d.name, why))
            continue
        cases[d.name] = {"id": d.name, "image": [files[m] for m in MODALITIES], "label": files["seg"], "files": files}
    return cases, skipped


def select(cases: dict[str, dict], ids: list[str]) -> list[dict]:
    missing = [i for i in ids if i not in cases]
    if missing:
        raise KeyError(f"{len(missing)} manifest patients not found in the data folder, e.g. {missing[:3]}")
    return [cases[i] for i in ids]


def build_transforms(train: bool, roi=(128, 128, 128), flip_axes=(0,), affine_prob: float = 0.2, noise_prob: float = 0.1):
    """Training/validation transforms. Deterministic part == notebook (Orientationd RAS, CropForegroundd, NormalizeIntensityd);
    augmentation is deliberately anatomy-preserving: left-right flip only (patients are scanned head-up, so A-P / S-I flips create
    impossible anatomy), mild affine (<=10 deg, 0.9-1.1x), intensity scale/shift and light Gaussian noise."""
    import torch
    from monai.transforms import (CastToTyped, Compose, ConvertToMultiChannelBasedOnBratsClassesd, CropForegroundd, EnsureChannelFirstd,
                                  EnsureTyped, LoadImaged, NormalizeIntensityd, Orientationd, RandAffined, RandFlipd, RandGaussianNoised,
                                  RandScaleIntensityd, RandShiftIntensityd, RandSpatialCropd, SpatialPadd)
    det = [
        LoadImaged(keys=["image", "label"], image_only=True), EnsureChannelFirstd(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"), Orientationd(keys=["image", "label"], axcodes="RAS"),
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

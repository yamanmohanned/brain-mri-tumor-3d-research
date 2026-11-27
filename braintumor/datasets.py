"""Dataset registry: how to find cases, which integer labels mean what, and whether that mapping has been verified.

Label numbers are NEVER assumed to mean the same thing across datasets: every dataset declares its own mapping, the audit
compares it with the labels actually present in the files, and unverified or undeclared mappings are rejected for training.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

CANONICAL = ("NCR", "ED", "ET")           # necrotic/non-enhancing core, oedema, enhancing tumour


class DatasetError(ValueError):
    pass


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    description: str
    case_glob: str
    files: dict
    labels: dict | None
    role: str                        # "pool" (train/val/test) or "external" (never trained on)
    verified: bool
    source: str = ""
    license: str = ""
    notes: str = ""

    def require_usable(self) -> None:
        if not self.labels or set(self.labels) != set(CANONICAL):
            raise DatasetError(f"{self.name}: no declared label mapping for {CANONICAL} - establish it from the dataset documentation and data first")
        if not self.verified:
            raise DatasetError(f"{self.name}: label mapping/file patterns are not VERIFIED on real data (configs/datasets.yaml: verified=false). "
                               "Run `python -m braintumor audit --dataset ... ` on the downloaded files, then set verified: true.")

    @property
    def label_ints(self) -> tuple[int, int, int]:
        return tuple(int(self.labels[k]) for k in CANONICAL)


def default_registry_path() -> Path:
    """configs/datasets.yaml, or the file named by $BRAINTUMOR_REGISTRY (used by tests and cloud runs with private datasets)."""
    import os
    if os.environ.get("BRAINTUMOR_REGISTRY"):
        return Path(os.environ["BRAINTUMOR_REGISTRY"])
    return Path(__file__).resolve().parents[1] / "configs" / "datasets.yaml"


def load_registry(path: str | Path | None = None) -> dict[str, DatasetSpec]:
    p = Path(path) if path else default_registry_path()
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    reg = {}
    for name, d in raw.items():
        reg[name] = DatasetSpec(name=name, description=d.get("description", ""), case_glob=d.get("case_glob", "*"), files=d.get("files") or {},
                                labels=d.get("labels"), role=d.get("role", "pool"), verified=bool(d.get("verified", False)),
                                source=d.get("source", ""), license=d.get("license", ""), notes=d.get("notes", ""))
    return reg


def get_spec(name: str, path: str | Path | None = None) -> DatasetSpec:
    reg = load_registry(path)
    if name not in reg:
        raise DatasetError(f"unknown dataset {name!r}; registered: {sorted(reg)}")
    return reg[name]


def regions_from_labels(labels: np.ndarray, ncr: int = 1, ed: int = 2, et: int = 4) -> dict[str, np.ndarray]:
    """Dataset labels -> {TC, WT, ET} using the dataset's OWN integer mapping (BraTS 2020/2021: 1,2,4; BraTS 2023 uses ET=3)."""
    tc = (labels == ncr) | (labels == et)
    return {"TC": tc, "WT": tc | (labels == ed), "ET": labels == et}


def check_label_mapping(spec: DatasetSpec, observed: set[int]) -> dict:
    """Programmatic validation of the declared mapping against the label values actually found in the files."""
    if not spec.labels:
        return {"declared": None, "observed": sorted(observed), "consistent": False, "reason": "no label mapping declared"}
    declared = {0, *map(int, spec.labels.values())}
    extra, missing = sorted(observed - declared), sorted(declared - observed - {0})
    ok = not extra
    return {"declared": dict(spec.labels), "observed": sorted(observed), "undeclared_values": extra, "declared_but_never_observed": missing,
            "consistent": ok, "reason": "" if ok else f"files contain label values {extra} that the mapping does not define"}

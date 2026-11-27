"""Patient-level data splits with saved, hash-protected manifests (no slice-level splitting, ever)."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import random
from pathlib import Path

import numpy as np


class SplitError(ValueError):
    pass


def ids_hash(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def legacy_split(ids: list[str], seed: int = 42, frac_train: float = 0.70, frac_val: float = 0.15) -> dict[str, list[str]]:
    """Exact reproduction of the original notebook's split: ``random.Random(seed).shuffle`` of the sorted patient list,
    then 70/15/15 by ``int(frac * n)``. Used so the legacy checkpoints can be evaluated without train/test leakage."""
    shuffled = sorted(ids)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_tr, n_va = int(frac_train * n), int(frac_val * n)
    return {"train": shuffled[:n_tr], "val": shuffled[n_tr:n_tr + n_va], "test": shuffled[n_tr + n_va:]}


def stratified_split(ids: list[str], strata: dict[str, int] | None = None, seed: int = 42,
                     fractions=(0.70, 0.15, 0.15)) -> dict[str, list[str]]:
    """Patient-level split, optionally stratified (e.g. by tumour-size tertile) so each split has a similar mix."""
    if abs(sum(fractions) - 1) > 1e-9:
        raise SplitError("fractions must sum to 1")
    rng = random.Random(seed)
    groups: dict[int, list[str]] = {}
    for i in sorted(ids):
        groups.setdefault(0 if strata is None else int(strata[i]), []).append(i)
    out = {"train": [], "val": [], "test": []}
    for g in sorted(groups):
        members = groups[g][:]
        rng.shuffle(members)
        n = len(members)
        n_tr, n_va = int(round(fractions[0] * n)), int(round(fractions[1] * n))
        out["train"] += members[:n_tr]
        out["val"] += members[n_tr:n_tr + n_va]
        out["test"] += members[n_tr + n_va:]
    return {k: sorted(v) for k, v in out.items()}


def kfold_splits(ids: list[str], k: int = 5, seed: int = 42, strata: dict[str, int] | None = None) -> list[dict[str, list[str]]]:
    """Patient-level k-fold: each fold's val set is disjoint; a fixed fold-local test set is NOT carved out (use for CV estimates)."""
    rng = random.Random(seed)
    folds: list[list[str]] = [[] for _ in range(k)]
    groups: dict[int, list[str]] = {}
    for i in sorted(ids):
        groups.setdefault(0 if strata is None else int(strata[i]), []).append(i)
    j = 0
    for g in sorted(groups):
        m = groups[g][:]
        rng.shuffle(m)
        for pid in m:
            folds[j % k].append(pid)
            j += 1
    return [{"train": sorted(sum((folds[f] for f in range(k) if f != t), [])), "val": sorted(folds[t])} for t in range(k)]


def verify_split(splits: dict[str, list[str]], duplicate_groups: list[list[str]] | None = None) -> dict:
    """Raise ``SplitError`` on any patient appearing in more than one split (or twice in one); also checks duplicate groups."""
    seen: dict[str, str] = {}
    for name, ids in splits.items():
        if len(set(ids)) != len(ids):
            raise SplitError(f"duplicate patient ids inside split {name!r}")
        for i in ids:
            if i in seen:
                raise SplitError(f"LEAKAGE: patient {i} is in both {seen[i]!r} and {name!r}")
            seen[i] = name
    for grp in duplicate_groups or []:
        where = {seen[i] for i in grp if i in seen}
        if len(where) > 1:
            raise SplitError(f"LEAKAGE: duplicate/near-duplicate patients {grp} are spread over splits {sorted(where)}")
    return {"n_unique_patients": len(seen), **{f"n_{k}": len(v) for k, v in splits.items()}}


def save_manifest(path: str | Path, splits: dict[str, list[str]], *, method: str, seed: int | None, dataset: str, notes: str = "") -> dict:
    verify_split(splits)
    all_ids = sum(splits.values(), [])
    man = {
        "schema": 1, "method": method, "seed": seed, "dataset": dataset, "notes": notes,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "counts": {k: len(v) for k, v in splits.items()}, "ids_sha256": ids_hash(all_ids),
        "unit_of_split": "patient (one MRI exam per id; slices are never split)", "splits": splits,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(man, indent=1), encoding="utf-8")
    return man


def load_manifest(path: str | Path) -> dict:
    man = json.loads(Path(path).read_text(encoding="utf-8"))
    verify_split(man["splits"])
    if ids_hash(sum(man["splits"].values(), [])) != man["ids_sha256"]:
        raise SplitError("manifest ids do not match their stored hash (file was edited)")
    man["file_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return man


def brats2020_ids() -> list[str]:
    """The 368 complete BraTS 2020 training patients used by the notebook (folder 355 has no standard *_seg file and was skipped)."""
    return [f"BraTS20_Training_{i:03d}" for i in range(1, 370) if i != 355]

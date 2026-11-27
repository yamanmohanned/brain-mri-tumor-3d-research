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


def save_manifest(path: str | Path, splits: dict[str, list[str]], *, method: str, seed: int | None, dataset: str, notes: str = "", extra: dict | None = None) -> dict:
    verify_split(splits)
    all_ids = sum(splits.values(), [])
    man = {
        "schema": 1, "method": method, "seed": seed, "dataset": dataset, "notes": notes,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "counts": {k: len(v) for k, v in splits.items()}, "ids_sha256": ids_hash(all_ids),
        "unit_of_split": "patient (one MRI exam per id; slices are never split)", "splits": splits,
    }
    if extra:
        man.update(extra)
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


# ----------------------------------------------------------------------------- multi-dataset split

def duplicate_groups_from_pairs(ids, pairs: list[dict]) -> list[list[str]]:
    """Union-find over candidate duplicate pairs -> groups (size > 1). Members of a group are never separated."""
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for p in pairs:
        a, b = p["pair"]
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    groups: dict[str, list[str]] = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    return [sorted(g) for g in groups.values() if len(g) > 1]


def build_combined_split(cases: list[dict], *, seed: int = 42, fractions=(0.70, 0.15, 0.15), legacy: dict[str, str] | None = None,
                         pairs: list[dict] | None = None, external_datasets=(), max_group_size: int = 6,
                         max_grouped_fraction: float = 0.60) -> tuple[dict[str, list[str]], dict]:
    """Patient-level split over several datasets.

    * ``cases``: audit records (id, dataset, size_stratum, problems, gt_*_cm3). Cases with audit problems are rejected, never split.
    * ``legacy``: {id: 'train'|'val'|'test'} assignments to KEEP (the legacy test patients become ``test_legacy``: they were used to compare
      SegResNet and U-Net in the notebook, so they are NOT an untouched benchmark). New pool patients fill ``train``/``val``/``test``.
    * candidate duplicates are grouped (union-find) and each group is assigned to exactly one split; a group that touches both an
      external cohort and the pool is removed from the external cohort.
    * ``external_datasets``: datasets that are never trained on; they form ``external``.
    Returns (splits, audit_info). Raises SplitError if any patient would appear twice."""
    legacy = legacy or {}
    rng = random.Random(seed)
    ext = set(external_datasets)
    rejected, ok = [], {}
    for r in cases:
        if r.get("problems"):
            rejected.append({"id": r["id"], "dataset": r["dataset"], "reason": "; ".join(r["problems"])[:200]})
        else:
            ok[r["id"]] = r
    groups = duplicate_groups_from_pairs(list(ok), pairs or [])
    n_grouped = sum(len(g) for g in groups)
    if groups and (max(len(g) for g in groups) > max_group_size or n_grouped > max_grouped_fraction * max(len(ok), 1)):
        raise SplitError(f"duplicate threshold too permissive: {n_grouped} of {len(ok)} patients fall into {len(groups)} duplicate groups "
                         f"(largest {max(len(g) for g in groups)}). Inspect the similarity distribution in the audit and raise --dup-corr; "
                         "large chained groups would assign big parts of a dataset wholesale to one split. (Genuine pairwise overlap between dataset releases is fine and is kept together.)")
    gof = {m: i for i, g in enumerate(groups) for m in g}
    units: dict[str, list[str]] = {}
    for i in sorted(ok):
        units.setdefault(f"g{gof[i]}" if i in gof else f"s:{i}", []).append(i)

    splits = {"train": [], "val": [], "test": [], "test_legacy": [], "external": []}
    conflicts, dropped_ext = [], []
    order = {"test": 3, "val": 2, "train": 1}
    free: dict[tuple, list[list[str]]] = {}
    for key, members in sorted(units.items()):
        in_ext = [m for m in members if ok[m]["dataset"] in ext]
        in_pool = [m for m in members if ok[m]["dataset"] not in ext]
        if in_ext and in_pool:                                 # same patient present in the pool and in the external cohort
            for m in in_ext:
                dropped_ext.append({"id": m, "dataset": ok[m]["dataset"], "reason": f"duplicate/overlap with pool patient(s) {in_pool[:3]}"})
            members, in_ext = in_pool, []
        if in_ext:
            splits["external"] += members
            continue
        forced = {legacy[m] for m in members if m in legacy}
        if forced:
            top = max(forced, key=order.get)
            if len(forced) > 1:
                conflicts.append({"members": members, "legacy_splits": sorted(forced), "assigned": top})
            splits["test_legacy" if top == "test" else top] += members
        else:
            free.setdefault((ok[members[0]]["dataset"], ok[members[0]].get("size_stratum", 0)), []).append(members)
    for stratum, ulist in sorted(free.items()):
        rng.shuffle(ulist)
        n = len(ulist)
        n_tr, n_va = int(round(fractions[0] * n)), int(round(fractions[1] * n))
        for k, members in enumerate(ulist):
            splits["train" if k < n_tr else "val" if k < n_tr + n_va else "test"] += members
    splits = {k: sorted(v) for k, v in splits.items()}
    verify_split(splits, groups)                               # zero patient overlap, duplicate groups intact

    def dist(ids):
        vols = [ok[i]["gt_WT_cm3"] for i in ids]
        by_ds = {}
        for i in ids:
            by_ds[ok[i]["dataset"]] = by_ds.get(ok[i]["dataset"], 0) + 1
        return {"n": len(ids), "by_dataset": by_ds, "wt_cm3_median": round(float(np.median(vols)), 1) if vols else None,
                "wt_cm3_q25_q75": [round(float(np.percentile(vols, 25)), 1), round(float(np.percentile(vols, 75)), 1)] if vols else None,
                "n_without_ET": sum(1 for i in ids if ok[i].get("gt_ET_cm3", 1) == 0)}
    info = {
        "seed": seed, "fractions": list(fractions), "per_split": {k: dist(v) for k, v in splits.items()},
        "patient_overlap_between_splits": 0, "n_rejected": len(rejected), "rejected": rejected, "duplicate_groups": groups,
        "legacy_group_conflicts": conflicts, "external_removed_for_overlap": dropped_ext, "external_datasets": sorted(ext),
        "notes": ["test_legacy = the notebook's test patients; the notebook compared two models on them, so they are not an untouched benchmark",
                  "an untouched test set exists only if `test` is non-empty (new patients never used for comparison)",
                  "`external` patients are never trained on, selected on, or tuned on"],
    }
    return splits, info

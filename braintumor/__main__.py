"""Command line: python -m braintumor <audit|split|train|evaluate|analyze|gui> ..."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import _preload  # noqa: F401  (torch must be imported before any Qt binding on this platform)


def _audit(a):
    from .audit import audit_dataset, legacy_split_consistency, save_audit
    from .splits import load_manifest
    res = audit_dataset(a.data_root, workers=a.workers, progress=lambda i, n: print(f"\r  audited {i}/{n}", end="", flush=True))
    print()
    save_audit(res, a.out)
    s = res["summary"]
    print(json.dumps({k: s[k] for k in ("n_folders", "n_complete", "n_with_problems", "skipped_incomplete", "shapes", "spacings", "orientations", "label_sets",
                                         "n_without_ET", "multifocal_WT_cases", "exact_duplicate_groups")}, indent=1, default=str))
    print("GT volume (cm3):", json.dumps(s["gt_volume_cm3"]))
    if s["near_duplicate_candidates"]:
        print("Near-duplicate candidates (manual review):", s["near_duplicate_candidates"][:10])
    if a.manifest:
        man = load_manifest(a.manifest)
        chk = legacy_split_consistency(res, man["splits"]["test"])
        print("Legacy split consistency with the notebook's printed facts:", json.dumps(chk))
        (Path(a.out) / "legacy_split_consistency.json").write_text(json.dumps(chk, indent=1))


def _split(a):
    from .audit import audit_dataset
    from .data import discover_cases
    from .splits import brats2020_ids, legacy_split, save_manifest, stratified_split, verify_split
    if a.method == "legacy":
        ids = sorted(discover_cases(a.data_root)[0]) if a.data_root else brats2020_ids()
        sp, notes = legacy_split(ids, a.seed), "reproduces the original notebook split"
    else:
        res = audit_dataset(a.data_root, workers=a.workers)
        ok = [r for r in res["cases"] if "size_stratum" in r and not r["problems"]]
        strata = {r["id"]: r["size_stratum"] for r in ok}
        sp, notes = stratified_split(list(strata), strata, a.seed), "stratified by whole-tumour size tertile; cases with audit problems excluded"
        verify_split(sp, res["summary"]["exact_duplicate_groups"])
    man = save_manifest(a.out, sp, method=a.method, seed=a.seed, dataset="BraTS2020 training", notes=notes)
    print(json.dumps(man["counts"]), "->", a.out)


def _analyze(a):
    from .export import export_all
    from .inference import InferenceConfig
    from .io import load_label_map, load_modalities
    from .model import load_checkpoint
    from .pipeline import AnalysisConfig, analyze
    vol = load_modalities({"t1": a.t1, "t1ce": a.t1ce, "t2": a.t2, "flair": a.flair})
    cfg = AnalysisConfig(infer=InferenceConfig(tta="flips4" if a.tta else "none", device=a.device))
    cid = a.case_id or Path(a.t1).name.split("_t1")[0].split(".")[0]
    if a.checkpoint:
        model, info = load_checkpoint(a.checkpoint)
        res = analyze(vol, model, info, cfg, progress=lambda f, m: print(f"\r  inference {f:4.0%} {m}", end="", flush=True), case_id=cid)
        print()
    elif a.seg:
        lab, _ = load_label_map(a.seg, like=vol)
        res = analyze(vol, labels=lab, label_source=str(a.seg), cfg=cfg, case_id=cid)
    else:
        sys.exit("Provide --checkpoint (model prediction) or --seg (measure an existing annotation).")
    out = export_all(res, a.out, mesh_format=a.mesh_format, overwrite=a.overwrite)
    wt = res.measurements["WT"]
    print(json.dumps({"source": res.source, "WT_volume_cm3": round(wt.get("volume_cm3", 0), 1), "max_diameter_mm": round(wt.get("max_diameter_mm", 0)),
                      "timings_s": res.timings, "exported": out}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="braintumor")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("audit", help="dataset integrity audit")
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", default="runs/audit")
    p.add_argument("--manifest", help="legacy manifest to cross-check against the notebook's printed facts")
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(fn=_audit)
    p = sub.add_parser("split", help="write a patient-level split manifest")
    p.add_argument("--data-root")
    p.add_argument("--method", choices=["legacy", "stratified"], default="legacy")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="splits/new.json")
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(fn=_split)
    p = sub.add_parser("analyze", help="headless analysis of one case")
    for m in ("t1", "t1ce", "t2", "flair"):
        p.add_argument(f"--{m}", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--seg", help="BraTS-style label map to measure instead of running a model")
    p.add_argument("--out", required=True)
    p.add_argument("--case-id")
    p.add_argument("--tta", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--mesh-format", default="stl", choices=["stl", "obj", "ply"])
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(fn=_analyze)
    sub.add_parser("train", help="see python -m braintumor.train -h")
    sub.add_parser("evaluate", help="see python -m braintumor.evaluate -h")
    sub.add_parser("gui", help="launch the desktop application")
    args, rest = ap.parse_known_args(argv)
    if args.cmd == "train":
        from .train import main as m
        return m(rest)
    if args.cmd == "evaluate":
        from .evaluate import main as m
        return m(rest)
    if args.cmd == "gui":
        from .gui.app import main as m
        return m(rest)
    if rest:
        ap.error(f"unrecognised arguments: {rest}")
    args.fn(args)


if __name__ == "__main__":
    main()

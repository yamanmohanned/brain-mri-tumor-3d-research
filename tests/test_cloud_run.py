"""Cloud stage runner: no silent CPU fallback, dataset verification, resume-aware training stage (phantom data, CPU smoke only)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from braintumor.splits import save_manifest, stratified_split
from tests.phantom import make_brats_like_dataset

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "cloud_run.py"


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=300)


def test_check_gpu_require_exits_nonzero_without_a_usable_gpu():
    import torch
    r = run("check-gpu", "--require")
    info = json.loads(r.stdout)
    if torch.cuda.is_available():
        pytest.skip("a CUDA GPU is present on this machine")
    assert r.returncode == 3 and info["cuda"] is False and "training is refused" in r.stderr


def test_train_stage_refuses_cpu_instead_of_falling_back(tmp_path):
    ids = make_brats_like_dataset(tmp_path / "d", n=6, shape=(36, 40, 32), seed=4)
    man = tmp_path / "m.json"
    save_manifest(man, stratified_split(ids, seed=1), method="t", seed=1, dataset="phantom")
    r = run("train", "--runs-dir", str(tmp_path / "runs"), "--run-name", "x", "--data-root", str(tmp_path / "d"), "--manifest", str(man),
            "--set", "roi=[32,32,32]")
    import torch
    if torch.cuda.is_available():
        pytest.skip("a CUDA GPU is present")
    assert r.returncode == 3 and not (tmp_path / "runs" / "x" / "last.pth").exists()


def test_train_stage_resumes_when_a_checkpoint_exists(tmp_path):
    ids = make_brats_like_dataset(tmp_path / "d", n=10, shape=(40, 44, 36), seed=5)
    man = tmp_path / "m.json"
    save_manifest(man, stratified_split(ids, seed=1, fractions=(0.6, 0.2, 0.2)), method="t", seed=1, dataset="phantom")
    base = ["train", "--runs-dir", str(tmp_path / "runs"), "--run-name", "r", "--data-root", str(tmp_path / "d"), "--manifest", str(man), "--allow-cpu",
            "--set", "roi=[32,32,32]", "--set", "epochs=2", "--set", "val_interval=1", "--set", "num_workers=0", "--set", "cache=false", "--set", "device=cpu",
            "--set", "warmup_epochs=1"]
    first = run(*base, "--max-hours", "1e-9")
    assert first.returncode == 0 and "STARTING" in first.stdout and "Paused for the time budget" in first.stdout
    second = run(*base)
    assert second.returncode == 0 and "RESUMING" in second.stdout and '"finished": true' in second.stdout
    lines = (tmp_path / "runs" / "r" / "metrics.jsonl").read_text().splitlines()
    assert [json.loads(x)["epoch"] for x in lines] == [1, 2]


def test_brats2020_layout_verification_reports_mismatch_honestly(tmp_path):
    sys.path.insert(0, str(SCRIPT.parent))
    import cloud_run
    make_brats_like_dataset(tmp_path / "d", n=5, shape=(36, 40, 32), seed=6)
    info = cloud_run.verify_brats2020(tmp_path / "d")
    assert info["n_complete"] == 5 and info["n_folders"] == 5 and info["matches_expected"] is False        # real BraTS 2020 has 369/368
    assert len(info["folder_listing_sha256"]) == 64

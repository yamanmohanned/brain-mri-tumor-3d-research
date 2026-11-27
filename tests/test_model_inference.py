"""Model construction, checkpoint compatibility and inference plumbing (random weights: NO accuracy claims here)."""
import numpy as np
import pytest
import torch

from braintumor.inference import Cancelled, InferenceConfig, n_windows, predict_probabilities, threshold_masks
from braintumor.model import ARCHS, CheckpointError, build_model, load_checkpoint, save_checkpoint, update_card, read_card

CFG = InferenceConfig(roi=(32, 32, 32), overlap=0.25, device="cpu")


@pytest.fixture(scope="module")
def small_net():
    torch.manual_seed(0)
    return build_model("segresnet").eval()


def test_architectures_have_notebook_signatures():
    sr = build_model("segresnet")
    assert sum(p.numel() for p in sr.parameters()) == 4_702_227       # parameter counts measured on the notebook configuration
    assert sum(p.numel() for p in build_model("unet").parameters()) == 4_810_074
    assert ARCHS["segresnet"]["in_channels"] == 4 and ARCHS["segresnet"]["out_channels"] == 3


def test_output_has_input_spatial_shape_and_valid_probabilities(small_net):
    x = np.random.default_rng(0).normal(size=(4, 40, 44, 36)).astype(np.float32)   # not a multiple of the ROI on purpose
    pred = predict_probabilities(small_net, x, CFG)
    assert pred.probs.shape == (3, 40, 44, 36)
    assert pred.probs.min() >= 0 and pred.probs.max() <= 1
    masks = threshold_masks(pred.probs)
    assert set(masks) == {"TC", "WT", "ET"} and masks["WT"].shape == (40, 44, 36)


def test_volume_smaller_than_roi_is_padded_and_cropped_back(small_net):
    x = np.random.default_rng(1).normal(size=(4, 24, 28, 20)).astype(np.float32)
    assert predict_probabilities(small_net, x, CFG).probs.shape == (3, 24, 28, 20)


def test_progress_reaches_one_and_cancel_stops(small_net):
    x = np.random.default_rng(2).normal(size=(4, 48, 48, 40)).astype(np.float32)
    seen = []
    predict_probabilities(small_net, x, CFG, progress=lambda f, m: seen.append(f))
    assert seen == sorted(seen) and seen[-1] == pytest.approx(1.0)
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 1
    with pytest.raises(Cancelled):
        predict_probabilities(small_net, x, CFG, cancel=cancel)


def test_tta_returns_std_and_is_flip_consistent(small_net):
    x = np.random.default_rng(3).normal(size=(4, 32, 32, 32)).astype(np.float32)
    cfg = InferenceConfig(roi=(32, 32, 32), device="cpu", tta="flips4")
    pred = predict_probabilities(small_net, x, cfg)
    assert pred.passes == 4 and pred.tta_std is not None and pred.tta_std.shape == pred.probs.shape
    assert pred.tta_std.max() > 0
    plain = predict_probabilities(small_net, x, CFG).probs
    assert not np.allclose(plain, pred.probs)


def test_window_count_estimate():
    assert n_windows((128, 128, 128), (128, 128, 128), 0.5) == 1
    assert n_windows((160, 192, 140), (128, 128, 128), 0.5) == 2 * 2 * 2


# ----------------------------------------------------------------------------- checkpoints

def test_v1_checkpoint_roundtrip_is_bit_identical(tmp_path, small_net):
    p = save_checkpoint(tmp_path / "m.pth", small_net, "segresnet", {"note": "test"})
    model, info = load_checkpoint(p)
    assert info["arch"] == "segresnet" and info["format"] == "braintumor_v1" and len(info["sha256"]) == 64
    x = torch.randn(1, 4, 32, 32, 32)
    with torch.no_grad():
        assert torch.equal(model(x), small_net(x))


def test_legacy_notebook_checkpoints_load_and_arch_is_detected(tmp_path):
    for arch in ("segresnet", "unet"):
        net = build_model(arch)
        torch.save(net.state_dict(), tmp_path / f"{arch}_best.pth")              # notebook *_best.pth layout
        torch.save({"epoch": 30, "model": net.state_dict(), "optimizer": {}, "scheduler": {}, "best_dice": 0.76,
                    "history": {"epoch": [30]}}, tmp_path / f"{arch}_last.pth")   # notebook *_last.pth layout
        for f in (f"{arch}_best.pth", f"{arch}_last.pth"):
            _, info = load_checkpoint(tmp_path / f)
            assert info["arch"] == arch and info["format"].startswith("legacy")


def test_incompatible_or_corrupt_checkpoints_give_clear_errors(tmp_path, small_net):
    with pytest.raises(CheckpointError, match="not found"):
        load_checkpoint(tmp_path / "nope.pth")
    (tmp_path / "junk.pth").write_bytes(b"not a checkpoint")
    with pytest.raises(CheckpointError, match="Cannot read"):
        load_checkpoint(tmp_path / "junk.pth")
    torch.save(small_net.state_dict(), tmp_path / "sr.pth")
    with pytest.raises(CheckpointError, match="do not match"):
        load_checkpoint(tmp_path / "sr.pth", arch="unet")
    torch.save({"w": torch.zeros(3)}, tmp_path / "odd.pth")
    with pytest.raises(CheckpointError):
        load_checkpoint(tmp_path / "odd.pth")


def test_checkpoint_card_roundtrip(tmp_path, small_net):
    p = save_checkpoint(tmp_path / "m.pth", small_net, "segresnet")
    assert read_card(p) == {}
    update_card(p, evidence_level="Not validated")
    update_card(p, n_test=5)
    assert read_card(p) == {"evidence_level": "Not validated", "n_test": 5}

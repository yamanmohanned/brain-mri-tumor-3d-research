"""Spatial-metadata preservation, loading/validation and preprocessing (synthetic fixtures only)."""
import nibabel as nib
import numpy as np
import pytest

from braintumor import MODALITIES
from braintumor.geometry import apply_affine, axcodes, make_working_grid, voxel_sizes, voxel_volume_mm3
from braintumor.io import (AmbiguousSeriesError, InputError, assess_input, guess_modality, load_dicom_series, load_label_map,
                           load_modalities)
from braintumor.preprocess import embed, foreground_slices, normalize_nonzero, prepare
from tests.phantom import make_phantom, save_phantom, write_dicom_series

KINDS = ["ras", "las", "lps", "permuted", "oblique"]


@pytest.mark.parametrize("kind", KINDS)
def test_working_grid_is_ras_and_keeps_world_position(kind):
    ph = make_phantom(affine_kind=kind)
    g = make_working_grid(ph["affine"], ph["seg"].shape)
    assert axcodes(g.work_affine) == "RAS"
    seg_w = g.to_work(ph["seg"].astype(np.float32), order=0)
    assert seg_w.shape == g.work_shape                       # regression: 3-cycle axis permutations ('permuted') once gave a wrong shape
    assert g.to_work(ph["data"]).shape == (4, *g.work_shape)
    # world centroid of the whole tumour must be identical on both grids (spatial alignment preserved)
    def world_centroid(mask, aff):
        return apply_affine(aff, np.argwhere(mask).mean(0))
    c0 = world_centroid(ph["seg"] > 0, ph["affine"])
    c1 = world_centroid(seg_w > 0, g.work_affine)
    assert np.allclose(c0, c1, atol=1e-6)
    assert (seg_w > 0).sum() == (ph["seg"] > 0).sum()  # lossless reorientation, no resampling


@pytest.mark.parametrize("kind", KINDS)
def test_roundtrip_to_orig_is_exact_without_resampling(kind):
    ph = make_phantom(affine_kind=kind)
    g = make_working_grid(ph["affine"], ph["seg"].shape)
    back = g.to_orig(g.to_work(ph["seg"], order=0), order=0)
    assert back.shape == ph["seg"].shape
    assert np.array_equal(back, ph["seg"])
    img = g.to_orig(g.to_work(ph["data"], order=1), order=1)
    assert np.allclose(img, ph["data"])


def test_anisotropic_input_is_resampled_to_1mm_and_volume_preserved():
    ph = make_phantom(shape=(80, 90, 30), spacing=(1.0, 1.0, 3.0), wt_radii=(16, 12, 12), center_frac=(0.5, 0.5, 0.5))
    g = make_working_grid(ph["affine"], ph["seg"].shape)
    assert g.resampled
    assert np.allclose(voxel_sizes(g.work_affine), 1.0, atol=1e-6)
    seg_w = g.to_work((ph["seg"] > 0).astype(np.uint8), order=0)
    vol_orig = (ph["seg"] > 0).sum() * 3.0
    vol_work = seg_w.sum() * voxel_volume_mm3(g.work_affine)
    assert abs(vol_work - vol_orig) / vol_orig < 0.03
    back = g.to_orig(seg_w, order=0)
    assert back.shape == ph["seg"].shape
    inter = ((back > 0) & (ph["seg"] > 0)).sum()
    assert 2 * inter / (((back > 0).sum()) + (ph["seg"] > 0).sum()) > 0.97


def test_load_modalities_and_label_map(tmp_path):
    ph = make_phantom(affine_kind="las")
    f = save_phantom(ph, tmp_path, "case")
    vol = load_modalities({m: f[m] for m in MODALITIES})
    assert vol.data.shape == (4, *ph["seg"].shape)
    assert np.allclose(vol.affine, ph["affine"])
    lab, aff = load_label_map(f["seg"], like=vol)
    assert np.array_equal(lab, ph["seg"])
    assert not [i for i in assess_input(vol) if i.severity == "error"]


def test_missing_modality_is_reported(tmp_path):
    f = save_phantom(make_phantom(), tmp_path, "case")
    with pytest.raises(InputError, match="FLAIR"):
        load_modalities({m: f[m] for m in ("t1", "t1ce", "t2")})


def test_mismatched_grids_are_rejected_not_resampled(tmp_path):
    f = save_phantom(make_phantom(), tmp_path, "case")
    other = make_phantom(shape=(72, 80, 60))
    nib.save(nib.Nifti1Image(other["data"][3], other["affine"]), str(f["flair"]))
    with pytest.raises(InputError, match="co-registered"):
        load_modalities({m: f[m] for m in MODALITIES})
    f = save_phantom(make_phantom(), tmp_path / "b", "case")
    ph = make_phantom()
    aff = ph["affine"].copy()
    aff[0, 3] += 5.0
    nib.save(nib.Nifti1Image(ph["data"][2], aff), str(f["t2"]))
    with pytest.raises(InputError, match="spatial transform"):
        load_modalities({m: f[m] for m in MODALITIES})


def test_nan_and_4d_inputs_rejected(tmp_path):
    f = save_phantom(make_phantom(), tmp_path, "case")
    bad = make_phantom()["data"][0]
    bad[3, 3, 3] = np.nan
    nib.save(nib.Nifti1Image(bad, np.eye(4)), str(tmp_path / "nan.nii.gz"))
    with pytest.raises(InputError, match="NaN"):
        load_modalities({**{m: f[m] for m in MODALITIES}, "t1": tmp_path / "nan.nii.gz"})
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8, 3), np.float32), np.eye(4)), str(tmp_path / "4d.nii.gz"))
    with pytest.raises(InputError, match="3D volume"):
        load_modalities({**{m: f[m] for m in MODALITIES}, "t1": tmp_path / "4d.nii.gz"})


def test_qc_flags_non_skull_stripped_and_oblique():
    ph = make_phantom(affine_kind="oblique")
    from braintumor.io import MRVolume
    vol = MRVolume(ph["data"], ph["affine"], MODALITIES)
    msgs = " ".join(i.message for i in assess_input(vol))
    assert "oblique" in msgs
    noisy = ph["data"] + 5.0  # no exact-zero background -> looks like a head with noise, not skull-stripped
    msgs = " ".join(i.message for i in assess_input(MRVolume(noisy, ph["affine"], MODALITIES)))
    assert "skull-stripped" in msgs


@pytest.mark.parametrize("name,expected", [("BraTS20_001_t1ce.nii.gz", "t1ce"), ("x_t1.nii", "t1"), ("x_flair.nii.gz", "flair"),
                                           ("x_t2.nii.gz", "t2"), ("sub-1_t1c.nii.gz", "t1ce"), ("random.nii.gz", None)])
def test_guess_modality(name, expected):
    assert guess_modality(name) == expected


# ----------------------------------------------------------------------------- DICOM

def _blob(shape=(40, 36, 20)):
    rng = np.random.default_rng(1)
    v = rng.integers(10, 500, shape).astype(np.int16)
    return v


@pytest.mark.parametrize("iop", [(1, 0, 0, 0, 1, 0), (0, 1, 0, 0, 0, -1)])
def test_dicom_series_geometry_matches_dicom_standard(tmp_path, iop):
    vol = _blob()
    origin = (-60.0, -70.0, -30.0)
    write_dicom_series(tmp_path / "s", vol, spacing=(0.9, 1.1, 2.0), iop=iop, origin=origin)
    data, aff, info = load_dicom_series(tmp_path / "s")
    assert np.array_equal(data, vol.astype(np.float32))
    # voxel (c, r, k) position in LPS = origin + c*colspacing*rowdir + r*rowspacing*coldir + k*step*normal ; RAS flips x,y
    iop = np.asarray(iop, float)
    normal = np.cross(iop[:3], iop[3:])
    c, r, k = 7, 5, 3
    lps = np.asarray(origin) + c * 0.9 * iop[:3] + r * 1.1 * iop[3:] + k * 2.0 * normal
    expect_ras = lps * [-1, -1, 1]
    got = apply_affine(aff, [[c, r, k]])[0]
    assert np.allclose(got, expect_ras, atol=1e-6)
    assert info["modality"] == "MR"


def test_dicom_rejects_non_mr_missing_slice_and_ambiguity(tmp_path):
    vol = _blob()
    write_dicom_series(tmp_path / "ct", vol, modality="CT")
    with pytest.raises(InputError, match="not MR"):
        load_dicom_series(tmp_path / "ct")
    write_dicom_series(tmp_path / "gap", vol, drop_slice=9)
    with pytest.raises(InputError, match="slice spacing|missing"):
        load_dicom_series(tmp_path / "gap")
    mixed = tmp_path / "mixed"
    write_dicom_series(mixed / "a", vol, desc="A")
    write_dicom_series(mixed / "b", vol, desc="B")
    with pytest.raises(AmbiguousSeriesError):
        load_dicom_series(mixed)
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "notes.txt").write_text("not dicom")
    with pytest.raises(InputError, match="No DICOM"):
        load_dicom_series(tmp_path / "empty")


# ----------------------------------------------------------------------------- preprocessing

def test_normalize_nonzero_keeps_background_zero():
    x = np.zeros((2, 6, 6, 6), np.float32)
    x[:, 2:4, 2:4, 2:4] = np.random.default_rng(0).uniform(10, 20, (2, 2, 2, 2))
    y = normalize_nonzero(x)
    assert (y[x == 0] == 0).all()
    for c in range(2):
        v = y[c][x[c] != 0]
        assert abs(v.mean()) < 1e-5 and abs(v.std() - 1) < 1e-4


def test_prepare_crop_and_embed_roundtrip():
    ph = make_phantom(affine_kind="las")
    from braintumor.io import MRVolume
    vol = MRVolume(ph["data"], ph["affine"], MODALITIES)
    prep = prepare(vol)
    assert prep.tensor.shape[0] == 4 and prep.tensor.shape[1:] == tuple(s.stop - s.start for s in prep.crop)
    assert prep.tensor.shape[1:] != prep.work_shape  # cropped to the brain
    back = embed(prep.tensor, prep, fill=0.0)
    assert back.shape == (4, *prep.work_shape)
    assert np.array_equal(foreground_slices(prep.raw_work), prep.crop)


def test_preprocessing_matches_monai_training_transforms(tmp_path):
    """Inference preprocessing must equal the training pipeline of the notebook (Orientationd->CropForegroundd->Normalize)."""
    pytest.importorskip("monai")
    from monai.transforms import Compose, CropForegroundd, EnsureChannelFirstd, LoadImaged, NormalizeIntensityd, Orientationd
    from braintumor.io import MRVolume
    ph = make_phantom(affine_kind="lps")
    f = save_phantom(ph, tmp_path, "case")
    tf = Compose([
        LoadImaged(keys=["image"], image_only=True), EnsureChannelFirstd(keys=["image"]), Orientationd(keys=["image"], axcodes="RAS"),
        CropForegroundd(keys=["image"], source_key="image", allow_smaller=True),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])
    ref = np.asarray(tf({"image": [str(f[m]) for m in MODALITIES]})["image"])
    vol = load_modalities({m: f[m] for m in MODALITIES})
    mine = prepare(vol).tensor
    assert ref.shape == mine.shape
    assert np.allclose(ref, mine, atol=1e-4)


def test_four_modality_dicom_volume_and_geometry_mismatch_rejection(tmp_path):
    from braintumor.io import load_modalities_dicom
    rng = np.random.default_rng(2)
    series = {}
    for m in MODALITIES:
        write_dicom_series(tmp_path / m, rng.integers(10, 500, (40, 36, 20)).astype(np.int16), spacing=(1.0, 1.0, 2.0), desc=m)
        series[m] = (tmp_path / m, None)
    vol = load_modalities_dicom(series)
    assert vol.data.shape == (4, 40, 36, 20) and vol.modalities == tuple(MODALITIES)
    assert np.allclose(vol.spacing, [1.0, 1.0, 2.0]) and axcodes(vol.affine) in ("LPS", "RPS", "LAS", "RAS", "LPI", "RPI", "LAI", "RAI")
    # FLAIR acquired on a shifted grid -> must be rejected, not resampled
    write_dicom_series(tmp_path / "shifted", rng.integers(10, 500, (40, 36, 20)).astype(np.int16), spacing=(1.0, 1.0, 2.0), origin=(-50.0, -70.0, -30.0))
    bad = dict(series, flair=(tmp_path / "shifted", None))
    with pytest.raises(InputError, match="co-registered"):
        load_modalities_dicom(bad)
    with pytest.raises(InputError, match="Missing required modalities"):
        load_modalities_dicom({k: v for k, v in series.items() if k != "t2"})

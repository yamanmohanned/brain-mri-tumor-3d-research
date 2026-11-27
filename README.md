# Brain MRI tumour segmentation, measurement and 3D reconstruction (research prototype)

> **Experimental research software - not a medical device.** Outputs are model predictions (or supplied annotations), not
> confirmed anatomy, and must not be used for diagnosis, grading or treatment decisions.
> Scientific status, measured results and open problems are in [REPORT.md](REPORT.md).

Pipeline: 4-modality MRI (T1, T1ce, T2, FLAIR; NIfTI or MR DICOM) -> validation/QC -> RAS 1 mm working grid -> 3D SegResNet / U-Net
(MONAI, BraTS regions WT/TC/ET) -> conservative post-processing -> marching-cubes surfaces in scanner mm -> physical measurements ->
interactive GUI, NIfTI/STL/OBJ/PLY/CSV/HTML export.

**Current state in this folder: there is no dataset and no trained checkpoint here** (they lived in the original Colab/Drive session). The software is
complete and tested on synthetic phantoms, but model-based analysis and any accuracy claim need the real BraTS data and weights - see
"Getting data and weights".

## Install (Windows, Python 3.12)

```powershell
python -m venv .venv --system-site-packages     # or a clean venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Use

```powershell
.\.venv\Scripts\python.exe -m braintumor gui                       # desktop application
.\.venv\Scripts\python.exe -m braintumor analyze --t1 a_t1.nii.gz --t1ce a_t1ce.nii.gz --t2 a_t2.nii.gz --flair a_flair.nii.gz `
    --checkpoint weights\best.pth --out exports\case1               # headless; add --seg gt.nii.gz instead of --checkpoint to measure an annotation
.\.venv\Scripts\python.exe -m pytest                               # all tests (GUI tests need a desktop session)
```

GUI: **Dashboard** (model, evidence level, hardware, history) -> **MRI Upload** (NIfTI or DICOM, validation) -> **Analysis** (axial/coronal/sagittal,
overlay, run) -> **3D View** -> **Results** (10-section report) -> **Export**. A checkpoint placed alone in `weights/` loads automatically; with several,
choose one on the Dashboard. Without a model you can still measure/reconstruct a supplied BraTS-style label map ("annotation" mode, labelled as such).

## Getting data and weights

* **Data:** BraTS 2020 training data (Kaggle `awsaf49/brats20-dataset-training-validation`, as used by the notebook) laid out as `<root>/BraTS20_Training_001/BraTS20_Training_001_{t1,t1ce,t2,flair,seg}.nii(.gz)`. Check its terms of use.
* **Weights:** the notebook saved `segresnet_best.pth` / `unet_best.pth` to Google Drive (`brain-tumor-3d-print/weights`). Copy them to `weights/`; they load
  unchanged (same MONAI architectures) but have no evaluation record until you run the evaluation below.
* **Training needs a GPU** (this machine has none usable: Kepler Quadro K3100M, CPU-only PyTorch; ~17 s per 128^3 step => ~70 min per epoch). Use Colab:

```python
# Colab cell (after uploading this folder to Drive and mounting it)
!pip install -q monai nibabel trimesh pydicom pyyaml
!python -m braintumor audit    --data-root /kaggle/input/brats20-dataset-training-validation --out runs/audit --manifest splits/legacy_seed42.json
!python -m braintumor.train    --data-root /kaggle/input/brats20-dataset-training-validation --manifest splits/legacy_seed42.json --out runs/segresnet_v2 --benchmark
!python -m braintumor.train    --config configs/train.yaml --data-root /kaggle/input/... --out runs/segresnet_v2        # add --resume after a disconnect
!python -m braintumor.evaluate --checkpoint runs/segresnet_v2/best.pth --data-root /kaggle/input/... --manifest splits/legacy_seed42.json --split test --out runs/eval_v2
```

`audit` verifies image/mask correspondence, geometry, labels, corruption and duplicates, and cross-checks `splits/legacy_seed42.json` (an exact reproduction of the
notebook's split) against facts the notebook printed. `evaluate` writes per-patient metrics, patient-level bootstrap CIs, size strata, lesion-wise detection, calibration,
pre-specified acceptance checks and an evidence level into `<checkpoint>.card.json`, which the GUI shows. Evaluating a legacy checkpoint: add `--assert-trained-on-manifest`
only if you know it was trained/selected on that manifest's train/val patients only (recorded as an attestation, not verified).

## Layout

`braintumor/` - `geometry` (RAS/1 mm working grid, exact round trip to the original grid), `io` (NIfTI/DICOM + QC), `preprocess`, `model` (+ checkpoint cards), `inference`,
`postprocess`, `mesh`, `measure`, `metrics`, `splits`, `audit`, `data`, `train`, `evaluate`, `pipeline`, `report`, `export`, `viz`, `gui/`.
`tests/` - pytest suite (synthetic phantoms with analytic ground truth). `splits/` - manifests. `configs/train.yaml`. `brain_tumor_project.ipynb` - the original notebook (untouched).

## Known environment issues

* **Import torch before any Qt binding** (done by `python -m braintumor` and the test config): on this machine importing PyQt6 first breaks `c10.dll` (`WinError 1114`).
* Only 5 GB free on `C:`; the venv and pip caches live on `E:`.

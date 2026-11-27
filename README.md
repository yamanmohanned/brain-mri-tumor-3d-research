# Brain MRI tumour segmentation, measurement and 3D reconstruction (research prototype)

> **Experimental research software - not a medical device.** Outputs are model predictions (or supplied annotations), not confirmed anatomy, and must not be used for
> diagnosis, grading or treatment decisions. Scientific status, measured results and open problems: [REPORT.md](REPORT.md). How to use everything: [USER_GUIDE.md](USER_GUIDE.md).

Pipeline: 4-modality MRI (T1, T1ce, T2, FLAIR; NIfTI or MR-DICOM) -> validation/QC -> RAS 1 mm working grid -> 3D SegResNet / U-Net (MONAI; BraTS regions WT/TC/ET) -> conservative
post-processing -> marching-cubes surfaces in scanner mm -> physical measurements -> PyQt6 + VTK GUI and NIfTI/STL/OBJ/PLY/CSV/HTML export. Evaluation: patient-level splits, BraTS-convention metrics,
bootstrap CIs, measurement errors, paired model comparison, pre-specified acceptance criteria and a conservative evidence label.

**State of this repository:** code, tests (138 passing, synthetic phantoms only), GUI and cloud runner are complete. **No dataset or trained checkpoint is stored here** (they live in cloud storage), and
**no model accuracy has been measured by this code yet** - see REPORT.md. **Training never runs on a local CPU**: `braintumor.train` refuses; use the cloud workflow.

## Install and run (Windows, Python 3.12)
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m braintumor gui          # desktop application
.\.venv\Scripts\python.exe -m pytest                  # tests (GUI tests need a desktop session)
```
Place a trained `.pth` in `weights\` (git-ignored). Without a model the GUI can still measure and reconstruct a supplied BraTS-style label map ("annotation" mode).

## Cloud GPU workflow (all heavy work)
Open `notebooks/cloud_gpu_pipeline.ipynb` in Colab (GPU runtime; Colab Secret `GITHUB_TOKEN`, read-only, this repo only). It runs: GPU check -> BraTS 2020 download into the VM -> dataset audit (+ legacy split check) ->
evaluation of the original checkpoint -> resumable training (persistent Drive run folder, wall-clock budget) -> baseline-vs-new comparison on the validation split -> 3D inference test.
The same stages are available as `python scripts/cloud_run.py <stage>` (Kaggle or any GPU VM). Details and provider comparison: USER_GUIDE.md section 18.

## Commands
`python -m braintumor` `audit` | `split` | `split-combined` | `analyze` | `train` | `evaluate` | `compare` | `gui`. Datasets and their label conventions are declared in `configs/datasets.yaml`
(unverified mappings are refused); split manifests live in `splits/` (`legacy_seed42.json` is historical and never modified).

## Layout
`braintumor/` (`geometry`, `io`, `preprocess`, `model`, `inference`, `postprocess`, `mesh`, `measure`, `metrics`, `splits`, `datasets`, `audit`, `data`, `train`, `evaluate`, `compare`, `pipeline`, `report`, `export`, `viz`, `gui/`) -
`tests/` - `scripts/cloud_run.py` - `notebooks/` (`cloud_gpu_pipeline.ipynb`, `legacy/` original Colab notebook, unmodified) - `configs/` - `splits/`.

## Environment notes
* Import torch before any Qt binding on Windows (`WinError 1114` otherwise); `python -m braintumor` and the test config do this.
* Never commit data, checkpoints or tokens (`.gitignore` covers `*.pth`, `*.nii*`, `data/`, `runs/`, `.env`); use `.env.example` as a template.

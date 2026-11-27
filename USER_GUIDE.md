# User guide

> **Research software, not a medical device.** Outputs are model predictions (or annotations you supply), not confirmed anatomy. They must not be used for diagnosis, grading or treatment decisions and do not replace assessment by a qualified radiologist. Scientific status and measured results: [REPORT.md](REPORT.md).

## 1. System requirements
Windows 10/11 (tested), Python 3.12, 8 GB RAM minimum (16 GB recommended), a GPU with OpenGL 3.2 for the 3D view (software rendering also works, slowly). **Inference runs on CPU** (about 1-2 minutes per scan on a 4-core laptop; seconds on a GPU). **Training needs a CUDA GPU** and is refused on CPU - use the cloud workflow in section 18.

## 2. Installation
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
On Windows, `torch` must be imported before PyQt6 (otherwise `WinError 1114`). `python -m braintumor ...` and the test setup do this for you.

## 3. Launching
```powershell
.\.venv\Scripts\python.exe -m braintumor gui
```
On first launch a short walkthrough appears (re-open it from the **Guide** page). The Dashboard shows a first-use checklist that ticks off as you go.

## 4. Required model checkpoints
Put a trained `.pth` into `weights\`. If exactly one is present it loads automatically; otherwise choose it on the Dashboard ("Load selected" / "Browse for checkpoint"). Supported: the SegResNet and 3D U-Net architectures of this project, including the original notebook's `segresnet_best.pth` / `unet_best.pth`. Checkpoints are not stored in Git (they are large and, if trained on restricted data, not freely redistributable); keep them in your cloud storage.
Next to each checkpoint, `<name>.card.json` holds its training record and evaluations. **No card = "Not validated".** The Dashboard shows the evidence level, training datasets and split.

## 5. Supported MRI formats
* **NIfTI** (`.nii`, `.nii.gz`): four 3D volumes, one per modality.
* **DICOM series (MR only)**: one series folder per modality; geometry is rebuilt from the headers. Rejected: non-MR series, several series mixed in one selection, duplicate or missing slices, tilted/mixed geometry.
* Compressed DICOM transfer syntaxes are not supported; export uncompressed or convert to NIfTI.

## 6. Required modalities
All four: **T1, T1ce (post-contrast), T2, FLAIR**, on the **same voxel grid** (co-registered), ideally **skull-stripped, about 1 mm isotropic** like the BraTS training data. The program does not register or guess; mismatches are rejected with an explanation. Scans prepared differently are flagged "out-of-distribution".

## 7. Uploading a scan
**MRI Upload** -> choose the four files (or "Auto-fill from folder" for BraTS-style names `*_t1`, `*_t1ce`, `*_t2`, `*_flair`), or the *DICOM series* tab. Optionally add a label map (BraTS labels 1, 2, 4): it is measured and shown as an **annotation**, never as a prediction.

## 8. Validating compatibility
Press **Validate & load scan**. You get the matrix size, voxel size, orientation and quality notes. *Errors* block analysis (missing/empty/constant modality, mismatched grids, corrupted data). *Warnings* do not block but lower trust: oblique axes, non-1 mm voxels (resampled), no spatial header, not skull-stripped, unusual brain volume. Fix the input rather than ignoring warnings.

## 9. Running the segmentation
**Analysis** -> **Run model segmentation** (needs a loaded model) or **Measure supplied annotation**. A progress bar shows the sliding-window inference; **Cancel** stops it. Options: *Flip-TTA* (4x slower; measures prediction stability), *Ignore components smaller than* (optional clean-up, default 0 = nothing removed; removals are logged).

## 10. Interpreting the results
Click or drag in a plane to move the crosshair; scroll to change slice. Colours: **green WT** (whole tumour), **red TC** (tumour core), **yellow ET** (enhancing). Left of the screen is the patient's right (radiological convention). *Always check the overlay against the images yourself.* The **Results** page separates: the model's prediction, the measurements, the model's validation evidence, quality warnings/uncertainty, limitations, and a plain interpretation that never states tumour type, grade or malignancy.

**Confidence vs accuracy vs certainty:** probability/uncertainty values describe the network's output, not correctness; accuracy (Dice etc.) is a population statistic on a research test set, not a statement about your scan; clinical certainty comes only from clinicians using all information. No number in this program is a diagnosis.

## 11. Inspecting the 3D reconstruction
**3D View**: rotate (left-drag), pan (middle-drag or Shift-drag), zoom (wheel). Per-surface visibility and transparency, a translucent brain for orientation, the maximum-diameter line, MRI slice planes at the crosshair, and optional display smoothing (volume restored; measurements always use the unsmoothed surface). Surfaces are marching-cubes meshes in scanner millimetres - they show the segmentation, not invented anatomy.

## 12. How tumour dimensions are calculated
All in millimetres from the scan's affine (never raw voxel counts):
* **Volume** = voxels x voxel volume. "+/-1 voxel" = volume after eroding/dilating by one voxel (a boundary-error range, not a confidence interval).
* **Maximum 3D diameter** = longest distance between two surface points (convex-hull vertices of the mesh). **L x W x H** = that diameter, the largest perpendicular extent, and the extent along the third axis.
* **Bounding box** = extent along scanner axes (R-L, A-P, S-I); differs from the true diameters when the tumour is oblique - shown separately on purpose.
* **Axial diameter** = longest in-plane distance in any axial slice. **Surface area** = mesh area. **Centroid** = scanner RAS mm. **Components** = 26-connected islands.
Lengths are shown to 1 mm and volumes to 0.1 cm3 - finer values would overstate what the images support. The Guide page explains each measurement in the application.

## 13. Exporting
**Export**: NIfTI mask (on the **original scan grid** with the original orientation/position; labels 1 core, 2 oedema-only, 4 enhancing), STL/OBJ/PLY meshes (scanner RAS+ mm; these formats carry no axis tag), report (JSON + HTML), measurements (CSV), images (3 planes + 3D screenshot). Existing files are never overwritten silently.

## 14. Understanding evaluation metrics
Dice (overlap, 1 best), IoU (stricter overlap), precision (over-segmentation), recall (missed tumour), specificity (almost always ~1; not informative alone), F1 (= Dice for one mask), **HD95** (boundary error in mm, 95th percentile), ASSD (mean surface distance), surface Dice (surface within 1/2 mm), volume/diameter errors, patient-level bootstrap 95% CIs. Missed regions count as failures (Dice 0), not dropped. Evidence levels: *Not validated* / *Experimental-preliminary* / *Validated on an internal held-out dataset* / *Independently evaluated on external data* - research evidence, **not** clinical approval.

## 15. Common errors
| Message | Meaning / action |
|---|---|
| Missing required modalities | Provide T1, T1ce, T2 and FLAIR. |
| ...must be co-registered on the same voxel grid | Volumes differ in size/position/orientation: register and resample first. |
| header has no sform/qform | Position/orientation undefined; mm measurements untrustworthy. |
| does not look skull-stripped | Out-of-distribution; results unreliable. |
| Voxel size ... not 1 mm isotropic | Resampled to 1 mm; accuracy for such inputs is not validated. |
| Modality=CT, not MR / Non-uniform slice spacing | Not a usable MR volume. |
| Refusing to train on CPU / NO USABLE CUDA GPU | By design: use the cloud workflow. |
| No model loaded | Put a `.pth` in `weights\`, or use annotation mode. |
| 3D viewport unavailable | Needs OpenGL 3.2; everything else works. |

## 16. Known limitations
Trained on BraTS-style adult glioma data only; behaviour on other tumour types, paediatric, post-operative or differently processed scans is unknown. Accuracy has **not** been established locally (data and weights were not available when this was written; see REPORT.md). Boundaries carry at least ~1 voxel uncertainty. DICOM support verified on synthetic series only. No external validation exists yet.

## 17. Recommended workflow for reliable results
1. Confirm the model's evidence level (Dashboard) and what it was trained/tested on. 2. Use skull-stripped, co-registered, ~1 mm inputs; fix every warning. 3. Check the overlay slice by slice in all three planes. 4. Read measurements together with their boundary ranges and methods. 5. For research claims, evaluate on a held-out or external cohort (below) - never on data the model or its selection has seen. 6. Keep the exported report with the case.

## 18. Cloud GPU training and evaluation (heavy work never runs on the laptop)
**Environment choice** (provider limits change - verify in the provider's UI): *Colab free* gives a T4 (~15 GB) with ephemeral disk and optional Google Drive persistence; sessions end unpredictably - chosen as primary because the original weights and results already live on your Drive and the BraTS 2020 Kaggle mirror downloads straight into the VM. *Kaggle Notebooks* give a weekly GPU quota (P100 or 2xT4), the public BraTS dataset mounted as an input (no download), 9-12 h sessions but only notebook-output persistence; use `scripts/cloud_run.py` unchanged with `--data-root /kaggle/input/...` and `--runs-dir /kaggle/working/runs`, and upload weights as a private Kaggle dataset. A rented VM (paid) is only worth it for runs longer than a few free sessions and needs your explicit approval.
**Run:** open `notebooks/cloud_gpu_pipeline.ipynb` in Colab, set a GPU runtime, add the Colab Secret `GITHUB_TOKEN` (fine-grained, read-only, this repo only), and run top to bottom. It audits the dataset, evaluates the original checkpoint, trains with automatic resume (`--max-hours`, atomic `last.pth` each epoch, versioned bests, `metrics.jsonl`, GPU-adaptive batch size + gradient accumulation), compares baseline vs new on the **validation** split, and runs a 3D inference test. Re-run the training cell after any disconnect.
**Datasets and splits:** `configs/datasets.yaml` registers datasets with their *own* label mappings; unverified mappings are refused. Audit each dataset (`python -m braintumor audit --dataset NAME ...`), then `python -m braintumor split-combined --audit NAME=DIR ... --legacy splits/legacy_seed42.json` builds a patient-level manifest that keeps the legacy assignments (legacy test patients become `test_legacy`: **not** an untouched benchmark), groups duplicate patients across datasets, never trains on the `external` cohort, and reports zero overlap. `braintumor.compare` gives paired baseline-vs-candidate statistics; use the validation split to choose, evaluate the final test/external split once.

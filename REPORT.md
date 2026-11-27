# Technical and scientific validation report

Brain MRI tumour segmentation, measurement and 3D reconstruction - research prototype, version 0.2.0. Report date: 2026-11-27.

> **Bottom line.** The software pipeline, evaluation framework and GUI are implemented and verified against synthetic ground truth. **No new model was trained and no model
> accuracy was measured in this work**, because this workspace contains no dataset and no checkpoint (both lived only in the original Colab/Google Drive session) and has no usable
> GPU. The only performance figures that exist are those printed by the original notebook; they could not be reproduced here and the method that produced them was optimistic
> (see 1.3). **Current evidence level of the model: Not validated** in this workspace (best case for the notebook's figures: *Experimental / preliminary*). This is a statement
> about research evidence, not clinical approval; nothing here supports clinical use.

## 1. Original project: architecture and problems found

### 1.1 What existed
One file, `brain_tumor_project.ipynb` (17 cells, Colab-only): MONAI **SegResNet** (4.70 M parameters) and **3D U-Net** (4.81 M) taking 4 channels [T1, T1ce, T2, FLAIR] and
producing 3 sigmoid channels [TC, WT, ET] (channel order verified against MONAI 1.6.2); 128^3 patches, batch 2, AdamW lr 1e-4, cosine schedule over 30 epochs, mixed precision;
BraTS 2020 (Kaggle download, 368 complete patients), patient-level 257/55/56 split from `random.Random(42)`; a 3D-print "brain cutaway" generator; a Streamlit app served through a
public Cloudflare tunnel. Task: 3D multi-label segmentation with proper 3D input - the architecture family is appropriate for BraTS; it was kept (same layers, so checkpoints stay
compatible) rather than replaced, because nothing measured here justifies a change.

### 1.2 Figures printed by the notebook (not reproducible here)

| | WT Dice | TC Dice | ET Dice | mean Dice | HD95 WT/TC/ET (mm) |
|---|---|---|---|---|---|
| SegResNet, 56 test patients | 0.891 | 0.737 | 0.626 | 0.751 | 11.3 / 12.1 / 9.8 |
| 3D U-Net, 56 test patients | 0.755 | 0.437 | 0.250 | 0.481 | 16.2 / 28.4 / 29.8 |

Training log (SegResNet): validation mean Dice 0.52 (epoch 10) -> 0.76 (epoch 30), training loss 0.83 -> 0.46 and still falling: the model was **under-trained, not converged**.
The U-Net's lower score reflects its training budget as much as its architecture, so the architecture comparison is confounded.

### 1.3 Problems (each confirmed in the notebook code/outputs)
1. **Not reproducible / not deliverable:** Colab and Drive paths, Kaggle download, no data or weights in the project folder, no requirements.
2. **Evaluation hid failures:** `compute_dice(ignore_empty=True)` and a `np.isfinite` filter drop empty-ground-truth cases and every case with an infinite HD95 (e.g. a missed region), so
   the means are optimistic, most of all for ET. Only Dice and HD95 were reported; no IoU/precision/recall/ASSD, no confidence intervals, no per-patient table, no failure analysis.
3. **Dice-only loss cannot penalise false positives for absent regions.** Measured on a toy patch: with the notebook's settings (`smooth_nr=0`) the gradient for an absent region is
   exactly 0.0, with Dice + BCE it is 0.165 (summed |grad|).
4. **Measurements were not physical:** volume = voxel count / 1000 and spacing hard-coded to 1 mm; bounding box in voxel counts; the scan affine was discarded, so exported STLs were in
   resampled-array coordinates, not scanner space; no mask export. Whole tumour was measured *after* largest-component selection, closing (2 iterations) and hole filling, which removes valid multifocal
   disease and changes volume, while TC/ET were measured raw (inconsistent). Taubin smoothing changed volume by 1.5-2.3 % (reported by the notebook).
5. **Training/inference preprocessing differed** (resampling present at inference, absent in the cached training pipeline; harmless for BraTS only because it is already 1 mm).
6. **Hard-coded CUDA** (`GradScaler("cuda")`, `torch.cuda.synchronize()`): cannot run on CPU.
7. **Privacy/safety:** patient scans served through an unauthenticated public tunnel; "clinical report"/"clinical metrics" naming and an unsupported "vs. 30-60 min manual" claim; no input QC, no uncertainty, no limitations text.
8. Split: valid patient-level shuffle (BraTS has one exam per ID) but stored with absolute Colab paths, unverified, unstratified; two models were compared on the test set.

## 2. What was implemented (package `braintumor/`)

| Area | Implementation |
|---|---|
| Spatial correctness | `geometry`: RAS reorientation (lossless axis permutation/flip) + 1 mm resampling only when needed; exact round trip to the original grid; masks are exported with the original affine. Found and fixed a real bug here (shape of 3-cycle axis permutations) via tests. |
| Input | `io`: NIfTI and MR-DICOM series (geometry rebuilt from headers; non-MR, mixed, duplicate or missing-slice series rejected); strict co-registration checks (no silent resampling); QC (not skull-stripped, anisotropic, oblique, empty/constant channels...). Errors stop the analysis. |
| Pre/post-processing | `preprocess` equals the notebook/MONAI training transforms (numerically tested). `postprocess`: ET⊂TC⊂WT by *adding* voxels; component removal is opt-in and logged; default removes nothing. |
| Model | `model`: same architectures; legacy `*_best.pth` / `*_last.pth` and new self-describing checkpoints load with `weights_only=True`; per-checkpoint `*.card.json` carries the evaluation record. |
| Inference | `inference`: sliding window with progress/cancel; optional 4-pass flip-TTA; uncertainty = threshold-sensitivity volume range, uncertain-band share, TTA spread (**not** calibrated confidences). |
| Measurement | `measure`: volume (mm3/cm3), +/-1-voxel boundary range, surface area, true max 3D Feret diameter, orthogonal L x W x H, axis-aligned box (kept distinct), max axial diameter, world-space centroid, 26-connected components; each with a method description; displayed rounded to 1 mm / 0.1 cm3. |
| 3D | `mesh`: marching cubes in scanner mm (outward normals, mirrored-affine safe); the measurement mesh is unsmoothed; optional display/print smoothing is volume-restoring and reports uncorrected drift and vertex shift. STL/OBJ/PLY export. |
| Evaluation | `metrics`, `evaluate`: Dice, IoU, precision, recall, specificity, F1 (== Dice for binary masks), HD95, ASSD, surface Dice (1/2 mm), volume error, lesion-wise detection, FP/FN volumes, size strata, ECE/Brier calibration, patient-level bootstrap CIs, BraTS empty-case conventions (penalised and unpenalised both reported), test-set use registry, pre-specified acceptance criteria and a conservative four-level evidence label. |
| Data integrity | `splits`, `audit`: patient-level manifests with hash and leakage checks (also duplicate groups); exact reproduction of the notebook split (`splits/legacy_seed42.json`, 257/55/56, contains the notebook's demo patient 283); dataset audit (correspondence, grids, labels, corruption, class balance, exact and near duplicates). |
| Training | `train`: Dice+BCE (justification in 1.3 #3), left-right flip + mild affine/noise, warm-up + cosine over the real epoch count, validation Dice with BraTS convention, early stopping, atomic checkpoints, resume, run-directory protection, saved config/environment/manifest hash, **cost estimate and a refusal to start a CPU run**; it never loads the test split. The loss and augmentation changes are **not yet ablated**. |
| Report/export | `report`, `export`: 10-section report (JSON + HTML) separating prediction / annotation / metrics / measurements / interpretation; NIfTI mask on the original grid, meshes, CSV, images; never overwrites silently. |
| GUI | PyQt6 + VTK (`gui/`): Dashboard, MRI Upload (NIfTI/DICOM, validation), Analysis (3 linked planes, overlay, background run, cancel), 3D View (opacity, brain reference, MRI planes, diameter line, measurements), Results, Export. The 3D viewport degrades to a message if OpenGL is missing. |

## 3. Dataset

Not available in this workspace. From the notebook: BraTS 2020 training, 369 folders, 368 complete (`BraTS20_Training_355` lacks a standard `_seg` file and was skipped), volumes 240x240x155,
1 mm isotropic, labels {0,1,2,4}. Limitations: adult glioma only, skull-stripped and atlas-registered; the Kaggle copy's 125-case "ValidationData" has no public labels so it cannot
measure accuracy; **no external dataset exists**. `braintumor audit` (implemented, tested on injected defects) must be run on the real data; it will also test the reconstructed split: the notebook picked the first test patient with 40-100 cm3 whole tumour (283, 46.9 cm3), so the three patients before it in the test list (296, 186, 151) must lie outside that range.

## 4. Training: configuration and actual status

* Legacy: 30 epochs on a Colab T4 (about 86 s/epoch with caching), never converged.
* New pipeline: implemented; **only smoke-tested** (2-3 CPU epochs on 14 synthetic phantoms, to verify checkpoint/resume/guards). **No real training run was made and no new checkpoint exists.**
* Hardware here: Core i7-4800MQ (4 cores), 15.6 GB RAM, Quadro K3100M (Kepler, unsupported by PyTorch 2.x), CPU-only PyTorch. Measured SegResNet 128^3: 3.4 s forward, 16.9 s per training step
  (batch 1) => about 72 min per epoch (257 patients x 16.9 s) => about 7.5 days for 150 epochs. Estimate on a T4 from the notebook's 86 s/epoch: 150 epochs ~ 3.6 h plus validation. `python -m braintumor.train --benchmark` prints the estimate for any machine.
* Continuing the legacy checkpoint is *possible* (resume logic is for new runs; legacy weights load for fine-tuning/evaluation) but not recommended blindly: the schedule ended at epoch 30 with loss still falling, so a fresh, longer run with the same architecture is the cleaner experiment.

## 5. Held-out evaluation metrics

**None produced by this pipeline** (no data, no weights). Figures in 1.2 are the notebook's and carry the caveats in 1.3. Pre-specified acceptance criteria (research assumptions, `braintumor.metrics.DEFAULT_CRITERIA`):
Dice mean 95 % CI lower bound >= 0.85 / 0.75 / 0.65 (WT/TC/ET); median HD95 (both masks non-empty) <= 8 / 10 / 10 mm; missed-region rate <= 5 %; |median WT volume error| <= 15 %.
They are anchored to published BraTS-2020 challenge performance and were **fixed after the notebook's numbers had been seen** (disclosed); they are not clinically derived and need review by a medical-imaging specialist.
Under these criteria the notebook's TC and ET means (0.737, 0.626) would not clear the lower-bound thresholds even before confidence intervals.

## 6. External validation status
None. The highest attainable level with current data is *Validated on an internal held-out dataset*; *Independently evaluated on external data* requires a different-source labelled dataset.

## 7. 3D reconstruction and measurement: verified accuracy and limits

Verified on analytic ellipsoid phantoms (synthetic - evidence of *code* correctness, not of clinical accuracy), over five orientations (RAS/LAS/LPS/axis-permuted/15-degree oblique) and anisotropic voxels:
volume within 3 % of the analytic value (discretisation), max diameter within 1.6 mm (marching cubes adds up to about one voxel), L/W/H within 1.6 mm, world centroid exact to <0.5 mm, mesh volume within 4 % of voxel volume, meshes watertight with outward normals, exported masks identical to the original grid for lossless cases and Dice >0.90 after 0.8x0.8x2 mm -> 1 mm -> back resampling.
Taubin smoothing changed a phantom's volume by 5.2 % before the volume-restoring rescale (now ~0 %, max vertex shift <1.5 mm) - one reason measurements use the unsmoothed surface.
Limits: 1 mm voxels bound length precision to about +/-1 mm and the boundary range of volume is shown; segmentation error is additional and unquantified until the real evaluation is run; display planes ignore obliquity (measurements do not); 3D print preparation (the notebook's brain cutaway) is not part of this version.

## 8. Tests

`python -m pytest tests` -> **116 passed in 97 s** (Windows 10, Python 3.12.7, torch 2.9.1 CPU, MONAI 1.6.2). All fixtures are synthetic (analytic phantoms, tiny BraTS-structured folders, a stand-in "network" that thresholds
phantom channels, random-weight networks): the tests prove the *code* is correct, **not** that any model is accurate.

| File | Tests | Covers |
|---|---|---|
| `test_geometry_io_preprocess` | 29 | RAS/1 mm working grid for 5 affine types, exact round trip, spatial-metadata preservation, NIfTI/DICOM loading and rejection paths (missing/mismatched modalities, NaN, 4D, non-MR, missing slices, ambiguous series, 4-modality DICOM), QC flags, preprocessing **numerically equal to the MONAI training transforms** |
| `test_measure_mesh` | 24 | volume/diameter/L-W-H/centroid/AABB vs analytic truth in 5 orientations + anisotropic voxels, components, boundary range, mesh normals/watertightness/world coordinates, smoothing drift, STL/OBJ/PLY round trip, post-processing never removes by default |
| `test_model_inference` | 10 | architecture sizes, output shape/probabilities, sub-ROI volumes, progress/cancel, TTA, v1 + notebook (`best`/`last`) checkpoint loading, arch detection, corrupt/incompatible checkpoint errors, cards |
| `test_metrics_splits` | 18 | Dice/IoU/precision/recall/specificity, HD95/ASSD vs an independent KD-tree, spacing scaling, empty-case conventions, lesion-wise detection, bootstrap CIs, calibration, acceptance and evidence logic, split reproduction/leakage/tamper detection |
| `test_pipeline_export` | 12 | full workflow per orientation (Dice >0.96 vs truth, centroid <0.5 mm, exported NIfTI on the original grid/affine), resampled anisotropic case, refusal of unsuitable input, annotation mode, 10-section report, HTML escaping, export, no silent overwrite |
| `test_data_train_eval` | 8 | audit with injected defects (duplicate, misaligned mask, corrupt file, missing file, bad label), nested Kaggle layout, training loop + checkpoint + **no test patient ever loaded**, resume, CPU refusal, cost estimate, evaluation outputs and conservative evidence label, config/manifest validity |
| `test_gui` | 12 | click/crosshair/pixel mapping, radiological orientation, 3D texture alignment with world coordinates, full workflow (upload -> run -> 3D -> results -> export -> history), error paths, cancel, checkpoint/evidence display |
| `test_cli` | 3 | `split`, `audit`, headless `analyze` |

Defects found by these tests and fixed: (1) wrong working-grid shape for 3-cycle axis permutations; (2) vertically flipped MRI planes in the 3D view (pyvista's top-down texture convention; the new test fails without the fix - mutation-checked);
(3) a QThread lifetime bug that crashed the GUI intermittently on cancel; (4) 5.2 % volume drift of Taubin smoothing (now volume-restoring and reported).
GUI stability: the GUI file passed 10/10 consecutive full runs after fix (3), and 19 of 20 runs overall since it; one run in an earlier loop timed out after ~193 s and did not recur (cause not diagnosed; the waiting tests now print the app state if it happens again).
The 3D viewport requires a real OpenGL desktop session, so the GUI tests cannot run headless/offscreen.

## 9. Requirements
Python 3.12, `requirements.txt` (torch 2.9.1, MONAI 1.6.2, nibabel, scikit-image, trimesh, pydicom, PyQt6, pyvista/VTK). Inference: CPU works (about 1-2 min per scan on this machine, measured 3.4 s per 128^3 window); training: CUDA GPU with >= 9 GB.
GUI needs OpenGL 3.2 (verified on Intel HD 4600). Import torch before Qt on this Windows setup (handled).

## 10. Known limitations and unresolved issues
* **Blocker:** no data and no weights here. Real end-to-end verification on BraTS volumes, the split consistency check, the first real evaluation and any training remain to be run (commands in README). Until then the evidence level is *Not validated*.
* The loss/augmentation/schedule changes are principled but unablated; they may or may not improve accuracy.
* DICOM loading is verified only on synthetic series (no scanner data, no compressed transfer syntaxes). Input QC for "not BraTS-like" scans is heuristic.
* Uncertainty outputs are descriptive (threshold sensitivity, TTA spread); calibration is implemented but not yet measured on real data.
* GUI tested on one Windows machine (Intel HD 4600, OpenGL 3.2); no high-DPI or other-OS testing.
* The ID list behind the reconstructed split comes from the notebook's rule (001-369 except 355); `audit --manifest` must confirm it against the real folder listing.

## 11. Completed / partial / proposed
**Completed and tested:** geometry, I/O (NIfTI, DICOM), preprocessing equivalence, inference plumbing, measurements, meshes, export, reports, metrics, splits, audit, train/evaluate code paths, CLI, GUI.
**Partial:** training and evaluation have run only on phantoms; the legacy checkpoints could not be loaded for lack of files (loading code is tested on layout-identical checkpoints).
**Proposed:** train on a GPU (150 epochs) and evaluate once on the untouched test split; ablate Dice+BCE and augmentation; try 5-fold patient-level CV for tighter CIs; obtain an external labelled set; have a specialist review the acceptance criteria.

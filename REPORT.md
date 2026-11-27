# Technical and scientific validation report

Brain MRI tumour segmentation, measurement and 3D reconstruction - research prototype, version 0.3.0. Report date: 2026-11-27.

> **Bottom line.** The pipeline, evaluation framework, cloud runner and GUI are implemented and verified against synthetic ground truth (138 automated tests pass).
> **No model was trained or evaluated in this work.** The BraTS data and the original checkpoints are not on this machine (they live in your Colab/Google Drive), this machine has no
> usable GPU, and neither Colab nor Kaggle is signed in here, so no cloud run could be started. The only accuracy figures that exist are the notebook's own; they could not be
> reproduced and were produced by an optimistic method (1.3). **Current evidence level of the model: Not validated.** This is research evidence, not clinical approval; nothing here supports clinical use.

## 1. Original project: architecture and problems found

### 1.1 What existed
One file, the Colab notebook (now `notebooks/legacy/brain_tumor_project.ipynb`, byte-identical, backed up outside the repo): MONAI **SegResNet** (4.70 M parameters) and **3D U-Net** (4.81 M), 4 input channels [T1, T1ce, T2, FLAIR],
3 sigmoid outputs [TC, WT, ET] (order verified against MONAI 1.6.2); 128^3 patches, batch 2, AdamW 1e-4, cosine over 30 epochs, AMP; BraTS 2020 (368 complete patients), patient-level 257/55/56 split from
`random.Random(42)`; a 3D-print cutaway generator; a Streamlit app behind a public Cloudflare tunnel. The architecture family is appropriate for BraTS-style 3D multi-label segmentation and was **kept** (same layers, so
checkpoints stay compatible): nothing measured here justifies replacing it. nnU-Net / attention models are candidates only for controlled experiments after a baseline is measured.

### 1.2 Figures printed by the notebook (not reproducible here)
| | WT Dice | TC Dice | ET Dice | mean | HD95 WT/TC/ET (mm) |
|---|---|---|---|---|---|
| SegResNet, 56 test patients | 0.891 | 0.737 | 0.626 | 0.751 | 11.3 / 12.1 / 9.8 |
| 3D U-Net, 56 test patients | 0.755 | 0.437 | 0.250 | 0.481 | 16.2 / 28.4 / 29.8 |

SegResNet validation mean Dice 0.52 (epoch 10) -> 0.76 (epoch 30) with training loss still falling (0.83 -> 0.46): **under-trained, not converged**. The U-Net comparison is confounded by training budget.

### 1.3 Problems (confirmed in the notebook code/outputs)
1. **Not reproducible:** Colab/Drive paths, no data or weights in the project, no requirements.
2. **Evaluation hid failures:** `ignore_empty=True` + an `isfinite` filter drop empty-GT and missed-region (infinite HD95) cases; only Dice/HD95, no CIs, no per-patient table.
3. **Dice-only loss cannot penalise false positives for an absent region.** Measured: gradient for an absent region is exactly 0.0 with the notebook's settings and 0.165 with Dice+BCE (also a regression test).
4. **Measurements were not physical:** voxel size hard-coded to 1 mm, boxes in voxel counts, affine discarded (STLs in array coordinates), no mask export; whole tumour measured after largest-component/closing/hole-filling (drops multifocal disease), TC/ET raw; Taubin drift 1.5-2.3 %.
5. Training/inference preprocessing differed (resampling only at inference); hard-coded CUDA calls; patient scans served through an unauthenticated public tunnel; "clinical report" wording and an unsupported "vs 30-60 min manual" claim; no QC, uncertainty or limitations text.

### 1.4 Provenance of `segresnet_best.pth` and status of the legacy test split (checked against the notebook)
* **Procedure (supported by code):** in both versions of `train_model` (cells 6, 8) weights are updated only from `train_loader` and the best epoch is chosen only on `val_loader`; neither references the test split. The test split first appears in cells 12-13, after training. Training and evaluation use the same saved split; it is deterministic (`Random(42)` over the sorted ids), and the reconstruction (`splits/legacy_seed42.json`, 257/55/56) reproduces the notebook's sizes and its demo patient 283.
* **File (not verifiable):** no checkpoint hash or timestamp was recorded; the SegResNet epoch-3 log is missing (cell 9 "resumed from epoch 3"); the U-Net's epochs 1-15 are not in this notebook; two Colab sessions are visible. I therefore **cannot attest** that the file you hold is exactly the file produced by these cells, and the evaluation is run **without** `--assert-trained-on-manifest`.
* **Not an untouched benchmark:** cell 12 evaluated SegResNet *and* U-Net on the test split and cell 13 called SegResNet the "winning" model. `splits/legacy_seed42.test_registry.json` records this (2 models compared), and the evaluator then refuses to label that split untouched. Legacy results are kept for reproducibility only; **future claims need a fresh test set and an external cohort** (section 3-4).

## 2. What was implemented (package `braintumor/`, `scripts/`, `notebooks/`)
| Area | Implementation |
|---|---|
| Spatial correctness | RAS reorientation (lossless) + 1 mm resampling only when needed; exact round trip to the original grid; masks exported on the original grid with the original affine. A real bug (3-cycle axis permutations) was found and fixed by tests. |
| Input | NIfTI and MR-DICOM series (geometry rebuilt from headers; non-MR/mixed/duplicate/missing slices rejected); strict co-registration checks; QC (not skull-stripped, anisotropic, oblique, empty/constant). |
| Processing | Preprocessing numerically equal to the training transforms; ET⊂TC⊂WT by *adding* voxels; component removal opt-in and logged; uncertainty = threshold-sensitivity range, uncertain-band share, TTA spread (not calibrated confidence). |
| Measurement/3D | Volume, +/-1-voxel range, surface area, true max 3D (Feret) diameter, orthogonal L x W x H, axis-aligned box (kept separate), axial diameter, world centroid, components - each with a method note; unsmoothed marching-cubes surfaces in scanner mm; optional volume-restoring display smoothing. |
| Evaluation | Dice, IoU, precision, recall, specificity, F1, HD95, ASSD, surface Dice, volume error, **new: predicted-vs-truth measurement errors** (volume abs/rel, max diameter, L/W/H, axial diameter, surface area, centroid distance), lesion-wise detection, FP/FN volumes, size strata, ECE/Brier, patient-level bootstrap CIs, **per-dataset breakdown**, BraTS empty-case conventions, **`test`/`test_legacy`/`external` splits**, test-use registry with historical comparisons, pre-specified acceptance criteria, four-level evidence label (one evaluation per split kept in the checkpoint card). **`braintumor.compare`**: paired baseline-vs-candidate differences with bootstrap CI + Wilcoxon, refusing to call unlike conditions comparable. |
| Datasets/splits | **`configs/datasets.yaml` registry**: per-dataset file patterns and *own* label mapping; unverified or undeclared mappings are refused for training; audit validates observed labels against the declaration. **Cross-dataset duplicate detection** (brain-masked FLAIR fingerprint + file hash), union-find grouping so duplicates are never separated, **`split-combined`**: keeps legacy assignments (legacy test -> `test_legacy`), places new patients in train/val/**fresh test**, reserves an **external** cohort (never trained on; overlap with the pool is removed), verifies zero overlap, refuses over-permissive duplicate thresholds, writes a split audit. Legacy manifest unchanged. |
| Training (cloud-ready) | Dice+BCE / dicefocal / generalized-Dice options; left-right flip + mild affine/noise; warm-up + cosine; GPU-adaptive micro-batch + gradient accumulation; AMP; atomic `last.pth` every epoch (model, optimiser, scheduler, scaler, history); versioned bests; `metrics.jsonl`; wall-clock budget with clean pause and `--resume`; cache on local disk separate from persistent run dir; environment/git/config hash recorded; **refuses CPU**; never loads test/external splits. Loss/augmentation changes are principled but **not ablated**. |
| Cloud | `scripts/cloud_run.py` (check-gpu `--require`, get-brats2020 with layout verification, audit, eval, train with auto-resume, infer3d) and `notebooks/cloud_gpu_pipeline.ipynb` (Colab; token only via Colab Secrets; writes to a new `runs_v2` folder, reads old Drive files only). |
| GUI | PyQt6 + VTK: Dashboard (model, evidence level, training datasets/split, hardware, history, **first-use checklist**), MRI Upload, Analysis, 3D View, Results, Export, **Guide** page (modalities, regions, measurement methods, metric definitions, confidence vs accuracy vs clinical certainty, errors), welcome walkthrough, tooltips on every important control. |
| Docs/Repo | One `USER_GUIDE.md` (17 required topics + cloud workflow), README, this report; private GitHub repository. |

## 3. Datasets: assessment (sources checked 2026-11-27; nothing below was downloaded)
| Dataset | Access / size | Labels | Licence | Overlap | Decision |
|---|---|---|---|---|---|
| **BraTS 2020 train** (Kaggle mirror, used by the notebook) | public, 369 folders / 368 complete | {1,2,4} observed in the notebook | BraTS data conditions | - | Primary pool. Not on this PC; downloads inside Colab. |
| **BraTS 2021 train** | Synapse syn25829067: registration + data request (**your account/action**); 1251 labelled | ET = 4 per the official page (unverified on files) | Synapse conditions | Official page: a **superset of BraTS 2020** (and includes UCSF-PDGM/UPenn parts) | Can add roughly 880 new patients **only after** duplicate grouping against 2020; cannot be external validation for a 2020-trained model. |
| **UCSF-PDGM** (TCIA) | 495 patients, **142 GB** NIfTI via IBM Aspera Connect (no CLI documented), 3T, skull-stripped, 1 mm | page names the compartments but **not the integer values** | CC BY 4.0 + TCIA usage policy | In BraTS 2021; TCIA metadata CSV maps IDs to BraTS 2021 | Best candidate **external** cohort relative to a 2020-only pool, but needs a subset download (142 GB is not practical on free tiers), label verification on data, and duplicate check. Registry entry `verified: false`. |
| **UPenn-GBM** (TCIA) | 630 patients | not verified | **conflicting** (CC BY 4.0 vs 3.0 on different TCIA pages) - confirm first | Penn/CBICA cases also occur in BraTS | Weak as external (likely overlap); not used. |
| **BraTS 2023 glioma** | Synapse data-access form + invitation (users report the group not accepting new members) | ET = 3 (forum report, unverified) | BraTS conditions | superset of 2021 | Blocked on access; label shift handled by the registry (ET integer is per dataset). |

**Integrated so far: none** (no new data could be acquired). Label mapping is validated programmatically per dataset (audit) and a mismatch blocks split building (tested). The split tooling is ready for the day datasets are audited in the cloud.

## 4. Splits
* Existing and verified: `splits/legacy_seed42.json` - 257 / 55 / 56 patients, zero overlap, single source (BraTS 2020), unchanged. Its ids are reconstructed from the notebook rule and still need the real-folder check (`audit --manifest`).
* Designed, implemented and tested, **not generated** (needs audits of real data): `splits/combined_patient_level_v1.json` via `split-combined` (config in USER_GUIDE section 18). It will contain `train`, `val`, `test` (new patients only: the only possible untouched test set), `test_legacy` (historical, contaminated), `external`, plus a split audit (counts, overlap = 0, dataset/size/label distribution, duplicates, rejections).

## 5. Training: configuration and actual status
* Legacy: 30 epochs on a Colab T4 (~86 s/epoch cached), not converged. **Retraining (not blind continuation) is justified**: validation Dice was still rising when the schedule ended; the cleaner experiment is a fresh, longer run (150 epochs, ~3.6 h on a T4 plus validation) with the same architecture and split, compared to the baseline on the **validation** split under identical conditions (`compare`). Architecture/loss ablations (U-Net, nnU-Net, attention models, dicefocal, gdice) are proposed as controlled validation-split experiments, not run.
* **No training run was made.** The training loop was exercised only on 3-epoch CPU phantom smoke tests (code paths: pause, resume, accumulation, versioning). This machine: i7-4800MQ, 15.6 GB, Quadro K3100M (Kepler, unsupported by PyTorch 2.x); measured 16.9 s per 128^3 step => ~72 min/epoch => ~7.5 days for 150 epochs: **not feasible locally, by design not attempted**.

## 6. Held-out evaluation and external validation
**None produced** (no data, no weights here). Tasks A-C were prepared (`notebooks/cloud_gpu_pipeline.ipynb` stages) but **not executed**. Pre-specified research acceptance criteria (`braintumor.metrics.DEFAULT_CRITERIA`): Dice mean 95 % CI lower bound >= 0.85 / 0.75 / 0.65 (WT/TC/ET); median HD95 <= 8 / 10 / 10 mm; missed rate <= 5 %; |median WT volume error| <= 15 % - anchored to published BraTS-2020 performance, **fixed after the notebook numbers had been seen** (disclosed), not clinically derived. External validation: none.
**Honest classification: Not validated.** Best attainable with the legacy split alone: *Experimental / preliminary* (the split is not untouched). *Validated on an internal held-out dataset* requires a fresh test set; *Independently evaluated on external data* requires an external cohort.

## 7. 3D reconstruction and measurement: verified accuracy and limits
On analytic ellipsoid phantoms (synthetic: evidence of *code* correctness only), five orientations plus anisotropic voxels: volume within 3 % (discretisation), max diameter within 1.6 mm (marching cubes adds up to ~1 voxel), L/W/H within 1.6 mm, centroid exact to <0.5 mm, mesh volume within 4 %, watertight outward-normal meshes, exports identical on the original grid (Dice >0.90 after 0.8x0.8x2 mm -> 1 mm -> back). Taubin smoothing drifted a phantom's volume by 5.2 % (now restored; measurements use the unsmoothed surface). Segmentation error itself is unquantified until the real evaluation exists.

## 8. Tests
`python -m pytest tests` -> **138 passed in 207 s** (Windows 10, Python 3.12.7, torch 2.9.1 CPU, MONAI 1.6.2). Fixtures are synthetic (analytic phantoms, tiny BraTS-structured folders, a threshold "stand-in network", random-weight networks): they prove the *code*, not any model.
geometry/io/preprocess 29 | measure/mesh 24 | metrics/splits 18 | model/inference 10 | pipeline/export 12 | data/train/eval 8 | **multidataset 15** (registry gating, label conversion == MONAI, label-mismatch blocking, cross-dataset duplicates, combined split with legacy preservation/external protection/determinism, giant-group refusal, measurement errors, paired comparison, evidence rules incl. `test_legacy`/`external`, loss gradients, time-budget pause + resume + accumulation) | **cloud_run 4** (no silent CPU fallback, resume through the script, layout verification) | GUI 15 | CLI 3.
Defects found by tests and fixed: wrong working-grid shape for axis 3-cycles; flipped MRI planes in 3D (mutation-checked); QThread lifetime crash on cancel; 5.2 % smoothing drift; dataset-wide flag from a single bad label. GUI stability: 10/10 consecutive full runs passed after the thread fix (19 of 20 overall; one earlier timeout never recurred and was not diagnosed). 3D tests need a real OpenGL desktop session.

## 9. Requirements
Python 3.12, `requirements.txt` (torch 2.9.1, MONAI 1.6.2, nibabel, scikit-image, trimesh, pydicom, PyQt6, pyvista/VTK). Inference: CPU works (~1-2 min per scan here). Training: CUDA GPU >= ~11 GB for micro-batch 2 (smaller GPUs use micro-batch 1 + accumulation). Import torch before Qt on Windows (handled).

## 10. Known limitations and unresolved issues
* **Blockers needing you:** (1) data and weights are not on this PC - they must be reached from Colab/Drive; (2) Colab/Kaggle are **not signed in** (no credentials on the machine): sign in (Colab) and approve the Drive consent, add the Colab Secret `GITHUB_TOKEN`; (3) BraTS 2021/2023 need Synapse access under your account; UCSF-PDGM needs Aspera and a subset choice.
* The legacy checkpoint's provenance is unverifiable (1.4); the legacy test split is contaminated by design and cannot support independent claims.
* Loss/augmentation/schedule changes are unablated; all recipes beyond the notebook's are hypotheses until the validation-split comparison is run.
* Duplicate detection is fingerprint-based: its threshold must be calibrated on real data (the audit prints the similarity distribution; the builder refuses absurd groupings).
* DICOM verified on synthetic series only; QC for "not BraTS-like" scans is heuristic; uncertainty is descriptive; calibration is implemented but unmeasured; GUI tested on one machine; provider limits quoted in USER_GUIDE are approximate and provider-controlled.

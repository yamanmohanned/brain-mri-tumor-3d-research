"""User-facing explanations shown in the GUI (Guide page, welcome dialog, tooltips). Definitions match the code (metrics.py, measure.py)."""

TOOLTIPS = {
    "nifti_pick": "Select one 3D NIfTI file (.nii or .nii.gz) for this modality. All four must cover the same brain on the same voxel grid.",
    "autofill": "Pick a folder whose file names end in _t1, _t1ce, _t2, _flair (BraTS style); matching files are filled in for you.",
    "dicom_pick": "Pick a folder holding ONE MR DICOM series for this modality. Non-MR series, mixed series and series with missing slices are rejected.",
    "seg_pick": "Optional: a label map (labels 1, 2, 4 as in BraTS). It is measured and shown as an ANNOTATION - never as a model prediction.",
    "validate": "Loads the files, checks that the four modalities share one grid, reads the spatial metadata and runs quality checks. Nothing is analysed yet.",
    "next": "Enabled only when the scan passed validation (no blocking errors).",
    "modality": "Choose which MRI contrast is shown behind the overlay. FLAIR usually shows the oedema best, T1ce the enhancing tumour.",
    "overlay_wt": "Whole tumour: necrotic/non-enhancing core + oedema + enhancing tumour (green).",
    "overlay_tc": "Tumour core: necrotic/non-enhancing core + enhancing tumour (red).",
    "overlay_et": "Enhancing tumour: the part that lights up after contrast (yellow).",
    "opacity": "How strongly the segmentation is drawn over the MRI.",
    "tta": "Runs the model 4 times on flipped copies and measures how much the results disagree (an uncertainty hint). 4x slower. Does not make the result more correct.",
    "min_component": "Optional clean-up: drop predicted islands smaller than this volume. 0 = keep everything (default). Anything removed is listed in the report.",
    "run_model": "Runs the loaded neural network on this scan. On a CPU this takes roughly 1-2 minutes. You can cancel at any time.",
    "run_annotation": "Measures and reconstructs the segmentation you supplied on the Upload page - no model is run.",
    "cancel": "Stops the running analysis; nothing is saved.",
    "plane_axial": "Horizontal slices (top-down). Left of the screen is the patient's RIGHT (radiological convention).",
    "view3d_wt": "Show/hide the surface. The slider sets its transparency.",
    "view3d_brain": "A translucent brain outline for orientation only. It is not used for any measurement.",
    "view3d_feret": "The longest straight distance between two points of the tumour surface (the true maximum diameter).",
    "view3d_planes": "Shows the MRI slices at the crosshair position inside the 3D scene so you can relate the surface to the anatomy.",
    "view3d_smooth": "Display-only smoothing of the surface. Volume is restored and measurements always use the unsmoothed surface.",
    "export_mask": "NIfTI label map on the ORIGINAL scan grid with the original orientation/position (1 = core, 2 = oedema only, 4 = enhancing).",
    "export_meshes": "3D surfaces in scanner millimetres (RAS+). STL/OBJ/PLY carry no coordinate tag, so import them knowing that.",
    "export_report": "Machine-readable JSON and a human-readable HTML report with all measurements, methods, warnings and limitations.",
    "export_csv": "One row per region with all measurements, for spreadsheets or statistics.",
    "export_images": "Axial/coronal/sagittal overlay image and a screenshot of the 3D view.",
    "load_model": "Loads a trained checkpoint (.pth). Only checkpoints that match the supported architectures are accepted.",
}

WELCOME_STEPS = [
    ("Load a model", "On the Dashboard, load a trained checkpoint (.pth). Without one you can still measure and reconstruct a segmentation you supply."),
    ("Upload and validate", "On MRI Upload, choose the four modalities (T1, T1ce, T2, FLAIR) as NIfTI files or DICOM series and press 'Validate & load scan'. Read any warnings."),
    ("Run the segmentation", "On Analysis, press 'Run model segmentation'. Scroll through the three planes and check the overlay against the images yourself."),
    ("Inspect in 3D", "On 3D View, rotate, zoom and adjust transparency; read the physical measurements next to the model."),
    ("Read Results", "Results lists measurements, how they were computed, uncertainty, warnings, limitations and the model's validation evidence."),
    ("Export", "On Export, save the mask (NIfTI), meshes, report, CSV and images."),
]

GUIDE_HTML = """
<h2>Quick start</h2>
<ol>
<li><b>Load a model</b> (Dashboard). The evidence level shown there tells you how well the model has actually been tested.</li>
<li><b>Upload</b> four co-registered modalities and press <i>Validate &amp; load scan</i>.</li>
<li><b>Run segmentation</b> (Analysis) and check the overlay against the MRI yourself.</li>
<li><b>3D View</b> for the reconstructed tumour and its measurements; <b>Results</b> for the full report; <b>Export</b> to save.</li>
</ol>

<h2>What the four MRI modalities are</h2>
<table border="1" cellpadding="4" cellspacing="0">
<tr><th>Modality</th><th>What it shows</th><th>Why the model needs it</th></tr>
<tr><td><b>T1</b></td><td>Native anatomy; fluid dark, fat bright</td><td>Structure; necrotic/non-enhancing core is often dark</td></tr>
<tr><td><b>T1ce</b></td><td>T1 after gadolinium contrast</td><td>The <i>enhancing tumour</i> lights up where the blood-brain barrier is disrupted</td></tr>
<tr><td><b>T2</b></td><td>Water bright</td><td>Tumour and oedema appear bright</td></tr>
<tr><td><b>FLAIR</b></td><td>T2 with the fluid signal of the ventricles suppressed</td><td>Peritumoural oedema / signal change is easiest to see</td></tr>
</table>
<p>All four are required, on the <b>same voxel grid</b> (co-registered), ideally <b>skull-stripped and about 1 mm isotropic</b>, like the BraTS training data. Scans prepared differently are flagged as out-of-distribution and the result may be unreliable.</p>

<h2>Tumour regions</h2>
<ul>
<li><b>WT - whole tumour</b>: necrotic/non-enhancing core + oedema + enhancing tumour (green)</li>
<li><b>TC - tumour core</b>: necrotic/non-enhancing core + enhancing tumour (red)</li>
<li><b>ET - enhancing tumour</b>: the contrast-enhancing part (yellow)</li>
</ul>
<p>Regions are nested (ET inside TC inside WT). The 'oedema' label also contains signal change that is not necessarily tumour cells.</p>

<h2>How the measurements are calculated</h2>
<ul>
<li><b>Everything is in millimetres</b>, using the scan's own voxel size and orientation (the affine) - never raw voxel counts.</li>
<li><b>Volume</b> = number of voxels x the volume of one voxel. The '+/-1 voxel' range shows how much it changes if every boundary were off by one voxel.</li>
<li><b>Maximum 3D diameter</b> = the longest distance between any two points of the tumour surface. <b>Length / Width / Height</b> = that diameter, the largest extent perpendicular to it, and the extent along the remaining axis.</li>
<li><b>Bounding box</b> = extent along the scanner axes (R-L, A-P, S-I). It depends on head position and can be smaller than the true diameter of an obliquely oriented tumour - the two are shown separately on purpose.</li>
<li><b>Axial diameter</b> = the longest in-plane distance in any axial slice (the kind of measurement used in 2D reporting).</li>
<li>Lengths are rounded to 1 mm and volumes to 0.1 cm3 because the images cannot support more precision.</li>
</ul>

<h2>Model confidence vs. segmentation accuracy vs. clinical certainty</h2>
<table border="1" cellpadding="4" cellspacing="0">
<tr><th>Term</th><th>What it is</th><th>What it is NOT</th></tr>
<tr><td><b>Model confidence</b> (probability maps, uncertainty bands, TTA spread)</td><td>How strongly the network's output favours 'tumour' at a voxel, and how stable that is</td><td>A measure of being correct. Networks can be confidently wrong, especially on unfamiliar scans. These values are not calibrated probabilities unless a calibration study says so.</td></tr>
<tr><td><b>Segmentation accuracy</b> (Dice, HD95, ...)</td><td>Agreement with expert outlines on a <i>research test set</i>, averaged over patients</td><td>A statement about this particular scan. Expert outlines also differ from each other.</td></tr>
<tr><td><b>Clinical certainty</b></td><td>A diagnosis or treatment decision made by qualified clinicians using all clinical, imaging and pathology information</td><td>Anything this software provides. No number here is a diagnosis, grade, malignancy statement or treatment recommendation.</td></tr>
</table>

<h2>Evaluation metrics in plain language</h2>
<table border="1" cellpadding="4" cellspacing="0">
<tr><th>Metric</th><th>Meaning</th><th>Good direction</th></tr>
<tr><td><b>Dice (DSC)</b></td><td>2 x overlap / (predicted size + true size). 1 = identical, 0 = no overlap. Forgiving of small boundary errors on large tumours; unstable for tiny lesions.</td><td>higher</td></tr>
<tr><td><b>IoU</b></td><td>Overlap / union of prediction and truth (Jaccard). Stricter than Dice.</td><td>higher</td></tr>
<tr><td><b>Precision</b></td><td>Share of the predicted tumour that is really tumour. Low = over-segmentation (false positives).</td><td>higher</td></tr>
<tr><td><b>Recall / sensitivity</b></td><td>Share of the true tumour that was found. Low = missed tumour (false negatives).</td><td>higher</td></tr>
<tr><td><b>Specificity</b></td><td>Share of healthy voxels correctly left empty. Almost always ~1 because background dominates - not informative alone.</td><td>higher</td></tr>
<tr><td><b>F1</b></td><td>Identical to Dice for a single binary mask.</td><td>higher</td></tr>
<tr><td><b>HD95 (mm)</b></td><td>95th percentile of the distances between predicted and true surfaces. Shows boundary errors that Dice hides.</td><td>lower</td></tr>
<tr><td><b>ASSD (mm)</b></td><td>Average distance between the two surfaces.</td><td>lower</td></tr>
<tr><td><b>Surface Dice</b></td><td>Share of surface points within a tolerance (1 or 2 mm) of the other surface.</td><td>higher</td></tr>
<tr><td><b>Volume / diameter error</b></td><td>Predicted minus true measurement (absolute and relative).</td><td>closer to 0</td></tr>
<tr><td><b>95% CI</b></td><td>Range that the average would likely fall in if the study were repeated; computed by resampling patients (bootstrap).</td><td>narrow</td></tr>
</table>
<p>Cases where a region is missed entirely are counted as failures (Dice 0), not silently dropped.</p>

<h2>Evidence levels (research validation, not clinical approval)</h2>
<ul>
<li><b>Not validated</b> - no held-out evaluation is attached to this checkpoint.</li>
<li><b>Experimental / preliminary</b> - evaluated, but evidence is incomplete or the pre-specified criteria were not met.</li>
<li><b>Validated on an internal held-out dataset</b> - untouched patient-level test set, deployed pipeline, confidence intervals, criteria met.</li>
<li><b>Independently evaluated on external data</b> - additionally evaluated on a different-source cohort that was never trained on.</li>
</ul>

<h2>Common problems</h2>
<table border="1" cellpadding="4" cellspacing="0">
<tr><th>Message</th><th>What to do</th></tr>
<tr><td>Missing required modalities</td><td>Provide all of T1, T1ce, T2, FLAIR. The model cannot run on fewer.</td></tr>
<tr><td>...must be co-registered on the same voxel grid</td><td>The four volumes differ in size/position. Register and resample them to one grid first; the program will not guess.</td></tr>
<tr><td>header has no sform/qform</td><td>The file does not say where it is in space. Millimetre measurements and orientation are not trustworthy.</td></tr>
<tr><td>does not look skull-stripped</td><td>The model was trained on skull-stripped data; results on raw heads are unreliable.</td></tr>
<tr><td>Voxel size is not 1 mm isotropic</td><td>The scan is resampled to 1 mm for the model; accuracy for such inputs is not validated.</td></tr>
<tr><td>Modality=CT, not MR / non-uniform slice spacing</td><td>The DICOM series is not a usable MR volume (wrong type or missing slices).</td></tr>
<tr><td>No model loaded</td><td>Place a .pth in the weights folder or use 'Browse for checkpoint'. Annotation mode needs no model.</td></tr>
<tr><td>3D viewport unavailable</td><td>OpenGL 3.2 is required. Measurements, reports and exports still work.</td></tr>
</table>
"""

METRIC_NOTE = ("Probability/uncertainty values are NOT accuracy, and accuracy on a research test set is NOT clinical certainty. "
               "Always review the segmentation against the images.")

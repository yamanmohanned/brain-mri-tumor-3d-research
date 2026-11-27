"""End-to-end analysis: validate -> preprocess -> segment -> post-process -> mesh -> measure -> report.

The same ``segment_volume`` function is used by the GUI, the CLI and the evaluation script, so what is evaluated
is exactly what is deployed.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from . import REGIONS
from .geometry import voxel_volume_mm3
from .inference import InferenceConfig, Prediction, predict_probabilities, threshold_masks
from .io import InputError, Issue, MRVolume, assess_input
from .measure import measure_all
from .mesh import brain_shell_mask, mask_to_mesh, mesh_report
from .metrics import brats_labels_to_regions
from .postprocess import postprocess, uncertainty_summary
from .preprocess import PreparedInput, embed, prepare


@dataclass
class AnalysisConfig:
    infer: InferenceConfig = field(default_factory=InferenceConfig)
    hierarchy: bool = True
    min_component_mm3: float = 0.0         # 0 = never remove a predicted component
    with_axial: bool = True
    brain_shell: bool = True


@dataclass
class AnalysisResult:
    source: str                             # "model" | "external_mask"
    source_detail: str
    volume: dict                            # MRVolume.describe()
    qc: list
    prep: PreparedInput
    masks: dict                             # region -> bool [X,Y,Z] on the working grid
    meshes: dict                            # region -> trimesh | None (faithful, unsmoothed)
    measurements: dict
    postprocess_log: dict = field(default_factory=dict)
    uncertainty: dict | None = None
    model_info: dict | None = None
    config: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    brain_mesh: object = None
    probs: np.ndarray | None = None         # cropped-grid probabilities (model runs only)
    case_id: str = "case"

    @property
    def work_affine(self) -> np.ndarray:
        return self.prep.work_affine


def segment_volume(prep: PreparedInput, model, cfg: AnalysisConfig, progress=None, cancel=None):
    """Model inference + thresholding + conservative post-processing on a prepared input.

    Returns ``(prediction, masks_on_working_grid, masks_on_cropped_grid, postprocess_log)``.
    """
    pred: Prediction = predict_probabilities(model, prep.tensor, cfg.infer, progress, cancel)
    vv = voxel_volume_mm3(prep.work_affine)
    cropped, log = postprocess(threshold_masks(pred.probs, cfg.infer.threshold), vv, cfg.hierarchy, cfg.min_component_mm3)
    full = {r: embed(m, prep) for r, m in cropped.items()}
    return pred, full, cropped, log


def require_valid(vol: MRVolume) -> list[Issue]:
    issues = assess_input(vol)
    errors = [i.message for i in issues if i.severity == "error"]
    if errors:
        raise InputError("Input is not suitable for analysis: " + " | ".join(errors))
    return issues


def analyze(vol: MRVolume, model=None, model_info: dict | None = None, cfg: AnalysisConfig | None = None,
            labels: np.ndarray | None = None, label_source: str = "", progress: Callable | None = None,
            cancel: Callable | None = None, case_id: str = "case") -> AnalysisResult:
    """Run the whole workflow. Pass ``model`` for a prediction, or ``labels`` (BraTS label map on the scan's own grid)
    to measure/reconstruct an existing segmentation (e.g. ground truth) without any model."""
    cfg = cfg or AnalysisConfig()
    if (model is None) == (labels is None):
        raise ValueError("Provide exactly one of model or labels")
    t0 = time.time()
    qc = require_valid(vol)
    prep = prepare(vol)
    timings = {"preprocess_s": round(time.time() - t0, 2)}
    pp_log, unc, probs = {}, None, None

    if model is not None:
        t1 = time.time()
        pred, masks, cropped, pp_log = segment_volume(prep, model, cfg, progress, cancel)
        probs = pred.probs
        unc = uncertainty_summary(pred.probs, cropped, voxel_volume_mm3(prep.work_affine), pred.tta_std)
        timings["inference_s"] = round(time.time() - t1, 2)
        source, detail = "model", (model_info or {}).get("path", "unnamed model")
    else:
        if labels.shape != vol.shape:
            raise InputError("Label map shape differs from the MRI volume")
        reg = brats_labels_to_regions(labels)
        masks = {r: prep.grid.to_work(reg[r].astype(np.uint8), order=0).astype(bool) for r in REGIONS}
        source, detail = "external_mask", label_source or "externally supplied segmentation"

    t2 = time.time()
    meshes = {r: mask_to_mesh(masks[r], prep.work_affine) for r in REGIONS}
    measurements = measure_all(masks, prep.work_affine, meshes, cfg.with_axial)
    timings["mesh_and_measure_s"] = round(time.time() - t2, 2)
    brain = None
    if cfg.brain_shell:
        ref = prep.raw_work[vol.modalities.index("t1")]
        brain = mask_to_mesh(brain_shell_mask(ref), prep.work_affine, step_size=2)
    timings["total_s"] = round(time.time() - t0, 2)
    cfg_dict = {"infer": asdict(cfg.infer), "hierarchy": cfg.hierarchy, "min_component_mm3": cfg.min_component_mm3}
    return AnalysisResult(source, detail, vol.describe(), qc, prep, masks, meshes, measurements, pp_log, unc, model_info, cfg_dict,
                          timings, brain, probs, case_id)


def mesh_summary(res: AnalysisResult) -> dict:
    return {r: mesh_report(m) for r, m in res.meshes.items()}

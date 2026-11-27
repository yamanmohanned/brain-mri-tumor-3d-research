"""Brain MRI tumour segmentation, measurement and 3D reconstruction (research prototype)."""

__version__ = "0.3.0"

#: Channel order the model was trained with (notebook: [t1, t1ce, t2, flair]).
MODALITIES = ("t1", "t1ce", "t2", "flair")
#: Output channel order of ``ConvertToMultiChannelBasedOnBratsClassesd`` (verified against MONAI 1.6.2).
REGIONS = ("TC", "WT", "ET")
REGION_NAMES = {
    "WT": "Whole tumour (necrotic/non-enhancing core + oedema + enhancing)",
    "TC": "Tumour core (necrotic/non-enhancing core + enhancing)",
    "ET": "Enhancing tumour",
}
#: BraTS 2020 label values.
LABEL_NCR_NET, LABEL_ED, LABEL_ET = 1, 2, 4

RESEARCH_DISCLAIMER = (
    "EXPERIMENTAL RESEARCH SYSTEM - NOT A MEDICAL DEVICE. Outputs are model predictions, not confirmed "
    "anatomy, and must not be used for diagnosis, grading or treatment decisions. They are not a "
    "substitute for assessment by a qualified radiologist."
)

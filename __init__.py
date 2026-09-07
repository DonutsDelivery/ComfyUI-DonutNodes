"""Register DonutNodes independently so one broken wheel cannot hide the pack."""

from .donut_dependencies import (
    DonutDependencyCheck,
    IMPORT_FAILURES,
    LOGGER,
    import_component,
    installed_versions,
    opencv_conflict,
)


NODE_CLASS_MAPPINGS = {"DonutDependencyCheck": DonutDependencyCheck}
NODE_DISPLAY_NAME_MAPPINGS = {"DonutDependencyCheck": "Donut Dependency Check"}
IMPORT_FAILURES.clear()

# Settings/CivitAI routes are useful, but their dependencies must not prevent
# unrelated model/conditioning nodes (or the diagnostic node) from loading.
import_component(__name__, "shared.server_routes")

# Keep the original registration order. The overrides are deliberate:
# failed overrides must NOT quietly expose an older, incompatible node class.
_NODE_MODULES = (
    "DonutDetailer", "DonutDetailer2", "DonutDetailer4", "DonutDetailer5",
    "DonutDetailerXLBlocks", "DonutClipEncode", "DonutWidenMerge",
    "donut_lora_nodes", "DonutSafeApplyLoRAStack", "hot_reload",
    "DonutSDXLTeaCache", "DonutBlockCalibration", "DonutFrequencyAnalysis",
    "DonutSpectralNoiseSharpener", "DonutKSamplerCFGLinear", "donut_lora_civitai",
    "DonutTiledUpscale", "DonutColorPreservingUpscale", "DonutDetailerZIT",
    "ModelMergeZIT", "DonutModelMergeKrea2", "ModelMergeZITBlocks",
    "DonutModelSave", "DonutFaceDetailer", "DonutUniversalDetailer",
    "DonutWeightVectorScale", "DonutGammaCorrection", "DonutAutoGamma",
    "DonutHistogramStretch", "DonutAutoWhiteBalance", "DonutSharpen",
    "DonutPromptInjection", "DonutZitConditioningRebalance", "DonutZitLayerBlendEncode",
    "DonutKrea2ImageConditioning", "DonutKrea2FusionControl", "DonutKrea2FusionPreset",
    "DonutImageAdjust",
    "donut_prompt", "donut_seed_plan", "donut_dynamic_lora",
    "donut_upscale_stage", "donut_prompt_injection_recursive", "donut_grouped_merge",
)
_REQUIRED_OVERRIDES = {
    "DonutSafeApplyLoRAStack": ("DonutApplyLoRAStack",),
    "DonutKrea2FusionPreset": ("DonutKrea2FusionControl",),
    "donut_upscale_stage": ("DonutTiledUpscale",),
    "donut_prompt_injection_recursive": ("DonutPromptInjection",),
    "donut_grouped_merge": ("DonutModelMergeKrea2",),
}
for _module in _NODE_MODULES:
    import_component(
        __name__, _module, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS,
        overrides=_REQUIRED_OVERRIDES.get(_module, ()),
    )

# Nodes folded into multipurpose nodes stay registered for saved workflows.
_DEPRECATED_DISPLAY = {
    "DonutModelSave":                  "Model Save (No Workflow) (DEPRECATED)",
    "DonutCheckpointSave":             "Checkpoint Save (No Workflow) (DEPRECATED)",
    "ModelMergeZITBlocks":             "Model Merge ZIT Blocks (DEPRECATED)",
    "Donut Simple Calibration":        "Donut Simple Calibration (DEPRECATED)",
    "Donut Sharpener (from reference)": "Donut Sharpener (from reference) (DEPRECATED)",
    "Donut Sharpener":                 "Donut Sharpener (DEPRECATED)",
    "DonutLoRACivitAIInfo":             "Donut LoRA CivitAI Info (DEPRECATED)",
    "DonutLoRAHashLookup":             "Donut LoRA Hash Lookup (DEPRECATED)",
    "DonutSampler (Advanced)":         "DonutSampler (Advanced) (DEPRECATED)",
    "DonutMultiModelSampler":          "DonutMultiModelSampler (DEPRECATED)",
    "Donut Detailer":                  "Donut Detailer (DEPRECATED)",
    "Donut Detailer 2":                "Donut Detailer 2 (DEPRECATED)",
    "Donut Detailer 4":                "Donut Detailer 4 (DEPRECATED)",
    "DonutAutoGamma":                  "Donut Auto Gamma (DEPRECATED)",
    "DonutGammaCorrection":            "Donut Gamma Correction (DEPRECATED)",
    "DonutAutoWhiteBalance":           "Donut Auto White Balance (DEPRECATED)",
    "DonutHistogramStretch":           "Donut Histogram Stretch (DEPRECATED)",
    "DonutHiRaLoAm":                   "Donut Local Contrast (DEPRECATED)",
    "DonutCAS":                        "Donut CAS (Contrast Adaptive Sharpen) (DEPRECATED)",
    "DonutFillerModel":                "Donut Filler Model (DEPRECATED)",
    "DonutFillerClip":                 "Donut Filler Clip (DEPRECATED)",
}
for _cid, _label in _DEPRECATED_DISPLAY.items():
    _cls = NODE_CLASS_MAPPINGS.get(_cid)
    if _cls is not None:
        _cls.DEPRECATED = True
        NODE_DISPLAY_NAME_MAPPINGS[_cid] = _label

if "DonutFiller" in NODE_CLASS_MAPPINGS:
    NODE_DISPLAY_NAME_MAPPINGS["DonutFiller"] = "Donut Filler (Model + CLIP)"

_conflict = opencv_conflict(installed_versions())
if _conflict:
    LOGGER.warning("[DonutNodes] %s", _conflict)
if IMPORT_FAILURES:
    LOGGER.warning(
        "[DonutNodes] Loaded %d nodes; %d component(s) failed. "
        "Open a blank workflow and add Donut Dependency Check to inspect the errors. "
        "Failed components: %s",
        len(NODE_CLASS_MAPPINGS), len(IMPORT_FAILURES), ", ".join(IMPORT_FAILURES),
    )

WEB_DIRECTORY = "./web"

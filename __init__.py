# Register server routes for settings API
from .shared import server_routes  # noqa: F401

# existing nodes' class mappings
from .DonutDetailer         import NODE_CLASS_MAPPINGS as m1
from .DonutDetailer         import NODE_DISPLAY_NAME_MAPPINGS as d_detailer
from .DonutDetailer2        import NODE_CLASS_MAPPINGS as m2
from .DonutDetailer4        import NODE_CLASS_MAPPINGS as m3
from .DonutDetailer5        import NODE_CLASS_MAPPINGS as m4
from .DonutDetailerXLBlocks import NODE_CLASS_MAPPINGS as m5
from .DonutClipEncode       import NODE_CLASS_MAPPINGS as m6
from .DonutWidenMerge       import NODE_CLASS_MAPPINGS as m7

# new LoRA nodes (include display names)
from .donut_lora_nodes      import NODE_CLASS_MAPPINGS        as m_lora
from .donut_lora_nodes      import NODE_DISPLAY_NAME_MAPPINGS as d_lora

# hot reload functionality
from .hot_reload            import NODE_CLASS_MAPPINGS        as m_reload
from .hot_reload            import NODE_DISPLAY_NAME_MAPPINGS as d_reload

# optuna optimization (temporarily disabled - DonutOptunaNode.py not in repo)
# from .DonutOptunaNode       import NODE_CLASS_MAPPINGS        as m_optuna
# from .DonutOptunaNode       import NODE_DISPLAY_NAME_MAPPINGS as d_optuna

# SDXL TeaCache (Base - High Performance)
from .DonutSDXLTeaCache     import NODE_CLASS_MAPPINGS        as m_teacache
from .DonutSDXLTeaCache     import NODE_DISPLAY_NAME_MAPPINGS as d_teacache

# Block Calibration
from .DonutBlockCalibration import NODE_CLASS_MAPPINGS        as m_calibration
from .DonutBlockCalibration import NODE_DISPLAY_NAME_MAPPINGS as d_calibration

# Frequency Analysis
from .DonutFrequencyAnalysis import NODE_CLASS_MAPPINGS       as m_freq_analysis
from .DonutFrequencyAnalysis import NODE_DISPLAY_NAME_MAPPINGS as d_freq_analysis

# Spectral Noise Sharpening (2024 Research-Based)
from .DonutSpectralNoiseSharpener import NODE_CLASS_MAPPINGS      as m_spectral_sharpener
from .DonutSpectralNoiseSharpener import NODE_DISPLAY_NAME_MAPPINGS as d_spectral_sharpener

# DonutSampler - CFG Linear Progression
from .DonutKSamplerCFGLinear import NODE_CLASS_MAPPINGS        as m_donut_sampler
from .DonutKSamplerCFGLinear import NODE_DISPLAY_NAME_MAPPINGS as d_donut_sampler

# LoRA CivitAI Integration
from .donut_lora_civitai import NODE_CLASS_MAPPINGS        as m_lora_civitai
from .donut_lora_civitai import NODE_DISPLAY_NAME_MAPPINGS as d_lora_civitai

# Tiled Upscale
from .DonutTiledUpscale import NODE_CLASS_MAPPINGS        as m_tiled_upscale
from .DonutTiledUpscale import NODE_DISPLAY_NAME_MAPPINGS as d_tiled_upscale

# ZIT Detailer (Z-Image Turbo / Lumina2)
from .DonutDetailerZIT import NODE_CLASS_MAPPINGS        as m_zit_detailer
from .DonutDetailerZIT import NODE_DISPLAY_NAME_MAPPINGS as d_zit_detailer

# ZIT Model Merge (Z-Image Turbo / Lumina2)
from .ModelMergeZIT import NODE_CLASS_MAPPINGS        as m_merge_zit
from .ModelMergeZIT import NODE_DISPLAY_NAME_MAPPINGS as d_merge_zit

# ZIT Model Merge Blocks (individual layer control)
from .ModelMergeZITBlocks import NODE_CLASS_MAPPINGS        as m_merge_zit_blocks
from .ModelMergeZITBlocks import NODE_DISPLAY_NAME_MAPPINGS as d_merge_zit_blocks

# Model Save (No Workflow)
from .DonutModelSave import NODE_CLASS_MAPPINGS        as m_model_save
from .DonutModelSave import NODE_DISPLAY_NAME_MAPPINGS as d_model_save

# Face Detailer with max_faces limit
from .DonutFaceDetailer import NODE_CLASS_MAPPINGS        as m_face_detailer
from .DonutFaceDetailer import NODE_DISPLAY_NAME_MAPPINGS as d_face_detailer

# Universal Detailer (Florence-2)
from .DonutUniversalDetailer import NODE_CLASS_MAPPINGS        as m_universal_detailer
from .DonutUniversalDetailer import NODE_DISPLAY_NAME_MAPPINGS as d_universal_detailer

# Prompt Receiver (HTTP API)
from .DonutPromptReceiver import NODE_CLASS_MAPPINGS        as m_prompt_receiver
from .DonutPromptReceiver import NODE_DISPLAY_NAME_MAPPINGS as d_prompt_receiver

# Gamma Correction
from .DonutGammaCorrection import NODE_CLASS_MAPPINGS        as m_gamma
from .DonutGammaCorrection import NODE_DISPLAY_NAME_MAPPINGS as d_gamma

# Auto Gamma Correction (4 methods)
from .DonutAutoGamma import NODE_CLASS_MAPPINGS        as m_auto_gamma
from .DonutAutoGamma import NODE_DISPLAY_NAME_MAPPINGS as d_auto_gamma

# Histogram Stretch
from .DonutHistogramStretch import NODE_CLASS_MAPPINGS        as m_hist_stretch
from .DonutHistogramStretch import NODE_DISPLAY_NAME_MAPPINGS as d_hist_stretch

# Auto White Balance
from .DonutAutoWhiteBalance import NODE_CLASS_MAPPINGS        as m_white_balance
from .DonutAutoWhiteBalance import NODE_DISPLAY_NAME_MAPPINGS as d_white_balance

# Sharpening Methods (USM, High Pass, Smart, Deconv, HiRaLoAm, CAS)
from .DonutSharpen import NODE_CLASS_MAPPINGS        as m_sharpen
from .DonutSharpen import NODE_DISPLAY_NAME_MAPPINGS as d_sharpen

# Prompt Injection (Style Prompts)
from .DonutPromptInjection import NODE_CLASS_MAPPINGS        as m_prompt_injection
from .DonutPromptInjection import NODE_DISPLAY_NAME_MAPPINGS as d_prompt_injection

# ZiT Conditioning Rebalance (input-side gain for Z-Image Turbo)
from .DonutZitConditioningRebalance import NODE_CLASS_MAPPINGS        as m_zit_rebalance
from .DonutZitConditioningRebalance import NODE_DISPLAY_NAME_MAPPINGS as d_zit_rebalance

# ZiT Layer-Blend Encode (multi-layer depth blend for Z-Image Turbo)
from .DonutZitLayerBlendEncode import NODE_CLASS_MAPPINGS        as m_zit_layerblend
from .DonutZitLayerBlendEncode import NODE_DISPLAY_NAME_MAPPINGS as d_zit_layerblend

# Image Adjust (unified tone/color/contrast/sharpen)
from .DonutImageAdjust import NODE_CLASS_MAPPINGS        as m_image_adjust
from .DonutImageAdjust import NODE_DISPLAY_NAME_MAPPINGS as d_image_adjust

# build globals
NODE_CLASS_MAPPINGS = {
    **m1, **m2, **m3, **m4, **m5,
    **m6, **m7,
    **m_lora,
    **m_reload,
    # **m_optuna,  # disabled - file not in repo
    **m_teacache,
    **m_calibration,
    **m_freq_analysis,
    **m_spectral_sharpener,
    **m_donut_sampler,
    **m_lora_civitai,
    **m_tiled_upscale,
    **m_zit_detailer,
    **m_merge_zit,
    **m_merge_zit_blocks,
    **m_model_save,
    **m_face_detailer,
    **m_universal_detailer,
    **m_prompt_receiver,
    **m_gamma,
    **m_auto_gamma,
    **m_hist_stretch,
    **m_white_balance,
    **m_sharpen,
    **m_prompt_injection,
    **m_zit_rebalance,
    **m_zit_layerblend,
    **m_image_adjust,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **d_lora,
    **d_reload,
    # **d_optuna,  # disabled - file not in repo
    **d_teacache,
    **d_calibration,
    **d_freq_analysis,
    **d_spectral_sharpener,
    **d_donut_sampler,
    **d_lora_civitai,
    **d_tiled_upscale,
    **d_zit_detailer,
    **d_merge_zit,
    **d_merge_zit_blocks,
    **d_model_save,
    **d_face_detailer,
    **d_universal_detailer,
    **d_prompt_receiver,
    **d_gamma,
    **d_auto_gamma,
    **d_hist_stretch,
    **d_white_balance,
    **d_sharpen,
    **d_prompt_injection,
    **d_zit_rebalance,
    **d_zit_layerblend,
    **d_detailer,
    **d_image_adjust,
}

# Nodes that were folded into multipurpose nodes. They stay registered so saved
# workflows keep loading, but ComfyUI's DEPRECATED flag hides them from the
# Add-Node menu. Labels keep a "(DEPRECATED)" suffix for any frontend that still
# lists deprecated nodes.
_DEPRECATED_DISPLAY = {
    "DonutModelSave":                  "Model Save (No Workflow) (DEPRECATED)",
    "DonutCheckpointSave":             "Checkpoint Save (No Workflow) (DEPRECATED)",
    "ModelMergeZITBlocks":             "Model Merge ZIT Blocks (DEPRECATED)",
    "Donut Simple Calibration":        "Donut Simple Calibration (DEPRECATED)",
    "Donut Sharpener (from reference)": "Donut Sharpener (from reference) (DEPRECATED)",
    "Donut Sharpener":                 "Donut Sharpener (DEPRECATED)",
    "DonutLoRACivitAIInfo":            "Donut LoRA CivitAI Info (DEPRECATED)",
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

# Display name for the combined filler (its module exports no display mapping)
NODE_DISPLAY_NAME_MAPPINGS["DonutFiller"] = "Donut Filler (Model + CLIP)"

# Web directory for custom JavaScript extensions
WEB_DIRECTORY = "./web"

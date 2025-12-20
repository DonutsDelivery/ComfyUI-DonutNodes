# Register server routes for settings API
from .shared import server_routes  # noqa: F401

# existing nodes' class mappings
from .DonutDetailer         import NODE_CLASS_MAPPINGS as m1
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
}

# Web directory for custom JavaScript extensions
WEB_DIRECTORY = "./web"

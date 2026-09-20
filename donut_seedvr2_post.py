"""Standalone SeedVR2 post-upscale stage.

Runs after the regular Donut/finishing upscale chain, not instead of it. The
regular hires-fix stages keep their Donut engine; this node receives their
result and refines it with the native SeedVR2 still-image pipeline. Model and
VAE imports stay behind execution, so older ComfyUI builds load and run the
normal workflow unchanged.
"""
from copy import deepcopy
from . import donut_seedvr2


class DonutSeedVR2Upscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "seedvr2_upscale_factor": ("FLOAT", {
                    "default": 2.0, "min": 1.0, "max": 8.0, "step": 0.5,
                    "tooltip": "Output size relative to the incoming image; the native pipeline returns even dimensions.",
                }),
                "resampling_method": (["lanczos", "nearest", "bilinear", "bicubic"], {"default": "lanczos"}),
            },
            "optional": {
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Off passes the incoming image through unchanged.",
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "seedvr2_model_name": donut_seedvr2.input_types()["seedvr2_model_name"],
                "seedvr2_vae_name": donut_seedvr2.input_types()["seedvr2_vae_name"],
                "seedvr2_steps": ("INT", {"default": 1, "min": 1, "max": 100, "lazy": True}),
                "seedvr2_denoise": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01, "lazy": True}),
                "seedvr2_color_correction": (["none", "lab", "wavelet", "adain"], {"default": "none", "lazy": True}),
                "seedvr2_vae_tile_size": donut_seedvr2.input_types()["seedvr2_vae_tile_size"],
                "seedvr2_vae_overlap": donut_seedvr2.input_types()["seedvr2_vae_overlap"],
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "post_upscale"
    CATEGORY = "donut/upscale"
    DESCRIPTION = ("Optional SeedVR2 refinement applied after the regular Donut "
                   "hires-fix upscale stages. Place it last in finishing; it "
                   "expects an already-upscaled image.")

    def check_lazy_status(self, image, enabled=True, **kwargs):
        if not enabled:
            return []
        required = donut_seedvr2.SHARED_INPUTS - {"seed"} - {"rescale_factor"} | {
            "seedvr2_model_name", "seedvr2_vae_name", "seedvr2_steps",
            "seedvr2_denoise", "seedvr2_color_correction", "seedvr2_vae_tile_size",
            "seedvr2_vae_overlap",
        }
        return [key for key, value in kwargs.items() if key in required and value is None]

    def post_upscale(self, image, seedvr2_upscale_factor=2.0, resampling_method="lanczos",
                     enabled=True, seed=0, **options):
        if not enabled:
            return (image,)
        unknown = set(options) - donut_seedvr2.DEFAULTS.keys()
        if unknown:
            raise TypeError("Unknown SeedVR2 settings: " + ", ".join(sorted(unknown)))
        result = donut_seedvr2.upscale(
            self, image, seed=seed,
            rescale_factor=seedvr2_upscale_factor,
            resampling_method=resampling_method, **options,
        )
        return (result,)


NODE_CLASS_MAPPINGS = {"DonutSeedVR2Upscale": DonutSeedVR2Upscale}
NODE_DISPLAY_NAME_MAPPINGS = {"DonutSeedVR2Upscale": "Donut SeedVR2 Post Upscale"}

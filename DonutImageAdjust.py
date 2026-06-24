"""
DonutImageAdjust - Unified image adjustment node.

Folds six IMAGE->IMAGE operations into a single node with an operation
selector. Each operation reuses the EXISTING implementation from its
original source file verbatim (no math is reimplemented here):

- auto_gamma       -> DonutAutoGamma.apply_auto_gamma
- gamma            -> DonutGammaCorrection.apply_gamma
- white_balance    -> DonutAutoWhiteBalance.apply_white_balance
- histogram_stretch-> DonutHistogramStretch.stretch_histogram
- local_contrast   -> DonutHiRaLoAm.apply_hiraloam
- cas              -> DonutCAS.apply_cas

The original six nodes remain registered (in their own files) as the
backwards-compatible aliases for already-saved workflows.

RETURN_TYPES = ("IMAGE", "FLOAT"). The FLOAT is auto_gamma's diagnostic
average gamma; for every other operation it is 1.0.
"""

from .DonutAutoGamma import DonutAutoGamma
from .DonutGammaCorrection import DonutGammaCorrection
from .DonutAutoWhiteBalance import DonutAutoWhiteBalance
from .DonutHistogramStretch import DonutHistogramStretch
from .DonutSharpen import DonutHiRaLoAm, DonutCAS


class DonutImageAdjust:
    """
    Unified image adjustment node folding six IMAGE->IMAGE operations.

    Choose the operation with the `operation` selector. The mode-specific
    widgets for each operation are namespaced by an operation prefix
    (ag_, gm_, wb_, hs_, lc_, cas_) and live in the optional block so the
    unused ones never block execution. Each widget keeps the original
    node's default.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "operation": ([
                    "auto_gamma",
                    "gamma",
                    "white_balance",
                    "histogram_stretch",
                    "local_contrast",
                    "cas",
                ], {
                    "default": "gamma",
                    "tooltip": "Which image adjustment operation to apply."
                }),
            },
            "optional": {
                # --- auto_gamma (DonutAutoGamma) ---
                "ag_method": (["simple_mean", "iagcwd", "gslf", "percentile"], {
                    "default": "gslf",
                    "tooltip": "Detection method: simple_mean (fastest), iagcwd (deviation-based), gslf (histogram-based, most sophisticated), percentile (robust to outliers)"
                }),
                "ag_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Correction strength (1.0 = full correction)"
                }),
                "ag_target": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 0.9,
                    "step": 0.05,
                    "tooltip": "Target value for simple_mean and percentile methods"
                }),
                "ag_percentile": ("FLOAT", {
                    "default": 50.0,
                    "min": 1.0,
                    "max": 99.0,
                    "step": 5.0,
                    "tooltip": "Which percentile to target (percentile method only)"
                }),
                "ag_direction": (["both", "brighten_only", "darken_only"], {
                    "default": "both",
                    "tooltip": "both: auto brighten/darken, brighten_only: only brighten dark images, darken_only: only darken bright images"
                }),

                # --- gamma (DonutGammaCorrection) ---
                "gm_gamma": ("FLOAT", {
                    "default": 0.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 1.0,
                    "tooltip": "Gamma adjustment (-100 to +100). Positive brightens shadows, negative darkens them."
                }),
                "gm_dither": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Dither amount to reduce banding (0 = off, 0.5 = subtle, 1.0 = 1 LSB worth of noise)"
                }),

                # --- white_balance (DonutAutoWhiteBalance) ---
                "wb_method": (["gray_world", "white_patch", "combined", "auto_levels", "photoshop"], {
                    "default": "gray_world",
                    "tooltip": "gray_world: shift means to gray. white_patch: scale by brightest. combined: both. auto_levels: per-channel stretch (Auto Tone). photoshop: per-channel + midtone snap (Auto Color)"
                }),
                "wb_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Correction strength (1.0 = full correction)"
                }),
                "wb_clip_percent": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Percent of pixels to clip from brightest/darkest (Photoshop default: 0.1%)"
                }),

                # --- histogram_stretch (DonutHistogramStretch) ---
                "hs_black_point": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 49.0,
                    "step": 0.5,
                    "tooltip": "Percentile for black point (0 = absolute min, 1+ = robust)"
                }),
                "hs_white_point": ("FLOAT", {
                    "default": 100.0,
                    "min": 51.0,
                    "max": 100.0,
                    "step": 0.5,
                    "tooltip": "Percentile for white point (100 = absolute max, 99- = robust)"
                }),
                "hs_mode": (["global", "per_channel", "luminance"], {
                    "default": "luminance",
                    "tooltip": "global: bounds from luminance, same stretch to RGB (preserves ratios). per_channel: stretch RGB independently (may shift colors). luminance: LAB L-only (preserves hue/saturation, matches LayerStyle)"
                }),

                # --- local_contrast (DonutHiRaLoAm) ---
                "lc_radius": ("FLOAT", {
                    "default": 20.0,
                    "min": 1.0,
                    "max": 200.0,
                    "step": 1.0,
                    "tooltip": "Large radius for local contrast enhancement"
                }),
                "lc_amount": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Keep low to avoid halos (0.1-0.3 typical)"
                }),
                "lc_protect_tones": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Protect shadows/highlights from clipping"
                }),
                "lc_fast_blur": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use separable blur (faster, identical quality)"
                }),

                # --- cas (DonutCAS) ---
                "cas_sharpness": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Sharpening strength"
                }),
                "cas_contrast": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Contrast adjustment (0 = neutral)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("image", "gamma")
    FUNCTION = "apply"
    CATEGORY = "donut/image"

    def apply(
        self, image, operation,
        # auto_gamma
        ag_method="gslf", ag_strength=1.0, ag_target=0.5, ag_percentile=50.0, ag_direction="both",
        # gamma
        gm_gamma=0.0, gm_dither=0.0,
        # white_balance
        wb_method="gray_world", wb_strength=1.0, wb_clip_percent=0.1,
        # histogram_stretch
        hs_black_point=0.0, hs_white_point=100.0, hs_mode="luminance",
        # local_contrast
        lc_radius=20.0, lc_amount=0.2, lc_protect_tones=0.0, lc_fast_blur=True,
        # cas
        cas_sharpness=0.5, cas_contrast=0.0,
    ):
        if operation == "auto_gamma":
            image_out, gamma_out = DonutAutoGamma().apply_auto_gamma(
                image, ag_method, ag_strength,
                target=ag_target, percentile=ag_percentile, direction=ag_direction,
            )
            return (image_out, gamma_out)

        if operation == "gamma":
            (image_out,) = DonutGammaCorrection().apply_gamma(image, gm_gamma, dither=gm_dither)
            return (image_out, 1.0)

        if operation == "white_balance":
            (image_out,) = DonutAutoWhiteBalance().apply_white_balance(
                image, wb_method, wb_strength, wb_clip_percent,
            )
            return (image_out, 1.0)

        if operation == "histogram_stretch":
            (image_out,) = DonutHistogramStretch().stretch_histogram(
                image, hs_black_point, hs_white_point, hs_mode,
            )
            return (image_out, 1.0)

        if operation == "local_contrast":
            (image_out,) = DonutHiRaLoAm().apply_hiraloam(
                image, lc_radius, lc_amount, lc_protect_tones, lc_fast_blur,
            )
            return (image_out, 1.0)

        if operation == "cas":
            (image_out,) = DonutCAS().apply_cas(image, cas_sharpness, cas_contrast)
            return (image_out, 1.0)

        return (image, 1.0)


NODE_CLASS_MAPPINGS = {
    "DonutImageAdjust": DonutImageAdjust,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutImageAdjust": "Donut Image Adjust",
}

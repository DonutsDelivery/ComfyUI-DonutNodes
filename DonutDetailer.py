import torch


def _run_detailer_patch(new_model, prefixes):
    """
    Shared patch pipeline (copied verbatim from the original DonutDetailer
    nodes). Given a cloned model and a {prefix: (weight_mult, bias_mult)} map,
    applies the multipliers to input_blocks.0.0 / out.0 / out.2.
    """
    # Get diffusion model for parameter access
    diffusion_model = new_model.get_model_object("diffusion_model")
    if diffusion_model is None:
        print("[DonutDetailer] Warning: Could not get diffusion_model")
        return (new_model,)

    with torch.no_grad():
        for name, param in diffusion_model.named_parameters():
            for prefix, (w_mult, b_mult) in prefixes.items():
                if prefix in name:
                    if ".weight" in name:
                        mult = w_mult
                    elif ".bias" in name:
                        mult = b_mult
                    else:
                        continue

                    # Skip if no change needed
                    if abs(mult - 1.0) < 1e-6:
                        continue

                    # Apply patch: to multiply by M, add original * (M-1)
                    patch_key = f"diffusion_model.{name}"
                    patch_strength = mult - 1.0
                    new_model.add_patches({patch_key: (param.data.clone(),)}, patch_strength)
                    break

    return (new_model,)


def _prefixes_scale_weight_bias(
    Scale_in, Weight_in, Bias_in,
    Scale_out0, Weight_out0, Bias_out0,
    Scale_out2, Weight_out2, Bias_out2,
):
    """D1 formula: Scale/Weight/Bias (copied verbatim from DonutDetailer)."""
    # Compute multipliers
    weight_in_mult = 1 - Scale_in * Weight_in
    bias_in_mult = 1 + Scale_in * Bias_in
    weight_out0_mult = 1 - Scale_out0 * Weight_out0
    bias_out0_mult = Scale_out0 * Bias_out0
    weight_out2_mult = 1 - Scale_out2 * Weight_out2
    bias_out2_mult = Scale_out2 * Bias_out2

    return {
        "input_blocks.0.0.": (weight_in_mult, bias_in_mult),
        "out.0.": (weight_out0_mult, bias_out0_mult),
        "out.2.": (weight_out2_mult, bias_out2_mult),
    }


def _prefixes_k_s1_s2(
    Multiplier_in, S1_in, S2_in,
    Multiplier_out0, S1_out0, S2_out0,
    Multiplier_out2, S1_out2, S2_out2,
):
    """D2 formula: K/S1/S2 (copied verbatim from DonutDetailer2)."""
    # Compute multipliers using the formula
    weight_in_mult = 1 - (Multiplier_in * S1_in * 0.01)
    bias_in_mult = 1 + (Multiplier_in * S2_in * 0.02)
    weight_out0_mult = 1 - (Multiplier_out0 * S1_out0 * 0.01)
    bias_out0_mult = 1 + (Multiplier_out0 * S2_out0 * 0.02)
    weight_out2_mult = 1 - (Multiplier_out2 * S1_out2 * 0.01)
    bias_out2_mult = 1 + (Multiplier_out2 * S2_out2 * 0.02)

    return {
        "input_blocks.0.0.": (weight_in_mult, bias_in_mult),
        "out.0.": (weight_out0_mult, bias_out0_mult),
        "out.2.": (weight_out2_mult, bias_out2_mult),
    }


def _prefixes_direct(
    Weight_in, Bias_in,
    Weight_out0, Bias_out0,
    Weight_out2, Bias_out2,
):
    """D4 formula: direct multipliers (copied verbatim from DonutDetailer4)."""
    return {
        "input_blocks.0.0.": (Weight_in, Bias_in),
        "out.0.": (Weight_out0, Bias_out0),
        "out.2.": (Weight_out2, Bias_out2),
    }


class DonutDetailerUnified:
    """
    Donut Detailer (Unified): single node merging the three detailer formulas
    (Scale/Weight/Bias, K/S1/S2, direct multipliers). They all target the same
    SDXL keys (input_blocks.0.0 / out.0 / out.2) and share one patch loop; only
    the per-group (weight_mult, bias_mult) derivation differs.
    """
    class_type = "MODEL"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "formula": (["scale_weight_bias", "k_s1_s2", "direct"],),
            },
            "optional": {
                # --- scale_weight_bias (D1) defaults ---
                "swb_Scale_in":    ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.001}),
                "swb_Weight_in":   ("FLOAT", {"default": 0.0, "min": -10.0,  "max": 10.0,  "step": 0.001}),
                "swb_Bias_in":     ("FLOAT", {"default": 0.0, "min": -10.0,  "max": 10.0,  "step": 0.001}),
                "swb_Scale_out0":  ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.001}),
                "swb_Weight_out0": ("FLOAT", {"default": 0.0, "min": -10.0,  "max": 10.0,  "step": 0.001}),
                "swb_Bias_out0":   ("FLOAT", {"default": 1.0, "min": -10.0,  "max": 10.0,  "step": 0.001}),
                "swb_Scale_out2":  ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.001}),
                "swb_Weight_out2": ("FLOAT", {"default": 0.0, "min": -10.0,  "max": 10.0,  "step": 0.001}),
                "swb_Bias_out2":   ("FLOAT", {"default": 1.0, "min": -10.0,  "max": 10.0,  "step": 0.001}),
                # --- k_s1_s2 (D2) defaults ---
                "ks_Multiplier_in":   ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "ks_S1_in":           ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "ks_S2_in":           ("FLOAT", {"default": 2.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "ks_Multiplier_out0": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "ks_S1_out0":         ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "ks_S2_out0":         ("FLOAT", {"default": 2.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "ks_Multiplier_out2": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "ks_S1_out2":         ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "ks_S2_out2":         ("FLOAT", {"default": 2.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                # --- direct (D4) defaults ---
                "dir_Weight_in":   ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                "dir_Bias_in":     ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                "dir_Weight_out0": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                "dir_Bias_out0":   ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                "dir_Weight_out2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                "dir_Bias_out2":   ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_patch"
    CATEGORY = "Model Patches"

    def apply_patch(
        self, model, formula,
        # scale_weight_bias (D1)
        swb_Scale_in=1.0, swb_Weight_in=0.0, swb_Bias_in=0.0,
        swb_Scale_out0=1.0, swb_Weight_out0=0.0, swb_Bias_out0=1.0,
        swb_Scale_out2=1.0, swb_Weight_out2=0.0, swb_Bias_out2=1.0,
        # k_s1_s2 (D2)
        ks_Multiplier_in=0.0, ks_S1_in=1.0, ks_S2_in=2.0,
        ks_Multiplier_out0=0.0, ks_S1_out0=1.0, ks_S2_out0=2.0,
        ks_Multiplier_out2=0.0, ks_S1_out2=1.0, ks_S2_out2=2.0,
        # direct (D4)
        dir_Weight_in=1.0, dir_Bias_in=1.0,
        dir_Weight_out0=1.0, dir_Bias_out0=1.0,
        dir_Weight_out2=1.0, dir_Bias_out2=1.0,
    ):
        # Clone using ComfyUI's method
        new_model = model.clone()

        if formula == "scale_weight_bias":
            prefixes = _prefixes_scale_weight_bias(
                swb_Scale_in, swb_Weight_in, swb_Bias_in,
                swb_Scale_out0, swb_Weight_out0, swb_Bias_out0,
                swb_Scale_out2, swb_Weight_out2, swb_Bias_out2,
            )
        elif formula == "k_s1_s2":
            prefixes = _prefixes_k_s1_s2(
                ks_Multiplier_in, ks_S1_in, ks_S2_in,
                ks_Multiplier_out0, ks_S1_out0, ks_S2_out0,
                ks_Multiplier_out2, ks_S1_out2, ks_S2_out2,
            )
        else:  # direct
            prefixes = _prefixes_direct(
                dir_Weight_in, dir_Bias_in,
                dir_Weight_out0, dir_Bias_out0,
                dir_Weight_out2, dir_Bias_out2,
            )

        return _run_detailer_patch(new_model, prefixes)


class DonutDetailer(DonutDetailerUnified):
    """
    Donut Detailer (alias): original Scale/Weight/Bias node. Keeps the exact
    original INPUT_TYPES and delegates to the shared engine.
    """
    class_type = "MODEL"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                # Input block parameters:
                "Scale_in":    ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.001}),
                "Weight_in":   ("FLOAT", {"default": 0.0, "min": -10.0,  "max": 10.0,  "step": 0.001}),
                "Bias_in":     ("FLOAT", {"default": 0.0, "min": -10.0,  "max": 10.0,  "step": 0.001}),
                # Output block 0 parameters:
                "Scale_out0":  ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.001}),
                "Weight_out0": ("FLOAT", {"default": 0.0, "min": -10.0,  "max": 10.0,  "step": 0.001}),
                "Bias_out0":   ("FLOAT", {"default": 1.0, "min": -10.0,  "max": 10.0,  "step": 0.001}),
                # Output block 2 parameters:
                "Scale_out2":  ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.001}),
                "Weight_out2": ("FLOAT", {"default": 0.0, "min": -10.0,  "max": 10.0,  "step": 0.001}),
                "Bias_out2":   ("FLOAT", {"default": 1.0, "min": -10.0,  "max": 10.0,  "step": 0.001}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_patch"
    CATEGORY = "Model Patches"

    def apply_patch(
        self, model,
        Scale_in, Weight_in, Bias_in,
        Scale_out0, Weight_out0, Bias_out0,
        Scale_out2, Weight_out2, Bias_out2
    ):
        new_model = model.clone()
        prefixes = _prefixes_scale_weight_bias(
            Scale_in, Weight_in, Bias_in,
            Scale_out0, Weight_out0, Bias_out0,
            Scale_out2, Weight_out2, Bias_out2,
        )
        return _run_detailer_patch(new_model, prefixes)


NODE_CLASS_MAPPINGS = {
    "DonutDetailerUnified": DonutDetailerUnified,
    "Donut Detailer": DonutDetailer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutDetailerUnified": "Donut Detailer (Unified)",
}

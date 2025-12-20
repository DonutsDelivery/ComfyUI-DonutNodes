import torch

class DonutDetailer2:
    """
    Donut Detailer 2: Group-specific adjustments using K/S1/S2 formula.
    Uses ComfyUI's patching system for proper model handling.
    """
    class_type = "MODEL"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                # Input block parameters:
                "Multiplier_in": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "S1_in":         ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "S2_in":         ("FLOAT", {"default": 2.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                # Output block 0 parameters:
                "Multiplier_out0": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "S1_out0":         ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "S2_out0":         ("FLOAT", {"default": 2.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                # Output block 2 parameters:
                "Multiplier_out2": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "S1_out2":         ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "S2_out2":         ("FLOAT", {"default": 2.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_patch"
    CATEGORY = "Model Patches"

    def apply_patch(self, model, Multiplier_in, S1_in, S2_in,
                    Multiplier_out0, S1_out0, S2_out0,
                    Multiplier_out2, S1_out2, S2_out2):
        """
        Applies group-specific adjustments using formulas:
          Weight multiplier = 1 - (K × S1 × 0.01)
          Bias multiplier   = 1 + (K × S2 × 0.02)
        """
        # Clone using ComfyUI's method
        new_model = model.clone()

        # Get diffusion model for parameter access
        diffusion_model = new_model.get_model_object("diffusion_model")
        if diffusion_model is None:
            print("[DonutDetailer2] Warning: Could not get diffusion_model")
            return (new_model,)

        # Compute multipliers using the formula
        weight_in_mult = 1 - (Multiplier_in * S1_in * 0.01)
        bias_in_mult = 1 + (Multiplier_in * S2_in * 0.02)
        weight_out0_mult = 1 - (Multiplier_out0 * S1_out0 * 0.01)
        bias_out0_mult = 1 + (Multiplier_out0 * S2_out0 * 0.02)
        weight_out2_mult = 1 - (Multiplier_out2 * S1_out2 * 0.01)
        bias_out2_mult = 1 + (Multiplier_out2 * S2_out2 * 0.02)

        # Prefixes for SDXL blocks
        prefixes = {
            "input_blocks.0.0.": (weight_in_mult, bias_in_mult),
            "out.0.": (weight_out0_mult, bias_out0_mult),
            "out.2.": (weight_out2_mult, bias_out2_mult),
        }

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

NODE_CLASS_MAPPINGS = {
    "Donut Detailer 2": DonutDetailer2,
}

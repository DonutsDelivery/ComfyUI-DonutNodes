import torch

class DonutDetailer:
    """
    Donut Detailer: Applies adjustments to input/output blocks in SDXL models.
    Uses ComfyUI's patching system for proper model handling.
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
        """
        Applies adjustments to three groups of parameters in an SDXL model.

        1. Input Block: Weight × (1 - Scale_in × Weight_in), Bias × (1 + Scale_in × Bias_in)
        2. Output Block 0: Weight × (1 - Scale_out0 × Weight_out0), Bias × (Scale_out0 × Bias_out0)
        3. Output Block 2: Weight × (1 - Scale_out2 × Weight_out2), Bias × (Scale_out2 × Bias_out2)
        """
        # Clone using ComfyUI's method
        new_model = model.clone()

        # Get diffusion model for parameter access
        diffusion_model = new_model.get_model_object("diffusion_model")
        if diffusion_model is None:
            print("[DonutDetailer] Warning: Could not get diffusion_model")
            return (new_model,)

        # Compute multipliers
        weight_in_mult = 1 - Scale_in * Weight_in
        bias_in_mult = 1 + Scale_in * Bias_in
        weight_out0_mult = 1 - Scale_out0 * Weight_out0
        bias_out0_mult = Scale_out0 * Bias_out0
        weight_out2_mult = 1 - Scale_out2 * Weight_out2
        bias_out2_mult = Scale_out2 * Bias_out2

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
    "Donut Detailer": DonutDetailer,
}

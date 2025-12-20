import torch

class DonutDetailer4:
    """
    Donut Detailer 4: Direct multipliers for input/output blocks.
    Uses ComfyUI's patching system for proper model handling.
    """
    class_type = "MODEL"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                # Multipliers for Input Block:
                "Weight_in": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                "Bias_in":   ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                # Multipliers for Output Block 0:
                "Weight_out0": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                "Bias_out0":   ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                # Multipliers for Output Block 2:
                "Weight_out2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                "Bias_out2":   ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_patch"
    CATEGORY = "Model Patches"

    def apply_patch(self, model, Weight_in, Bias_in, Weight_out0, Bias_out0, Weight_out2, Bias_out2):
        """
        Applies direct multipliers to three groups of parameters in an SDXL model.
        With default values (all 1.0), the node acts as a bypass.
        """
        # Clone using ComfyUI's method
        new_model = model.clone()

        # Get diffusion model for parameter access
        diffusion_model = new_model.get_model_object("diffusion_model")
        if diffusion_model is None:
            print("[DonutDetailer4] Warning: Could not get diffusion_model")
            return (new_model,)

        # Prefixes for SDXL blocks
        prefixes = {
            "input_blocks.0.0.": (Weight_in, Bias_in),
            "out.0.": (Weight_out0, Bias_out0),
            "out.2.": (Weight_out2, Bias_out2),
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
    "Donut Detailer 4": DonutDetailer4,
}

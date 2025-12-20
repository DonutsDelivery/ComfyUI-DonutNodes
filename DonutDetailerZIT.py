import torch

class DonutDetailerZIT:
    """
    Per-group weight/bias tuning for Z-Image Turbo (Lumina2) models.

    ZIT uses 30 transformer layers grouped by function:
      - Early (0-5): Translation/encoding - converts prompt to internal representation
      - Lower-Mid (6-14): Composition/layout - structural arrangement
      - Upper-Mid (15-23): Details/attributes - fine-grained features
      - Late (24-29): Refinement/aesthetics - style and quality

    Uses ComfyUI's patching system to apply multipliers without deepcopy issues.
    """
    class_type = "MODEL"

    # Layer ranges for each group
    LAYER_GROUPS = {
        "early": range(0, 6),      # 0-5
        "lowmid": range(6, 15),    # 6-14
        "upmid": range(15, 24),    # 15-23
        "late": range(24, 30),     # 24-29
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                # Early layers (0-5): Translation/encoding
                "early_weight": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001,
                    "tooltip": "Layers 0-5: Translation/encoding - converts prompt to internal representation"
                }),
                "early_bias": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001,
                    "tooltip": "Layers 0-5: Translation/encoding"
                }),
                # Lower-mid layers (6-14): Composition/layout
                "lowmid_weight": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001,
                    "tooltip": "Layers 6-14: Composition/layout - structural arrangement"
                }),
                "lowmid_bias": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001,
                    "tooltip": "Layers 6-14: Composition/layout"
                }),
                # Upper-mid layers (15-23): Details/attributes
                "upmid_weight": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001,
                    "tooltip": "Layers 15-23: Details/attributes - fine-grained features"
                }),
                "upmid_bias": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001,
                    "tooltip": "Layers 15-23: Details/attributes"
                }),
                # Late layers (24-29): Refinement/aesthetics
                "late_weight": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001,
                    "tooltip": "Layers 24-29: Refinement/aesthetics - style and quality"
                }),
                "late_bias": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001,
                    "tooltip": "Layers 24-29: Refinement/aesthetics"
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_patch"
    CATEGORY = "donut/Model Patches"

    def apply_patch(self, model, early_weight, early_bias, lowmid_weight, lowmid_bias,
                    upmid_weight, upmid_bias, late_weight, late_bias):
        # Clone using ComfyUI's proper method
        new_model = model.clone()

        # Build layer -> multipliers mapping
        layer_weights = {}
        layer_biases = {}
        for i in self.LAYER_GROUPS["early"]:
            layer_weights[i] = early_weight
            layer_biases[i] = early_bias
        for i in self.LAYER_GROUPS["lowmid"]:
            layer_weights[i] = lowmid_weight
            layer_biases[i] = lowmid_bias
        for i in self.LAYER_GROUPS["upmid"]:
            layer_weights[i] = upmid_weight
            layer_biases[i] = upmid_bias
        for i in self.LAYER_GROUPS["late"]:
            layer_weights[i] = late_weight
            layer_biases[i] = late_bias

        # Get the model's key patches - we need to access the underlying weights
        # to create patches that multiply them
        diffusion_model = new_model.get_model_object("diffusion_model")

        if diffusion_model is None:
            print("[DonutDetailerZIT] Warning: Could not get diffusion_model")
            return (new_model,)

        # Build patches for each layer
        # add_patches adds (patch_weight * strength) to original weight
        # To multiply by M: original + patch * strength = original * M
        # So: patch = original, strength = (M - 1)
        patches = {}

        with torch.no_grad():
            for name, param in diffusion_model.named_parameters():
                # Check if this parameter belongs to a transformer layer
                for i in range(30):
                    layer_prefix = f"layers.{i}."
                    if layer_prefix in name:
                        # Determine multiplier based on weight vs bias
                        if ".weight" in name:
                            mult = layer_weights.get(i, 1.0)
                        elif ".bias" in name:
                            mult = layer_biases.get(i, 1.0)
                        else:
                            mult = 1.0

                        # Skip if multiplier is 1.0 (no change needed)
                        if abs(mult - 1.0) < 1e-6:
                            break

                        # Create patch: to multiply by M, add original * (M-1)
                        # The key format for add_patches is "diffusion_model." + param name
                        patch_key = f"diffusion_model.{name}"
                        patch_strength = mult - 1.0

                        # Clone the parameter data for the patch
                        patch_tensor = param.data.clone()

                        # Store as tuple (patch_tensor,) - format expected by add_patches
                        patches[patch_key] = (patch_tensor,)

                        # Apply the patch with calculated strength
                        new_model.add_patches({patch_key: (patch_tensor,)}, patch_strength)
                        break

        return (new_model,)


NODE_CLASS_MAPPINGS = {
    "DonutDetailerZIT": DonutDetailerZIT,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutDetailerZIT": "Donut Detailer ZIT",
}

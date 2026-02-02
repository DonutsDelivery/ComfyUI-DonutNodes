class ModelMergeZITBlocks:
    """
    Model merge for Z-Image Turbo (Lumina2) models with per-layer ratio control.

    Like ModelMergeZIT but with individual control for all 30 transformer layers
    instead of bundled groups.

    Layer functions (approximate):
      - Layers 0-5: Translation/encoding - converts prompt to internal representation
      - Layers 6-14: Composition/layout - structural arrangement
      - Layers 15-23: Details/attributes - fine-grained features
      - Layers 24-29: Refinement/aesthetics - style and quality

    Non-layer components:
      - x_embedder: Patch embedding (converts image patches to tokens)
      - t_embedder: Timestep embedding
      - cap_embedder: Caption/text embedding
      - context_refiner: Context refinement
      - noise_refiner: Noise refinement
      - final_layer: Final output layer
      - norm_final: Final normalization
      - other: Remaining (pad tokens, etc.)

    Ratio 0.0 = use model1, Ratio 1.0 = use model2
    """

    NUM_LAYERS = 30

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "model1": ("MODEL",),
                "model2": ("MODEL",),
            }
        }

        # Add individual layer inputs (0-29)
        for i in range(cls.NUM_LAYERS):
            # Add tooltip describing which functional region this layer belongs to
            if i <= 5:
                region = "Translation/encoding"
            elif i <= 14:
                region = "Composition/layout"
            elif i <= 23:
                region = "Details/attributes"
            else:
                region = "Refinement/aesthetics"

            inputs["required"][f"layer_{i}"] = ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": f"Layer {i}: {region}"
            })

        # Add non-layer component inputs (individual control)
        inputs["required"]["x_embedder"] = ("FLOAT", {
            "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
            "tooltip": "Patch embedding (converts image patches to tokens)"
        })
        inputs["required"]["t_embedder"] = ("FLOAT", {
            "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
            "tooltip": "Timestep embedding"
        })
        inputs["required"]["cap_embedder"] = ("FLOAT", {
            "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
            "tooltip": "Caption/text embedding"
        })
        inputs["required"]["context_refiner"] = ("FLOAT", {
            "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
            "tooltip": "Context refinement"
        })
        inputs["required"]["noise_refiner"] = ("FLOAT", {
            "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
            "tooltip": "Noise refinement"
        })
        inputs["required"]["final_layer"] = ("FLOAT", {
            "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
            "tooltip": "Final output layer"
        })
        inputs["required"]["norm_final"] = ("FLOAT", {
            "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
            "tooltip": "Final normalization"
        })
        inputs["required"]["other"] = ("FLOAT", {
            "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
            "tooltip": "Remaining (pad tokens, etc.)"
        })

        return inputs

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "merge"
    CATEGORY = "advanced/model_merging"

    def merge(self, model1, model2, **kwargs):
        # Clone model1 as base
        m = model1.clone()

        # Get patches from model2's diffusion model
        kp = model2.get_key_patches("diffusion_model.")

        # Extract layer ratios from kwargs
        layer_ratios = {}
        for i in range(self.NUM_LAYERS):
            layer_ratios[i] = kwargs.get(f"layer_{i}", 1.0)

        # Extract non-layer component ratios
        x_embedder = kwargs.get("x_embedder", 1.0)
        t_embedder = kwargs.get("t_embedder", 1.0)
        cap_embedder = kwargs.get("cap_embedder", 1.0)
        context_refiner = kwargs.get("context_refiner", 1.0)
        noise_refiner = kwargs.get("noise_refiner", 1.0)
        final_layer = kwargs.get("final_layer", 1.0)
        norm_final = kwargs.get("norm_final", 1.0)
        other = kwargs.get("other", 1.0)

        for k in kp:
            ratio = other  # Default for unknown non-layer params

            # Check if this key belongs to a specific layer
            k_model = k[len("diffusion_model."):] if k.startswith("diffusion_model.") else k

            # First check for layer membership
            is_layer = False
            for i in range(self.NUM_LAYERS):
                layer_prefix = f"layers.{i}."
                if k_model.startswith(layer_prefix):
                    ratio = layer_ratios.get(i, other)
                    is_layer = True
                    break

            # If not a layer, check for known non-layer components
            if not is_layer:
                key_prefix = k_model.split('.')[0]

                if key_prefix == "x_embedder":
                    ratio = x_embedder
                elif key_prefix == "t_embedder":
                    ratio = t_embedder
                elif key_prefix == "cap_embedder":
                    ratio = cap_embedder
                elif key_prefix == "context_refiner":
                    ratio = context_refiner
                elif key_prefix == "noise_refiner":
                    ratio = noise_refiner
                elif key_prefix == "final_layer":
                    ratio = final_layer
                elif key_prefix == "norm_final":
                    ratio = norm_final
                # else: ratio stays as 'other'

            # Skip if ratio is 0 (keep model1 entirely for this key)
            if ratio == 0.0:
                continue

            # Apply merge: add_patches(patches, strength_patch, strength_model)
            # ratio=0 -> keep model1, ratio=1 -> use model2
            m.add_patches({k: kp[k]}, ratio, 1.0 - ratio)

        return (m,)


NODE_CLASS_MAPPINGS = {
    "ModelMergeZITBlocks": ModelMergeZITBlocks,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelMergeZITBlocks": "Model Merge ZIT Blocks",
}

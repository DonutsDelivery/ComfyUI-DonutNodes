class ModelMergeZIT:
    """
    Model merge for Z-Image Turbo (Lumina2) models with per-group ratio control.

    Works like ModelMergeBlocks but for ZIT architecture which uses 30 transformer
    layers instead of UNet blocks.

    Layer groups (matching DonutLoRAStack presets):
      - Early (0-5): Translation/encoding - converts prompt to internal representation
      - Lower-Mid (6-14): Composition/layout - structural arrangement
      - Upper-Mid (15-23): Details/attributes - fine-grained features
      - Late (24-29): Refinement/aesthetics - style and quality

    Non-layer components:
      - x_embedder: Patch embedding (converts image patches to tokens)
      - t_embedder: Timestep embedding
      - cap_embedder: Caption/text embedding
      - final_layer: Final normalization and projection
      - other: Any remaining components

    Ratio 0.0 = use model1, Ratio 1.0 = use model2
    """

    # Layer ranges for each group
    LAYER_GROUPS = {
        "early": range(0, 6),      # 0-5
        "lowmid": range(6, 15),    # 6-14
        "upmid": range(15, 24),    # 15-23
        "late": range(24, 30),     # 24-29
    }

    # Non-layer component prefixes
    NON_LAYER_COMPONENTS = ["x_embedder", "t_embedder", "cap_embedder", "final_layer"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model1": ("MODEL",),
                "model2": ("MODEL",),
                "early": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Layers 0-5: Translation/encoding"
                }),
                "lowmid": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Layers 6-14: Composition/layout"
                }),
                "upmid": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Layers 15-23: Details/attributes"
                }),
                "late": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Layers 24-29: Refinement/aesthetics"
                }),
                "x_embedder": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Patch embedding (converts image patches to tokens)"
                }),
                "t_embedder": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Timestep embedding"
                }),
                "cap_embedder": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Caption/text embedding"
                }),
                "final_layer": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Final normalization and projection"
                }),
                "other": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Any remaining non-layer parameters"
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "merge"
    CATEGORY = "advanced/model_merging"

    def merge(self, model1, model2, early, lowmid, upmid, late,
              x_embedder, t_embedder, cap_embedder, final_layer, other):
        # Clone model1 as base
        m = model1.clone()

        # Get patches from model2's diffusion model
        kp = model2.get_key_patches("diffusion_model.")

        print(f"[ModelMergeZIT] Got {len(kp)} patches from model2")
        print(f"[ModelMergeZIT] Ratios: early={early}, lowmid={lowmid}, upmid={upmid}, late={late}")
        print(f"[ModelMergeZIT] x_embedder={x_embedder}, t_embedder={t_embedder}, cap_embedder={cap_embedder}, final_layer={final_layer}, other={other}")

        # Build layer -> ratio mapping
        layer_ratios = {}
        for i in self.LAYER_GROUPS["early"]:
            layer_ratios[i] = early
        for i in self.LAYER_GROUPS["lowmid"]:
            layer_ratios[i] = lowmid
        for i in self.LAYER_GROUPS["upmid"]:
            layer_ratios[i] = upmid
        for i in self.LAYER_GROUPS["late"]:
            layer_ratios[i] = late

        # Non-layer component ratios
        component_ratios = {
            "x_embedder": x_embedder,
            "t_embedder": t_embedder,
            "cap_embedder": cap_embedder,
            "final_layer": final_layer,
        }

        applied_count = 0
        skipped_count = 0
        non_layer_keys = []

        for k in kp:
            ratio = other  # Default for unknown non-layer params

            # Check if this key belongs to a specific layer
            k_model = k[len("diffusion_model."):] if k.startswith("diffusion_model.") else k

            # First check for layer membership
            is_layer = False
            for i in range(30):
                layer_prefix = f"layers.{i}."
                if k_model.startswith(layer_prefix):
                    ratio = layer_ratios.get(i, other)
                    is_layer = True
                    break

            # If not a layer, check for known non-layer components
            if not is_layer:
                non_layer_keys.append(k_model)
                for component, comp_ratio in component_ratios.items():
                    if k_model.startswith(f"{component}.") or k_model.startswith(f"{component}_"):
                        ratio = comp_ratio
                        break

            # Skip if ratio is 0 (keep model1 entirely for this key)
            if ratio == 0.0:
                skipped_count += 1
                continue

            # Apply merge: add_patches(patches, strength_patch, strength_model)
            # strength_patch = how much of model2, strength_model = how much of model1
            # ratio=0 -> keep model1, ratio=1 -> use model2
            m.add_patches({k: kp[k]}, ratio, 1.0 - ratio)
            applied_count += 1

        print(f"[ModelMergeZIT] Applied {applied_count} patches, skipped {skipped_count}")
        if non_layer_keys:
            unique_prefixes = set(k.split('.')[0] for k in non_layer_keys)
            print(f"[ModelMergeZIT] Non-layer key prefixes: {sorted(unique_prefixes)}")

        return (m,)


NODE_CLASS_MAPPINGS = {
    "ModelMergeZIT": ModelMergeZIT,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelMergeZIT": "Model Merge ZIT",
}

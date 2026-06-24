class ModelMergeZIT:
    """
    Unified model merge for Z-Image Turbo (Lumina2) models.

    A single "granularity" selector switches between two merge modes that were
    previously two separate nodes:

      - "grouped" (default): per-group ratio control (the original ModelMergeZIT
        behavior). Transformer layers are bundled into 4 ranges and the refiner
        and final components are bundled together.
      - "blocks": per-layer ratio control (the original ModelMergeZITBlocks
        behavior). Every one of the 30 transformer layers and every non-layer
        component is controlled individually.

    Works like ModelMergeBlocks but for ZIT architecture which uses 30
    transformer layers instead of UNet blocks.

    Layer groups (matching DonutLoRAStack presets):
      - Early (0-5): Translation/encoding - converts prompt to internal representation
      - Lower-Mid (6-14): Composition/layout - structural arrangement
      - Upper-Mid (15-23): Details/attributes - fine-grained features
      - Late (24-29): Refinement/aesthetics - style and quality

    Non-layer components:
      - x_embedder: Patch embedding (converts image patches to tokens)
      - t_embedder: Timestep embedding
      - cap_embedder: Caption/text embedding
      - refiners: context_refiner + noise_refiner (grouped mode)
      - final: final_layer + norm_final (grouped mode)
      - other: Remaining (pad tokens, etc.)

    Ratio 0.0 = use model1, Ratio 1.0 = use model2
    """

    NUM_LAYERS = 30

    # Layer ranges for each group
    LAYER_GROUPS = {
        "early": range(0, 6),      # 0-5
        "lowmid": range(6, 15),    # 6-14
        "upmid": range(15, 24),    # 15-23
        "late": range(24, 30),     # 24-29
    }

    # Non-layer component prefixes (grouped)
    EMBEDDER_COMPONENTS = ["x_embedder", "t_embedder", "cap_embedder"]
    REFINER_COMPONENTS = ["context_refiner", "noise_refiner"]
    FINAL_COMPONENTS = ["final_layer", "norm_final"]

    @classmethod
    def INPUT_TYPES(cls):
        # Back-compat: the original grouped ModelMergeZIT widgets stay in
        # "required" in their ORIGINAL order so already-saved grouped workflows
        # deserialize byte-identically (ComfyUI lays out required widgets first,
        # then optional, in dict order). The selector and all blocks-only widgets
        # are APPENDED afterwards in "optional", every default 1.0, so the active
        # mode reproduces the original node's behavior and unused widgets never
        # block execution.
        required = {
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
            "refiners": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "context_refiner + noise_refiner"
            }),
            "final": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "final_layer + norm_final"
            }),
            "other": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "Remaining (pad tokens, etc.)"
            }),
        }

        optional = {}

        # Selector (appended after the original grouped widgets)
        optional["granularity"] = (["grouped", "blocks"], {
            "default": "grouped",
            "tooltip": "grouped: per-group ratios (original ModelMergeZIT). "
                       "blocks: per-layer ratios (original ModelMergeZITBlocks).",
        })

        # --- Blocks-mode per-layer widgets (layer_0 .. layer_29) ---
        for i in range(cls.NUM_LAYERS):
            if i <= 5:
                region = "Translation/encoding"
            elif i <= 14:
                region = "Composition/layout"
            elif i <= 23:
                region = "Details/attributes"
            else:
                region = "Refinement/aesthetics"

            optional[f"layer_{i}"] = ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": f"Layer {i}: {region}"
            })

        # --- Blocks-mode individual non-layer components ---
        optional["context_refiner"] = ("FLOAT", {
            "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
            "tooltip": "Context refinement"
        })
        optional["noise_refiner"] = ("FLOAT", {
            "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
            "tooltip": "Noise refinement"
        })
        optional["final_layer"] = ("FLOAT", {
            "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
            "tooltip": "Final output layer"
        })
        optional["norm_final"] = ("FLOAT", {
            "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
            "tooltip": "Final normalization"
        })

        return {
            "required": required,
            "optional": optional,
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "merge"
    CATEGORY = "advanced/model_merging"

    def _resolve_ratios(self, granularity, kwargs):
        """Build a key-prefix-independent layer->ratio map and component getters."""
        if granularity == "blocks":
            layer_ratios = {}
            for i in range(self.NUM_LAYERS):
                layer_ratios[i] = kwargs.get(f"layer_{i}", 1.0)

            x_embedder = kwargs.get("x_embedder", 1.0)
            t_embedder = kwargs.get("t_embedder", 1.0)
            cap_embedder = kwargs.get("cap_embedder", 1.0)
            context_refiner = kwargs.get("context_refiner", 1.0)
            noise_refiner = kwargs.get("noise_refiner", 1.0)
            final_layer = kwargs.get("final_layer", 1.0)
            norm_final = kwargs.get("norm_final", 1.0)
            other = kwargs.get("other", 1.0)
        else:
            # grouped
            early = kwargs.get("early", 1.0)
            lowmid = kwargs.get("lowmid", 1.0)
            upmid = kwargs.get("upmid", 1.0)
            late = kwargs.get("late", 1.0)
            refiners = kwargs.get("refiners", 1.0)
            final = kwargs.get("final", 1.0)

            layer_ratios = {}
            for i in self.LAYER_GROUPS["early"]:
                layer_ratios[i] = early
            for i in self.LAYER_GROUPS["lowmid"]:
                layer_ratios[i] = lowmid
            for i in self.LAYER_GROUPS["upmid"]:
                layer_ratios[i] = upmid
            for i in self.LAYER_GROUPS["late"]:
                layer_ratios[i] = late

            x_embedder = kwargs.get("x_embedder", 1.0)
            t_embedder = kwargs.get("t_embedder", 1.0)
            cap_embedder = kwargs.get("cap_embedder", 1.0)
            context_refiner = refiners
            noise_refiner = refiners
            final_layer = final
            norm_final = final
            other = kwargs.get("other", 1.0)

        return {
            "layer_ratios": layer_ratios,
            "x_embedder": x_embedder,
            "t_embedder": t_embedder,
            "cap_embedder": cap_embedder,
            "context_refiner": context_refiner,
            "noise_refiner": noise_refiner,
            "final_layer": final_layer,
            "norm_final": norm_final,
            "other": other,
        }

    def merge(self, model1, model2, granularity="grouped", **kwargs):
        # Clone model1 as base
        m = model1.clone()

        # Get patches from model2's diffusion model
        kp = model2.get_key_patches("diffusion_model.")

        r = self._resolve_ratios(granularity, kwargs)
        layer_ratios = r["layer_ratios"]
        other = r["other"]

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
                    ratio = r["x_embedder"]
                elif key_prefix == "t_embedder":
                    ratio = r["t_embedder"]
                elif key_prefix == "cap_embedder":
                    ratio = r["cap_embedder"]
                elif key_prefix == "context_refiner":
                    ratio = r["context_refiner"]
                elif key_prefix == "noise_refiner":
                    ratio = r["noise_refiner"]
                elif key_prefix == "final_layer":
                    ratio = r["final_layer"]
                elif key_prefix == "norm_final":
                    ratio = r["norm_final"]
                # else: ratio stays as 'other'

            # Skip if ratio is 0 (keep model1 entirely for this key)
            if ratio == 0.0:
                continue

            # Apply merge: add_patches(patches, strength_patch, strength_model)
            # strength_patch = how much of model2, strength_model = how much of model1
            # ratio=0 -> keep model1, ratio=1 -> use model2
            m.add_patches({k: kp[k]}, ratio, 1.0 - ratio)

        return (m,)


NODE_CLASS_MAPPINGS = {
    "ModelMergeZIT": ModelMergeZIT,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelMergeZIT": "Model Merge ZIT",
}

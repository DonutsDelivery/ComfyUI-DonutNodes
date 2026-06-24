from .ModelMergeZIT import ModelMergeZIT


class ModelMergeZITBlocks(ModelMergeZIT):
    """
    Thin back-compat alias of the unified ModelMergeZIT engine, pinned to the
    "blocks" granularity (per-layer ratio control).

    Keeps its ORIGINAL 38-widget INPUT_TYPES (30 per-layer widgets + 8 non-layer
    components, all in "required", exact original order) so already-saved
    ModelMergeZITBlocks workflows deserialize byte-identically. All merge logic
    lives in the shared engine; this subclass only forces granularity="blocks".

    Like ModelMergeZIT (grouped) but with individual control for all 30
    transformer layers instead of bundled groups.

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
        # Pin granularity to "blocks"; delegate to the shared engine.
        kwargs.pop("granularity", None)
        return super().merge(model1, model2, granularity="blocks", **kwargs)


NODE_CLASS_MAPPINGS = {
    "ModelMergeZITBlocks": ModelMergeZITBlocks,
}

# Deprecated: the unified "ModelMergeZIT" node (granularity="blocks") replaces
# this in the menu. Keep the class registered above so saved workflows still
# load, but drop its display name so it no longer appears in the add-node menu.
NODE_DISPLAY_NAME_MAPPINGS = {}

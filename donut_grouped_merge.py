"""Replace ratio fan-out wiring with optional grouped controls; retain all sliders."""
from copy import deepcopy
from .DonutModelMergeKrea2 import NODE_CLASS_MAPPINGS as _nodes
_Base = _nodes["DonutModelMergeKrea2"]


class DonutModelMergeKrea2Grouped(_Base):
    @classmethod
    def INPUT_TYPES(cls):
        result = deepcopy(_Base.INPUT_TYPES())
        result.setdefault("optional", {}).update({
            "ratio_mode": (["Per block", "Grouped"], {"default": "Per block"}),
            "body_ratio": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "fusion_ratio": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        })
        return result
    FUNCTION = "merge_grouped"

    def merge_grouped(self, ratio_mode="Per block", body_ratio=1.0, fusion_ratio=1.0, **kwargs):
        if ratio_mode not in ("Per block", "Grouped"):
            raise ValueError(f"Unknown ratio mode: {ratio_mode}")
        if ratio_mode == "Grouped":
            if not 0 <= body_ratio <= 1 or not 0 <= fusion_ratio <= 1:
                raise ValueError("Grouped merge ratios must be between 0 and 1")
            for name in ("first.", "last.", *(f"blocks.{i}." for i in range(28))):
                kwargs[name] = body_ratio
            for name in ("txtfusion.layerwise_blocks.0.", "txtfusion.layerwise_blocks.1.", "txtfusion.projector.",
                         "txtfusion.refiner_blocks.0.", "txtfusion.refiner_blocks.1."):
                kwargs[name] = fusion_ratio
        return getattr(_Base, _Base.FUNCTION)(self, **kwargs)


NODE_CLASS_MAPPINGS = {"DonutModelMergeKrea2": DonutModelMergeKrea2Grouped}
NODE_DISPLAY_NAME_MAPPINGS = {"DonutModelMergeKrea2": "Donut Model Merge Krea2"}

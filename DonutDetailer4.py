from .DonutDetailer import (
    DonutDetailerUnified,
    _prefixes_direct,
    _run_detailer_patch,
)


class DonutDetailer4(DonutDetailerUnified):
    """
    Donut Detailer 4 (alias): original direct-multiplier node. Keeps the exact
    original INPUT_TYPES and delegates to the shared engine.
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
        new_model = model.clone()
        prefixes = _prefixes_direct(
            Weight_in, Bias_in,
            Weight_out0, Bias_out0,
            Weight_out2, Bias_out2,
        )
        return _run_detailer_patch(new_model, prefixes)


NODE_CLASS_MAPPINGS = {
    "Donut Detailer 4": DonutDetailer4,
}

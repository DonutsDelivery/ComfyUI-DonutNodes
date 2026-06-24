from .DonutDetailer import (
    DonutDetailerUnified,
    _prefixes_k_s1_s2,
    _run_detailer_patch,
)


class DonutDetailer2(DonutDetailerUnified):
    """
    Donut Detailer 2 (alias): original K/S1/S2 formula node. Keeps the exact
    original INPUT_TYPES and delegates to the shared engine.
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
        new_model = model.clone()
        prefixes = _prefixes_k_s1_s2(
            Multiplier_in, S1_in, S2_in,
            Multiplier_out0, S1_out0, S2_out0,
            Multiplier_out2, S1_out2, S2_out2,
        )
        return _run_detailer_patch(new_model, prefixes)


NODE_CLASS_MAPPINGS = {
    "Donut Detailer 2": DonutDetailer2,
}

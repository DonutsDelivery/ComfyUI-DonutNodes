import torch

class DonutDetailerXLBlocks:
    """
    Per-block weight/bias tuning for SDXL models.
    Uses ComfyUI's patching system for proper model handling.
    """
    class_type = "MODEL"

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "model": ("MODEL",),
        }
        # Define block groups: (prefix, count)
        groups = [
            ("input_blocks", 9),
            ("middle_block", 3),
            ("output_blocks", 9),
            ("out", 1),
        ]
        for prefix, count in groups:
            for i in range(count):
                name = prefix if prefix == "out" else f"{prefix}_{i}"
                required[f"{name}_weight"] = ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001
                })
                required[f"{name}_bias"] = ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001
                })
        return {"required": required}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_patch"
    CATEGORY = "Model Patches"

    def apply_patch(self, model, **kwargs):
        # Clone using ComfyUI's method
        new_model = model.clone()

        # Get diffusion model for parameter access
        diffusion_model = new_model.get_model_object("diffusion_model")
        if diffusion_model is None:
            print("[DonutDetailerXLBlocks] Warning: Could not get diffusion_model")
            return (new_model,)

        with torch.no_grad():
            for name, param in diffusion_model.named_parameters():
                # input_blocks.0 – input_blocks.8
                for i in range(9):
                    pfx = f"input_blocks.{i}."
                    key = f"input_blocks_{i}"
                    if pfx in name:
                        if ".weight" in name:
                            mult = kwargs.get(f"{key}_weight", 1.0)
                        elif ".bias" in name:
                            mult = kwargs.get(f"{key}_bias", 1.0)
                        else:
                            continue
                        if abs(mult - 1.0) >= 1e-6:
                            patch_key = f"diffusion_model.{name}"
                            new_model.add_patches({patch_key: (param.data.clone(),)}, mult - 1.0)
                        break

                # middle_block.0 – middle_block.2
                for i in range(3):
                    pfx = f"middle_block.{i}."
                    key = f"middle_block_{i}"
                    if pfx in name:
                        if ".weight" in name:
                            mult = kwargs.get(f"{key}_weight", 1.0)
                        elif ".bias" in name:
                            mult = kwargs.get(f"{key}_bias", 1.0)
                        else:
                            continue
                        if abs(mult - 1.0) >= 1e-6:
                            patch_key = f"diffusion_model.{name}"
                            new_model.add_patches({patch_key: (param.data.clone(),)}, mult - 1.0)
                        break

                # output_blocks.0 – output_blocks.8
                for i in range(9):
                    pfx = f"output_blocks.{i}."
                    key = f"output_blocks_{i}"
                    if pfx in name:
                        if ".weight" in name:
                            mult = kwargs.get(f"{key}_weight", 1.0)
                        elif ".bias" in name:
                            mult = kwargs.get(f"{key}_bias", 1.0)
                        else:
                            continue
                        if abs(mult - 1.0) >= 1e-6:
                            patch_key = f"diffusion_model.{name}"
                            new_model.add_patches({patch_key: (param.data.clone(),)}, mult - 1.0)
                        break

                # the final out.* layer
                if "out." in name and "output_blocks" not in name:
                    key = "out"
                    if ".weight" in name:
                        mult = kwargs.get(f"{key}_weight", 1.0)
                    elif ".bias" in name:
                        mult = kwargs.get(f"{key}_bias", 1.0)
                    else:
                        continue
                    if abs(mult - 1.0) >= 1e-6:
                        patch_key = f"diffusion_model.{name}"
                        new_model.add_patches({patch_key: (param.data.clone(),)}, mult - 1.0)

        return (new_model,)


NODE_CLASS_MAPPINGS = {
    "Donut Detailer XL Blocks": DonutDetailerXLBlocks,
}

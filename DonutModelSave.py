import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_management
import torch
import os


DTYPE_OPTIONS = ["fp8_e4m3fn", "fp8_e5m2", "fp16", "bf16", "fp32"]

DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp8_e4m3fn": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}


def convert_state_dict_dtype_inplace(sd, dtype):
    """Convert all float tensors in state dict to target dtype in-place to save memory."""
    target_dtype = DTYPE_MAP.get(dtype, torch.float16)
    keys = list(sd.keys())
    for k in keys:
        v = sd[k]
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            # Move to CPU first, then convert dtype to minimize GPU memory
            sd[k] = v.cpu().to(target_dtype)
            del v
    # Clear GPU cache after conversion
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return sd


def get_patched_state_dict(model_patcher, prefix="diffusion_model."):
    """
    Get state dict with all patches (merges, LoRAs, etc.) properly applied.
    This handles model merges that store patches in the ModelPatcher.
    """
    # Get the base model's state dict
    base_sd = model_patcher.model.diffusion_model.state_dict()

    # Get all patches for the diffusion model
    patches = model_patcher.get_key_patches(prefix)

    # Build the patched state dict
    patched_sd = {}
    for key in base_sd:
        full_key = prefix + key
        weight = base_sd[key]

        if full_key in patches:
            # Apply patches using the model's calculate_weight method
            try:
                patched_weight = model_patcher.calculate_weight(patches[full_key], weight, full_key)
                patched_sd[key] = patched_weight
            except Exception as e:
                print(f"[DonutModelSave] Warning: Failed to apply patch for {key}: {e}")
                patched_sd[key] = weight
        else:
            patched_sd[key] = weight

    return patched_sd


class DonutModelSave:
    """
    Save diffusion model without embedding workflow metadata.
    Supports saving in different precisions: fp8, fp16, bf16, fp32.
    """

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "filename_prefix": ("STRING", {"default": "diffusion_models/ComfyUI"}),
                "dtype": (DTYPE_OPTIONS, {"default": "fp8_e4m3fn"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "advanced/model_merging"

    def save(self, model, filename_prefix, dtype="fp8_e4m3fn"):
        full_output_folder, filename, counter, subfolder, filename_prefix = \
            folder_paths.get_save_image_path(filename_prefix, self.output_dir)

        # Load model to GPU (needed for weight calculations)
        comfy.model_management.load_models_gpu([model])

        # Get state dict with patches (merges, LoRAs) properly applied
        sd = get_patched_state_dict(model)

        # Unload model from GPU to free VRAM before conversion
        comfy.model_management.unload_all_models()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Convert to target dtype in-place (also moves to CPU)
        sd = convert_state_dict_dtype_inplace(sd, dtype)

        # Ensure contiguous
        for k in sd:
            t = sd[k]
            if isinstance(t, torch.Tensor) and not t.is_contiguous():
                sd[k] = t.contiguous()

        # Save without workflow metadata
        output_path = os.path.join(full_output_folder, f"{filename}_{counter:05}_.safetensors")
        comfy.utils.save_torch_file(sd, output_path, metadata={})
        print(f"[DonutModelSave] Saved {dtype} model to {output_path}")

        return {}


class DonutCheckpointSave:
    """
    Save full checkpoint (model + clip + vae) without embedding workflow metadata.
    Supports saving in different precisions: fp8, fp16, bf16, fp32.
    """

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "filename_prefix": ("STRING", {"default": "checkpoints/ComfyUI"}),
                "dtype": (DTYPE_OPTIONS, {"default": "fp8_e4m3fn"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "advanced/model_merging"

    def save(self, model, clip, vae, filename_prefix, dtype="fp8_e4m3fn"):
        full_output_folder, filename, counter, subfolder, filename_prefix = \
            folder_paths.get_save_image_path(filename_prefix, self.output_dir)

        # Load models to GPU (needed for weight calculations)
        load_models = [model]
        if clip is not None:
            load_models.append(clip.load_model())
        comfy.model_management.load_models_gpu(load_models)

        # Get patched diffusion model state dict
        diffusion_sd = get_patched_state_dict(model)

        # Add "model.diffusion_model." prefix for checkpoint format
        sd = {}
        for k, v in diffusion_sd.items():
            sd["model.diffusion_model." + k] = v

        # Add CLIP weights
        if clip is not None:
            clip_sd = clip.get_sd()
            if clip_sd:
                sd.update(clip_sd)

        # Add VAE weights
        if vae is not None:
            vae_sd = vae.get_sd()
            if vae_sd:
                sd.update(vae_sd)

        # Unload models from GPU to free VRAM before conversion
        comfy.model_management.unload_all_models()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Convert to target dtype in-place (also moves to CPU)
        sd = convert_state_dict_dtype_inplace(sd, dtype)

        # Ensure contiguous
        for k in sd:
            t = sd[k]
            if isinstance(t, torch.Tensor) and not t.is_contiguous():
                sd[k] = t.contiguous()

        # Save without workflow metadata
        output_path = os.path.join(full_output_folder, f"{filename}_{counter:05}_.safetensors")
        comfy.utils.save_torch_file(sd, output_path, metadata={})
        print(f"[DonutCheckpointSave] Saved {dtype} checkpoint to {output_path}")

        return {}


NODE_CLASS_MAPPINGS = {
    "DonutModelSave": DonutModelSave,
    "DonutCheckpointSave": DonutCheckpointSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutModelSave": "Model Save (No Workflow)",
    "DonutCheckpointSave": "Checkpoint Save (No Workflow)",
}

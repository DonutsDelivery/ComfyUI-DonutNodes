"""
DonutSave / DonutModelSave / DonutCheckpointSave

No-workflow save nodes with explicit dtype control.

DonutModelSave is the dedicated diffusion-model-only saver: it accepts only MODEL,
never CLIP/VAE, and writes under diffusion_models/ by default. This is useful when
ComfyUI loaded a model in fp8 but the desired baked LoRA/merge output should be
saved as bf16/fp16/fp32 instead.

The state-dict construction path follows comfy.sd.save_checkpoint:
load_models_gpu(...) -> ModelPatcher.state_dict_for_saving(...). This routes
through ComfyUI's LazyCastingParam machinery, so patches such as LoRAs and model
merges are materialized into the saved weights while still supporting low-VRAM
partial loading.
"""

import os

import folder_paths
import torch

import comfy.model_base
import comfy.model_management
import comfy.model_sampling
import comfy.sd
import comfy.utils
from comfy.cli_args import args

try:
    from .model_lifecycle import offload_models
except ImportError:
    from model_lifecycle import offload_models


DTYPE_OPTIONS = ["original", "bf16", "fp16", "fp32", "fp8_e4m3fn", "fp8_e5m2"]

DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp8_e4m3fn": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}


def _build_modelspec_metadata(model, filename, counter):
    """
    Replicate the metadata block that comfy_extras.nodes_model_merging.save_checkpoint
    builds, MINUS the workflow fields (prompt / extra_pnginfo).
    """
    metadata = {}
    extra_keys = {}

    enable_modelspec = True
    if isinstance(model.model, comfy.model_base.SDXL):
        if isinstance(model.model, comfy.model_base.SDXL_instructpix2pix):
            metadata["modelspec.architecture"] = "stable-diffusion-xl-v1-edit"
        else:
            metadata["modelspec.architecture"] = "stable-diffusion-xl-v1-base"
    elif isinstance(model.model, comfy.model_base.SDXLRefiner):
        metadata["modelspec.architecture"] = "stable-diffusion-xl-v1-refiner"
    elif isinstance(model.model, comfy.model_base.SVD_img2vid):
        metadata["modelspec.architecture"] = "stable-video-diffusion-img2vid-v1"
    elif isinstance(model.model, comfy.model_base.SD3):
        metadata["modelspec.architecture"] = "stable-diffusion-v3-medium"
    else:
        enable_modelspec = False

    if enable_modelspec:
        metadata["modelspec.sai_model_spec"] = "1.0.0"
        metadata["modelspec.implementation"] = "sgm"
        metadata["modelspec.title"] = "{} {}".format(filename, counter)

    model_sampling = model.get_model_object("model_sampling")
    if isinstance(model_sampling, comfy.model_sampling.ModelSamplingContinuousEDM):
        if isinstance(model_sampling, comfy.model_sampling.V_PREDICTION):
            extra_keys["edm_vpred.sigma_max"] = torch.tensor(model_sampling.sigma_max).float()
            extra_keys["edm_vpred.sigma_min"] = torch.tensor(model_sampling.sigma_min).float()

    if model.model.model_type == comfy.model_base.ModelType.EPS:
        metadata["modelspec.predict_key"] = "epsilon"
    elif model.model.model_type == comfy.model_base.ModelType.V_PREDICTION:
        metadata["modelspec.predict_key"] = "v"
        extra_keys["v_pred"] = torch.tensor([])
        if getattr(model_sampling, "zsnr", False):
            extra_keys["ztsnr"] = torch.tensor([])

    return metadata, extra_keys


def _materialize_and_cast(sd, dtype):
    """
    Force-materialize patched/LazyCastingParam tensors and optionally cast them.

    The completed state dict is CPU-resident. This avoids requiring the full baked
    model to fit in VRAM; ComfyUI can use its normal partial-loading path while
    patched weights are materialized for saving. System RAM must still be large
    enough for the completed output state dict.
    """
    target_dtype = DTYPE_MAP.get(dtype) if dtype != "original" else None

    out = {}
    keys = list(sd.keys())
    for k in keys:
        t = sd[k]
        # LazyCastingParam.to("cpu") triggers patch_weight_to_device, which is
        # the canonical path that bakes attached ModelPatcher patches/LoRAs.
        if isinstance(t, torch.Tensor):
            t = t.to("cpu")
            if target_dtype is not None and t.is_floating_point():
                t = t.to(target_dtype)
            if not t.is_contiguous():
                t = t.contiguous()
        out[k] = t
        # Release the original wrapper/reference promptly while building output.
        sd[k] = None

    return out


def _save_via_comfy(model, clip, vae, output_path, filename, counter, dtype):
    """
    Faithful reproduction of comfy.sd.save_checkpoint with:
      - workflow metadata stripped
      - optional dtype conversion
    """
    metadata, extra_keys = _build_modelspec_metadata(model, filename, counter)
    if args.disable_metadata:
        metadata = {}

    clip_sd = None
    load_models = [model]
    if clip is not None:
        load_models.append(clip.load_model())
        clip_sd = clip.get_sd()
    vae_sd = None
    if vae is not None:
        vae_sd = vae.get_sd()

    # Do not force_patch_weights here. ComfyUI's normal load_models_gpu path may
    # partially load large models on low-VRAM systems, and LazyCastingParam
    # materialization below bakes patches during the CPU state-dict pass.
    comfy.model_management.load_models_gpu(load_models)

    clip_vision_sd = None
    sd = model.state_dict_for_saving(clip_sd, vae_sd, clip_vision_sd)
    for k in extra_keys:
        sd[k] = extra_keys[k]

    sd = _materialize_and_cast(sd, dtype)

    offload_models(comfy.model_management, *load_models)

    comfy.utils.save_torch_file(sd, output_path, metadata=metadata)


class DonutSave:
    """
    Connection-driven unified save node. Saves the diffusion model only when
    clip/vae are unwired, or a full checkpoint (model + clip + vae) when they
    are connected. No workflow metadata is embedded and dtype is selectable.
    """

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "dtype": (DTYPE_OPTIONS, {
                    "default": "original",
                    "tooltip": "Output floating-point dtype. 'original' preserves the loaded dtype, including fp8 if the model was loaded as fp8.",
                }),
            },
            "optional": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "advanced/model_merging"

    def save(self, model, filename_prefix, dtype="original", clip=None, vae=None):
        full_output_folder, filename, counter, subfolder, filename_prefix = \
            folder_paths.get_save_image_path(filename_prefix, self.output_dir)

        output_path = os.path.join(
            full_output_folder, f"{filename}_{counter:05}_.safetensors"
        )

        _save_via_comfy(
            model=model,
            clip=clip,
            vae=vae,
            output_path=output_path,
            filename=filename,
            counter=counter,
            dtype=dtype,
        )
        kind = "checkpoint" if (clip is not None or vae is not None) else "model"
        print(f"[DonutSave] Saved {dtype} {kind} to {output_path}")
        return {}


class DonutModelSave(DonutSave):
    """Save only the diffusion MODEL, with patches/LoRAs baked and dtype explicit."""

    DEPRECATED = False
    DESCRIPTION = (
        "Saves only the diffusion model as safetensors with no workflow metadata. "
        "Attached ModelPatcher changes such as LoRAs/merges are baked into the saved weights. "
        "Choose bf16/fp16/fp32 to override a model that ComfyUI loaded as fp8."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "filename_prefix": ("STRING", {
                    "default": "diffusion_models/ComfyUI",
                    "tooltip": "Saved under ComfyUI/output by default; use diffusion_models/... for model-library style organization.",
                }),
                "dtype": (DTYPE_OPTIONS, {
                    "default": "bf16",
                    "tooltip": "Saved weight dtype. BF16 is the default to avoid accidentally preserving an fp8-loaded model. 'original' keeps the currently loaded dtype.",
                }),
            }
        }

    def save(self, model, filename_prefix, dtype="bf16"):
        return super().save(
            model=model,
            filename_prefix=filename_prefix,
            dtype=dtype,
        )


class DonutCheckpointSave(DonutSave):
    """Legacy full checkpoint alias (model + CLIP + VAE)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "filename_prefix": ("STRING", {"default": "checkpoints/ComfyUI"}),
                "dtype": (DTYPE_OPTIONS, {"default": "original"}),
            }
        }

    def save(self, model, clip, vae, filename_prefix, dtype="original"):
        return super().save(
            model=model,
            filename_prefix=filename_prefix,
            dtype=dtype,
            clip=clip,
            vae=vae,
        )


NODE_CLASS_MAPPINGS = {
    "DonutSave": DonutSave,
    "DonutModelSave": DonutModelSave,
    "DonutCheckpointSave": DonutCheckpointSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutSave": "Donut Save (No Workflow)",
    "DonutModelSave": "Donut Diffusion Model Save (No Workflow)",
}

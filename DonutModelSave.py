"""No-workflow model/checkpoint save nodes with explicit dtype control.

The save path follows ComfyUI's normal validation first. A narrow fallback allows
an already-existing symlink located under ComfyUI's output directory, which is a
common way to redirect large model writes to a faster/larger SSD.

Serialization uses ComfyUI's lazy state-dict path. Ordinary patches are baked one
weight at a time; Donut Experimental-bypass LoRAs are converted to equivalent
ordinary patches for saving; Donut Model Merge Krea2 Experimental-bypass hard
swaps are composed from the retained model2 source state.
"""

from contextlib import nullcontext
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
    from .donut_save_path import get_model_save_path
    from .donut_bypass_materialization import (
        clone_with_bypass_as_regular_patches,
        get_bypass_components,
    )
    from .donut_krea2_merge_serialization import (
        clone_with_regular_components,
        clone_without_krea2_merge_runtime,
        compose_krea2_merge_unet_state_dict,
        get_krea2_merge_bypass_info,
    )
    from .DonutExtractLoRA import DonutExtractLoRA
except ImportError:
    from model_lifecycle import offload_models
    from donut_save_path import get_model_save_path
    from donut_bypass_materialization import (
        clone_with_bypass_as_regular_patches,
        get_bypass_components,
    )
    from donut_krea2_merge_serialization import (
        clone_with_regular_components,
        clone_without_krea2_merge_runtime,
        compose_krea2_merge_unet_state_dict,
        get_krea2_merge_bypass_info,
    )
    from DonutExtractLoRA import DonutExtractLoRA


DTYPE_OPTIONS = ["original", "bf16", "fp16", "fp32", "fp8_e4m3fn", "fp8_e5m2"]
DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp8_e4m3fn": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}


def _build_modelspec_metadata(model, filename, counter):
    """Build ComfyUI-style model-spec metadata without workflow metadata."""
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
        metadata["modelspec.title"] = f"{filename} {counter}"

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
    """Materialize lazy patched tensors to CPU and optionally cast floating weights."""
    target_dtype = DTYPE_MAP.get(dtype) if dtype != "original" else None
    out = {}

    for key in list(sd.keys()):
        tensor = sd[key]
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.to("cpu")
            if target_dtype is not None and tensor.is_floating_point():
                tensor = tensor.to(target_dtype)
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
        out[key] = tensor
        sd[key] = None

    return out


def _save_via_comfy(model, clip, vae, output_path, filename, counter, dtype):
    """Save while materializing Donut runtime-only LoRA and Krea2 merge bypasses."""
    lora_bypass_components = get_bypass_components(model)
    krea2_merge_info = get_krea2_merge_bypass_info(model)
    has_runtime_serialization = bool(lora_bypass_components) or krea2_merge_info is not None

    use_ejected = getattr(model, "use_ejected", None)
    context = (
        use_ejected()
        if has_runtime_serialization and callable(use_ejected)
        else nullcontext()
    )

    with context:
        # Final-model bypass LoRAs become ordinary patches for model1 and any
        # partial-blend keys. Exact Krea2 model2 swaps are replaced separately.
        save_model = clone_with_bypass_as_regular_patches(model)
        source_for_save = None
        krea2_plans = None

        if krea2_merge_info is not None:
            source_model, krea2_plans, _attachment_keys = krea2_merge_info
            save_model = clone_without_krea2_merge_runtime(save_model, krea2_merge_info)

            swapped_weight_keys = {plan[1] for plan in krea2_plans}
            source_for_save = clone_with_regular_components(
                source_model,
                lora_bypass_components,
                allowed_keys=swapped_weight_keys,
            )

        metadata, extra_keys = _build_modelspec_metadata(save_model, filename, counter)
        if args.disable_metadata:
            metadata = {}

        clip_sd = None
        load_models = [save_model]
        if source_for_save is not None:
            load_models.append(source_for_save)
        if clip is not None:
            load_models.append(clip.load_model())
            clip_sd = clip.get_sd()

        vae_sd = vae.get_sd() if vae is not None else None

        # Keep ComfyUI's normal partial-loading path. Lazy state-dict tensors bake
        # regular patches only as individual weights are materialized to CPU.
        comfy.model_management.load_models_gpu(load_models)

        clip_vision_sd = None
        if source_for_save is not None:
            unet_sd, swapped_modules, swapped_state_keys = compose_krea2_merge_unet_state_dict(
                save_model,
                source_for_save,
                krea2_plans,
            )
            sd = save_model.model.state_dict_for_saving(
                unet_sd,
                clip_state_dict=clip_sd,
                vae_state_dict=vae_sd,
                clip_vision_state_dict=clip_vision_sd,
            )
            print(
                "[DonutSave] Materializing Krea2 Experimental-bypass merge: "
                f"{swapped_modules} model2 module swap(s), "
                f"{swapped_state_keys} direct state key(s)"
            )
        else:
            sd = save_model.state_dict_for_saving(clip_sd, vae_sd, clip_vision_sd)

        for key, value in extra_keys.items():
            sd[key] = value

        sd = _materialize_and_cast(sd, dtype)
        offload_models(comfy.model_management, *load_models)

    comfy.utils.save_torch_file(sd, output_path, metadata=metadata)


class DonutSave:
    """Connection-driven no-workflow model/checkpoint saver."""

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
                    "tooltip": (
                        "Output floating-point dtype. 'original' preserves the loaded "
                        "dtype, including fp8 if the model was loaded as fp8."
                    ),
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
        full_output_folder, filename, counter, _subfolder, _prefix = get_model_save_path(
            folder_paths,
            filename_prefix,
            self.output_dir,
        )
        output_path = os.path.join(
            full_output_folder,
            f"{filename}_{counter:05}_.safetensors",
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
    """Legacy model-only saver ID kept for saved-workflow compatibility."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "filename_prefix": ("STRING", {
                    "default": "diffusion_models/ComfyUI",
                    "tooltip": (
                        "Saved under ComfyUI/output. Existing symlinks beneath output "
                        "are supported, e.g. output/diffusion_models -> a faster SSD."
                    ),
                }),
                "dtype": (DTYPE_OPTIONS, {
                    "default": "bf16",
                    "tooltip": (
                        "Saved weight dtype. BF16 is the default to avoid accidentally "
                        "preserving an fp8-loaded model. 'original' keeps the loaded dtype."
                    ),
                }),
            }
        }

    def save(self, model, filename_prefix, dtype="bf16"):
        return super().save(model=model, filename_prefix=filename_prefix, dtype=dtype)


class DonutDiffusionModelSave(DonutModelSave):
    """Visible diffusion-model-only save node with explicit output dtype."""

    DEPRECATED = False
    DESCRIPTION = (
        "Saves only the diffusion MODEL as safetensors with no workflow metadata. "
        "Ordinary ModelPatcher changes, Donut Experimental-bypass LoRAs, and "
        "Donut Model Merge Krea2 Experimental-bypass hard swaps are baked into the "
        "saved weights. Existing output-directory symlinks are supported for fast "
        "model storage. BF16 is the default so an fp8-loaded model is not silently "
        "re-saved as fp8."
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
    "DonutDiffusionModelSave": DonutDiffusionModelSave,
    "DonutCheckpointSave": DonutCheckpointSave,
    "DonutExtractLoRA": DonutExtractLoRA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutSave": "Donut Save (No Workflow)",
    "DonutDiffusionModelSave": "Donut Diffusion Model Save (No Workflow)",
    "DonutExtractLoRA": "Donut Extract LoRA (Raw → Patched)",
}

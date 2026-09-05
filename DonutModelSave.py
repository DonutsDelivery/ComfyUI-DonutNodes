"""
DonutSave / DonutModelSave / DonutCheckpointSave

No-workflow save nodes with explicit dtype control.

DonutModelSave is the dedicated diffusion-model-only saver: it accepts only MODEL,
never CLIP/VAE, and writes under diffusion_models/ by default. This is useful when
ComfyUI loaded a model in fp8 but the desired baked LoRA/merge output should be
saved as bf16/fp16/fp32 instead.

The state-dict construction path follows comfy.sd.save_checkpoint:
load_models_gpu(...) -> ModelPatcher state_dict_for_saving machinery. Ordinary
patches are materialized one weight at a time. Donut Experimental-bypass LoRAs
are converted on a temporary clone to equivalent ordinary adapter patches.
Donut Model Merge Krea2 Experimental-bypass hard swaps are composed from the
retained model2 source state so those runtime-only swaps are also baked.
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
        # Drop the original entry so wrappers can be released promptly.
        sd[k] = None

    return out


def _save_via_comfy(model, clip, vae, output_path, filename, counter, dtype):
    """Save a model/checkpoint while materializing Donut runtime-only bypasses."""
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
        # Convert final-model bypass LoRAs to ordinary patches for every normal
        # model1/partial-blend key. The Krea2 exact-swap keys are replaced from
        # model2 below, so their final bypass-LoRA components are also applied to
        # a source-model clone before composing the state dict.
        save_model = clone_with_bypass_as_regular_patches(model)
        source_for_save = None
        krea2_plans = None

        if krea2_merge_info is not None:
            source_model, krea2_plans, _attachment_keys = krea2_merge_info
            save_model = clone_without_krea2_merge_runtime(
                save_model,
                krea2_merge_info,
            )

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
            # Keeping source model2 explicit here preserves ComfyUI's normal
            # partial-loading behavior on low-VRAM systems instead of forcing a
            # dense merged model into VRAM.
            load_models.append(source_for_save)
        if clip is not None:
            load_models.append(clip.load_model())
            clip_sd = clip.get_sd()
        vae_sd = None
        if vae is not None:
            vae_sd = vae.get_sd()

        # Do not force_patch_weights. Normal partial loading is important on
        # low-VRAM systems; lazy state-dict tensors apply regular patches as each
        # tensor is materialized on CPU.
        comfy.model_management.load_models_gpu(load_models)

        clip_vision_sd = None
        if source_for_save is not None:
            unet_sd, swapped_modules, swapped_state_keys = \
                compose_krea2_merge_unet_state_dict(
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
            sd = save_model.state_dict_for_saving(
                clip_sd,
                vae_sd,
                clip_vision_sd,
            )

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
    """Legacy model-only saver ID kept for saved-workflow compatibility."""

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


class DonutDiffusionModelSave(DonutModelSave):
    """Visible diffusion-model-only save node with explicit output dtype."""

    DEPRECATED = False
    DESCRIPTION = (
        "Saves only the diffusion MODEL as safetensors with no workflow metadata. "
        "Ordinary ModelPatcher changes, Donut Experimental-bypass LoRAs, and "
        "Donut Model Merge Krea2 Experimental-bypass hard swaps are baked into "
        "the saved weights. BF16 is the default so a model loaded in fp8 is not "
        "silently re-saved as fp8."
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

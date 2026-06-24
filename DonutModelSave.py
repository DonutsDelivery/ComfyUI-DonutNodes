"""
DonutModelSave / DonutCheckpointSave

Mirrors ComfyUI's stock CheckpointSave / ModelSave behavior exactly, but:
  - Strips the embedded workflow (no `prompt` / `extra_pnginfo` in metadata)
  - Adds an optional dtype conversion (fp8/fp16/bf16/fp32)

The state-dict construction path is the same as comfy.sd.save_checkpoint:
load_models_gpu(...) -> ModelPatcher.state_dict_for_saving(...). This routes
through ComfyUI's LazyCastingParam machinery, which is the canonical way
patches (LoRAs, merges) get baked into a saved checkpoint.
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


DTYPE_OPTIONS = ["original", "fp8_e4m3fn", "fp8_e5m2", "fp16", "bf16", "fp32"]

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
    Walk the state dict, force-materialize any LazyCastingParam wrappers
    (this is what causes patches/LoRAs to actually get applied to the
    saved tensors), and optionally cast floating-point tensors to a
    target dtype. Result lives on CPU and is contiguous.
    """
    target_dtype = DTYPE_MAP.get(dtype) if dtype != "original" else None

    out = {}
    keys = list(sd.keys())
    for k in keys:
        t = sd[k]
        # LazyCastingParam.to("cpu") triggers patch_weight_to_device, which
        # is the canonical way patched weights get materialized for save.
        if isinstance(t, torch.Tensor):
            t = t.to("cpu")
            if target_dtype is not None and t.is_floating_point():
                t = t.to(target_dtype)
            if not t.is_contiguous():
                t = t.contiguous()
        out[k] = t
        # Drop the original entry so we release the wrapper promptly.
        sd[k] = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out


def _save_via_comfy(model, clip, vae, output_path, filename, counter, dtype):
    """
    Faithful reproduction of comfy.sd.save_checkpoint with:
      - workflow metadata stripped
      - optional dtype conversion
    """
    # Build modelspec metadata exactly the way CheckpointSave would,
    # but without prompt / extra_pnginfo. args.disable_metadata is
    # respected too — if the user disabled metadata globally we emit none.
    metadata, extra_keys = _build_modelspec_metadata(model, filename, counter)
    if args.disable_metadata:
        metadata = {}

    # --- the rest mirrors comfy.sd.save_checkpoint exactly ---
    clip_sd = None
    load_models = [model]
    if clip is not None:
        load_models.append(clip.load_model())
        clip_sd = clip.get_sd()
    vae_sd = None
    if vae is not None:
        vae_sd = vae.get_sd()

    # Match comfy.sd.save_checkpoint exactly: no force_patch_weights.
    # Patches that aren't physically applied in-place are wrapped in
    # LazyCastingParam and resolved when we call .to("cpu") below.
    # (force_patch_weights=True would assert-fail under partial loading,
    # which is what kicks in on low-VRAM systems.)
    comfy.model_management.load_models_gpu(load_models)

    clip_vision_sd = None
    sd = model.state_dict_for_saving(clip_sd, vae_sd, clip_vision_sd)
    for k in extra_keys:
        sd[k] = extra_keys[k]

    # Materialize lazy wrappers, optionally cast dtype, ensure contiguous.
    sd = _materialize_and_cast(sd, dtype)

    # Free GPU once we have CPU copies of everything.
    comfy.model_management.unload_all_models()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    comfy.utils.save_torch_file(sd, output_path, metadata=metadata)


class DonutSave:
    """
    Connection-driven unified save node. Saves the diffusion model only when
    clip/vae are unwired, or a full checkpoint (model + clip + vae) when they
    are connected. Behaves like ComfyUI's stock ModelSave / CheckpointSave
    nodes, except no workflow is embedded and dtype is selectable.

    The save() method here is the shared engine; DonutModelSave and
    DonutCheckpointSave are thin alias subclasses that keep their original
    INPUT_TYPES for byte-identical deserialization of saved workflows.
    """

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "dtype": (DTYPE_OPTIONS, {"default": "original"}),
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
    """
    Alias of DonutSave preserving the original ModelSave INPUT_TYPES
    (model + filename_prefix + dtype, no clip/vae). Delegates to
    DonutSave.save().
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "filename_prefix": ("STRING", {"default": "diffusion_models/ComfyUI"}),
                "dtype": (DTYPE_OPTIONS, {"default": "original"}),
            }
        }

    def save(self, model, filename_prefix, dtype="original"):
        return super().save(
            model=model,
            filename_prefix=filename_prefix,
            dtype=dtype,
        )


class DonutCheckpointSave(DonutSave):
    """
    Alias of DonutSave preserving the original CheckpointSave INPUT_TYPES
    (model + clip + vae + filename_prefix + dtype). Delegates to
    DonutSave.save().
    """

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
}

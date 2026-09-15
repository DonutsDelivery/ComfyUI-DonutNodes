"""Native Krea2 Turbo SDA: one uninterrupted DonutSampler run.

SDA is scheduled, not put in the ordinary LoRA stack. Comfy patches uses native
weight hooks; Experimental bypass scopes forward adapters to early predictions.
Neither path restarts the solver, reseeds it, or disables upstream LoRAs.
"""
from copy import deepcopy
import hashlib
import math
import os
import re

import comfy.utils
import folder_paths

try:
    from .DonutKSamplerCFGLinear import DonutSampler as _BaseDonutSampler
    from .donut_lora_execution import publish_execution_mode, resolve_execution_mode
except ImportError:
    from DonutKSamplerCFGLinear import DonutSampler as _BaseDonutSampler
    from donut_lora_execution import publish_execution_mode, resolve_execution_mode

SDA_LORA_NAME = "krea2/krea2_turbo_sda_v1.0_comfy.safetensors"
SDA_SHA256 = "0fafed045c53c4acd6165eb55da6ec04b24785b1eeed6f1be37b2cdcb66dba2b"
SDA_FILE_SIZE = 469315664
SDA_SUPPORTED_STEPS = 8
SDA_GATE_STEPS = 2


def _schedule_module():
    # Disabled SDA must not require a newer ComfyUI hooks/bypass API.
    try:
        from . import donut_sda_schedule
    except ImportError:
        import donut_sda_schedule
    return donut_sda_schedule


def _sda_path():
    path = folder_paths.get_full_path("loras", SDA_LORA_NAME)
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            f"SDA needs ComfyUI/models/loras/{SDA_LORA_NAME}. "
            "Install F16's krea2_turbo_sda_v1.0_comfy.safetensors (not the "
            "Diffusers file). The Registry/Manager build requires manual "
            "installation; the full GitHub build can use Download missing. "
            "Enabling SDA never downloads a model automatically."
        )
    return path


def _file_identity(path):
    stat = os.stat(path)
    return (os.path.realpath(path), stat.st_dev, stat.st_ino, stat.st_size,
            stat.st_mtime_ns, stat.st_ctime_ns)


def _load_verified_lora(path):
    if os.path.getsize(path) != SDA_FILE_SIZE:
        raise ValueError(
            f"Wrong SDA file size: expected {SDA_FILE_SIZE} bytes. Download "
            "krea2_turbo_sda_v1.0_comfy.safetensors again; do not rename the Diffusers file."
        )
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != SDA_SHA256:
        raise ValueError("SDA SHA-256 mismatch: the file is not the pinned F16 ComfyUI adapter.")
    return comfy.utils.load_torch_file(path, safe_load=True)


def _validate_sda_sampling(kwargs, supported_samplers):
    if not kwargs["turbo_mode"]:
        raise ValueError("SDA diversity requires Turbo mode.")
    if kwargs["steps"] != SDA_SUPPORTED_STEPS:
        raise ValueError("SDA diversity requires the complete 8-step Krea2 Turbo schedule.")
    denoise = float(kwargs["denoise"])
    if not math.isfinite(denoise) or denoise != 1.0:
        raise ValueError("SDA requires full-denoise base generation (denoise=1); disable it for refinement.")
    if kwargs["edit_mode"] or kwargs["latent_image"].get("noise_mask") is not None:
        raise ValueError("SDA is not supported for editing/inpainting or masked generation.")
    if kwargs["mode"] not in ("simple", "advanced"):
        raise ValueError("SDA uses one model/sampler run; select simple or advanced, not multi_model.")
    sampler_name = kwargs["sampler_name"]
    # Bleh is a preset indirection, not another solver. Admit its name here;
    # the runtime guard resolves the LIVE registered SAMPLER and verifies the
    # underlying kernel/options before executing it. Never assume slot 0 is ER-SDE.
    is_bleh_preset = isinstance(sampler_name, str) and re.fullmatch(r"bleh_preset_[0-9]+", sampler_name)
    if sampler_name not in supported_samplers and not is_bleh_preset:
        raise ValueError(
            f"SDA cannot schedule sampler {sampler_name!r}. Supported underlying "
            f"solvers: {', '.join(supported_samplers)}, including verified Bleh presets. "
            "Other/adaptive/multi-evaluation solvers do not yet have a verified two-step gate."
        )
    # Simple mode ignores these dormant advanced controls. Do not reject a
    # previously saved simple workflow because its hidden step range is stale.
    if kwargs["mode"] == "advanced":
        if kwargs["start_at_step"] != 0 or kwargs["end_at_step"] < SDA_SUPPORTED_STEPS:
            raise ValueError("SDA advanced sampling must cover all 8 steps, starting at step 0.")
        if kwargs["add_noise"] != "enable" or kwargs["return_with_leftover_noise"] != "disable":
            raise ValueError("SDA advanced sampling requires initial noise and a fully denoised result.")
    if not kwargs.get("nag_enabled", False):
        if any(float(kwargs[name]) != 1.0 for name in ("cfg_start", "cfg_halfway", "cfg_end")):
            raise ValueError("SDA's Krea2 Turbo reference uses CFG 1; set all three CFG values to 1.")


def _validate_model(model):
    diffusion = getattr(getattr(model, "model", None), "diffusion_model", None)
    if (getattr(diffusion, "txtlayers", None) != 12
            or getattr(diffusion, "txtdim", None) != 2560
            or not hasattr(diffusion, "txtfusion") or not hasattr(diffusion, "blocks")):
        raise ValueError("SDA requires a compatible, uncompiled Krea2 diffusion model.")
    if hasattr(diffusion, "_orig_mod") or getattr(diffusion, "_compiled_call_impl", None) is not None:
        raise ValueError("Disable torch.compile for scheduled SDA; compiled forward gating is not validated.")
    try:
        from .donut_sda_merge import checked_merge_info
    except ImportError:
        from donut_sda_merge import checked_merge_info
    # A hard swap is supported. Require its real source/plan rather than
    # rejecting the merge or silently patching an unused primary-model layer.
    checked_merge_info(model)


class DonutSampler(_BaseDonutSampler):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = deepcopy(super().INPUT_TYPES())
        optional = inputs.setdefault("optional", {})
        # Append only: preserve every previously serialized widget position.
        optional["sda_enabled"] = ("BOOLEAN", {
            "default": False,
            "tooltip": "Krea2 Turbo SDA: first 2 of 8 steps in ONE uninterrupted run. "
                       "Supports Euler, ER-SDE and DPM++ 2M, including V4's Bleh preset (ODE) + beta. "
                       "Preserves preset options and supports Donut hard-swap merges in Experimental bypass. "
                       "Requires the F16 ComfyUI SDA file; does not auto-download.",
        })
        optional["sda_strength"] = ("FLOAT", {
            "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
            "tooltip": "SDA strength; 1.0 is the reference. Zero is an exact SDA-off pass-through.",
        })
        return inputs

    def sample(
        self, model, seed, steps, cfg_start, cfg_halfway, cfg_end, halfway_step,
        sampler_name, scheduler, positive, negative, latent_image, denoise,
        mode="simple", cfg_curve="linear", add_noise="enable", start_at_step=0,
        end_at_step=10000, return_with_leftover_noise="disable",
        randomize_seed_per_model="enable", switch_at_step_1=10,
        switch_at_step_2=15, model_2=None, model_3=None, edit_mode=False,
        source_image=None, vae=None, clip=None, edit_prompt="",
        edit_negative_prompt="", grounding_px=768, edit_model=None,
        turbo_mode=False, source_image_b=None, edit_inpaint=None,
        sda_enabled=False, sda_strength=1.0, **nag_options,
    ):
        kwargs = dict(
            seed=seed, steps=steps, cfg_start=cfg_start, cfg_halfway=cfg_halfway,
            cfg_end=cfg_end, halfway_step=halfway_step, sampler_name=sampler_name,
            scheduler=scheduler, positive=positive, negative=negative,
            latent_image=latent_image, denoise=denoise, mode=mode,
            cfg_curve=cfg_curve, add_noise=add_noise, start_at_step=start_at_step,
            end_at_step=end_at_step, return_with_leftover_noise=return_with_leftover_noise,
            randomize_seed_per_model=randomize_seed_per_model,
            switch_at_step_1=switch_at_step_1, switch_at_step_2=switch_at_step_2,
            model_2=model_2, model_3=model_3, edit_mode=edit_mode,
            source_image=source_image, vae=vae, clip=clip, edit_prompt=edit_prompt,
            edit_negative_prompt=edit_negative_prompt, grounding_px=grounding_px,
            edit_model=edit_model, turbo_mode=turbo_mode, source_image_b=source_image_b,
            edit_inpaint=edit_inpaint, **nag_options,
        )
        if not sda_enabled:
            return super().sample(model=model, **kwargs)
        strength = float(sda_strength)
        if not math.isfinite(strength) or not 0.0 <= strength <= 2.0:
            raise ValueError("SDA strength must be finite and between 0 and 2.")
        if strength == 0.0:
            return super().sample(model=model, **kwargs)

        schedule = _schedule_module()
        _validate_sda_sampling(kwargs, schedule.SDA_SAMPLERS)
        _validate_model(model)
        path = _sda_path()
        identity = _file_identity(path)
        cache = getattr(self, "_sda_file_cache", None)
        if cache is None or cache[0] != identity:
            self._sda_file_cache = None
            lora = _load_verified_lora(path)
            if _file_identity(path) != identity:
                raise RuntimeError("SDA file changed while loading; finish the download and retry.")
            self._sda_file_cache = (identity, lora)
        else:
            lora = cache[1]
        patches = schedule.map_sda_weights(model, lora)
        execution_mode = resolve_execution_mode(model)
        scheduled, kwargs["positive"], kwargs["negative"] = schedule.prepare_sda(
            model, positive, negative, patches, strength, execution_mode)
        publish_execution_mode(scheduled, execution_mode)
        # Keep the caller's simple/advanced mode, CFG handling, NAG, latent,
        # scheduler, callbacks and seed. There is exactly ONE base sampler call.
        latent, info = super().sample(model=scheduled, **kwargs)
        return latent, (
            f"SDA: verified {SDA_LORA_NAME}; strength={strength:g}; {execution_mode}; "
            f"single run, ON 1-2 / OFF 3-8\n{info}"
        )


NODE_CLASS_MAPPINGS = {"DonutSampler": DonutSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"DonutSampler": "DonutSampler"}

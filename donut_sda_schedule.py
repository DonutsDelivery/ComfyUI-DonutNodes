"""Single-run SDA scheduling. No sampler function or global model is replaced.

Supported solvers evaluate the denoiser once per schedule entry. The third
sigma is therefore the exact OFF boundary, regardless of the scheduler's
non-linear spacing. Adaptive/multi-evaluation solvers are rejected by the node;
a 25% diffusion-time hook or a model-call counter is not a two-step schedule.
"""
from contextlib import ExitStack
from copy import copy
import logging

import torch
import comfy.hooks
import comfy.lora
import comfy.lora_convert
import comfy.patcher_extension

SDA_STEPS = 8
SDA_GATE_STEPS = 2
# Deliberately narrow: these stock solvers have one prediction per interval.
# Do not add a solver without checking sub-evaluations, churn and step semantics.
SDA_SAMPLERS = ("euler", "er_sde", "dpmpp_2m")
SDA_WRAPPER_KEY = "donut_krea2_sda_single_run"


def validate_sigmas(sigmas):
    if not torch.is_tensor(sigmas) or sigmas.ndim != 1 or sigmas.numel() != SDA_STEPS + 1:
        raise ValueError("SDA requires the complete 8-step sigma schedule (9 sigma values).")
    values = sigmas.detach().float().cpu()
    if not torch.isfinite(values).all() or values[-1] != 0 or not torch.all(values[:-1] > values[1:]):
        raise ValueError("SDA requires finite, strictly descending sigmas ending at zero.")
    return values.tolist()


def sda_active(timestep, sigmas):
    """Use raw sampler sigmas, not the model's converted 0..1000 timesteps."""
    schedule = validate_sigmas(sigmas)
    t = torch.as_tensor(timestep).detach().float()
    if t.numel() == 0 or not torch.isfinite(t).all():
        raise ValueError("SDA received an empty or non-finite sampling timestep.")
    active = t > schedule[SDA_GATE_STEPS]
    if not bool(torch.all(active == active.flatten()[0])):
        raise ValueError("SDA cannot mix enabled/disabled sigmas within one model batch.")
    return bool(active.flatten()[0])


class _SDAKeyframes(comfy.hooks.HookKeyframeGroup):
    """Native weight hooks with a schedule-index boundary, not percent=0.25."""

    def __init__(self):
        super().__init__()
        self.add(comfy.hooks.HookKeyframe(strength=1.0, start_percent=0.0, guarantee_steps=0))
        self.add(comfy.hooks.HookKeyframe(strength=0.0, start_percent=1.0, guarantee_steps=0))

    def clone(self):
        return type(self)()

    def prepare_current_keyframe(self, curr_t, transformer_options):
        index = 0 if sda_active(curr_t, transformer_options.get("sample_sigmas")) else 1
        changed = index != self._current_index
        if self._current_strength is None or changed:
            logging.info("[Donut SDA] Weight hook %s at sigma=%.7g",
                         "ON" if index == 0 else "OFF", float(curr_t))
        self._current_index = index
        self._current_keyframe = self.keyframes[index]
        self._current_strength = self._current_keyframe.strength
        return changed


def _attach_hooks(positive, negative, hook_group):
    """Keep upstream hooks and conditioning metadata; never mutate inputs."""
    combined = {None: hook_group}

    def attach(conditioning):
        output = []
        for tensor, metadata in conditioning:
            existing = metadata.get("hooks")
            if existing not in combined:
                combined[existing] = comfy.hooks.HookGroup.combine_all_hooks([existing, hook_group])
            output.append([tensor, dict(metadata, hooks=combined[existing])])
        return output

    return attach(positive), attach(negative)


def map_sda_weights(model, lora):
    """Use the stock ComfyUI loader's conversion/key map, then require coverage."""
    converted = comfy.lora_convert.convert_lora(lora)
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    patches = comfy.lora.load_lora(converted, key_map)
    expected = {key for key, value in converted.items()
                if torch.is_tensor(value) and value.ndim >= 2}
    consumed = set().union(*(getattr(adapter, "loaded_keys", set()) for adapter in patches.values()))
    if not patches or not expected or expected - consumed:
        raise ValueError(
            f"SDA mapped {len(patches)} adapters but left weight tensors unmatched: "
            f"{sorted(expected - consumed)[:3]}. Use F16's ComfyUI-format file and "
            "a compatible Krea2 model; partial/no-op loading is not supported."
        )
    model_keys = model.model.state_dict().keys()
    missing = [key for key in patches if (key[0] if isinstance(key, tuple) else key) not in model_keys]
    if missing:
        raise ValueError(f"SDA targets are missing on this model: {missing[:3]}")
    return patches


class _ScopedSDABypass:
    """Nest only SDA around a physical model forward, then restore it in finally.

    APPLY_MODEL is outside diffusion forward replacements such as NAG, but
    inside conditioning-hook weight selection. Existing Donut bypass hooks are
    left installed. No second persistent injection group is created, avoiding
    cross-group ejection-order bugs and never converting ordinary LoRAs.
    """

    def __init__(self, patches, strength, targets=None):
        self.patches = patches
        self.strength = strength
        self.targets = targets
        self._last_active = None

    def __call__(self, executor, x, t, c_concat=None, c_crossattn=None,
                 control=None, transformer_options=None, **kwargs):
        options = transformer_options or {}
        active = sda_active(t, options.get("sample_sigmas"))
        if active != self._last_active:
            logging.info("[Donut SDA] Runtime adapter %s at sigma=%.7g",
                         "ON" if active else "OFF", float(torch.as_tensor(t).flatten()[0]))
            self._last_active = active
        if not active:
            return executor(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)

        from comfy.weight_adapter import BypassInjectionManager
        # Each forward owns its adapters and their device casts. Do not mutate
        # cached CPU tensors or the upstream LoRA manager, even on interruption.
        groups = (self.targets.roots(executor.class_obj, runtime=True)
                  if self.targets is not None else [("primary", executor.class_obj, self.patches)])
        # Build every group before injecting any. SDA must target the retained
        # source for exact swaps, but the primary for ordinary/partial merges.
        # The source's own apply_model is NOT called by a linear forward swap,
        # so putting a second APPLY_MODEL wrapper on that source would do nothing.
        prepared = []
        for label, root, patches in groups:
            manager = BypassInjectionManager()
            for key, adapter in patches.items():
                manager.add_adapter(key, copy(adapter), strength=self.strength)
            injections = manager.create_injections(root)
            if manager.get_hook_count() != len(patches):
                raise RuntimeError(f"SDA could not bind every {label} adapter to the sampling model.")
            prepared.extend(injections)
        # ExitStack attempts ALL cleanups, even if an individual eject raises.
        # Register before inject so partial source failures restore both roots.
        with ExitStack() as cleanup:
            for injection in prepared:
                cleanup.callback(injection.eject, None)
                injection.inject(None)
            return executor(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)


def _sampling_guard(executor, model_wrap, sigmas, extra_args, callback, noise,
                    latent_image=None, denoise_mask=None, disable_pbar=False):
    schedule = validate_sigmas(sigmas)
    sampler = executor.class_obj
    try:
        from .donut_sda_sampler import inspect_sda_sampler
    except ImportError:
        from donut_sda_sampler import inspect_sda_sampler
    route, options = inspect_sda_sampler(sampler, SDA_SAMPLERS)
    logging.info("[Donut SDA] Solver: %s; s_noise=%s; max_stage=%s; noise_scaler=%s",
                 route, options.get("s_noise", "default"), options.get("max_stage", "default"),
                 getattr(options.get("noise_scaler"), "__name__", "default"))
    logging.info("[Donut SDA] Single run: ON steps 1-2, OFF steps 3-8; cutoff sigma=%.7g", schedule[2])
    # No second sampler invocation: history, stochastic noise state, seed and
    # callbacks are passed through without restart or additional initial noise.
    # In particular, execute Bleh's original wrapper and configured SAMPLER;
    # constructing a fresh er_sde here would discard V4's deterministic ODE mode.
    return executor(model_wrap, sigmas, extra_args, callback, noise,
                    latent_image, denoise_mask, disable_pbar)


def prepare_sda(model, positive, negative, patches, strength, execution_mode):
    patched = model.clone()
    wrappers = comfy.patcher_extension.WrappersMP
    if execution_mode == "Experimental bypass":
        try:
            from .DonutSafeApplyLoRAStack import _partition_bypass_targets
        except ImportError:
            from DonutSafeApplyLoRAStack import _partition_bypass_targets
        try:
            from .donut_sda_merge import SDAMergeTargets
        except ImportError:
            from donut_sda_merge import SDAMergeTargets
        targets = SDAMergeTargets.build(patched, patches)
        # Check the module actually used by each adapter, not the unused
        # primary copy. This also retains source-side quantization guards.
        for label, root, selected in targets.roots(patched.model, patched):
            components = {key: [(adapter, strength)] for key, adapter in selected.items()}
            _, regular, reasons = _partition_bypass_targets(root, set(root.state_dict()), components)
            if regular:
                raise ValueError(
                    f"SDA cannot schedule these {label} targets as Experimental bypass: "
                    f"{list(reasons.items())[:2]}. No always-on fallback was applied."
                )
        if targets.plans:
            logging.info("[Donut SDA] Hard-swap routing: %d primary / %d retained model2 adapter(s)",
                         len(targets.primary), len(targets.source))
        patched.add_wrapper_with_key(wrappers.APPLY_MODEL, SDA_WRAPPER_KEY,
                                     _ScopedSDABypass(patches, strength, targets))
    elif execution_mode == "Comfy patches":
        try:
            from .donut_sda_merge import SDAMergeTargets
        except ImportError:
            from donut_sda_merge import SDAMergeTargets
        targets = SDAMergeTargets.build(patched, patches)
        if targets.source:
            raise ValueError(
                "A Donut hard-swap merge inherits Experimental bypass. Keep that execution mode "
                "for SDA source targets; a mixed Comfy-patches override cannot hook an unused layer."
            )
        hook = comfy.hooks.WeightHook(strength_model=strength, strength_clip=0.0)
        hook.need_weight_init = False
        hook.weights = patches
        hook.weights_clip = {}
        hook.hook_keyframe = _SDAKeyframes()
        group = comfy.hooks.HookGroup()
        group.add(hook)
        positive, negative = _attach_hooks(positive, negative, group)
    else:
        raise ValueError(f"Unsupported SDA execution mode: {execution_mode}")

    patched.add_wrapper_with_key(wrappers.SAMPLER_SAMPLE, SDA_WRAPPER_KEY, _sampling_guard)
    return patched, positive, negative

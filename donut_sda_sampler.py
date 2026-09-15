"""Read-only SDA compatibility checks for stock and Bleh preset samplers.

V4 selects bleh_preset_0, not stock er_sde. Bleh stores a configured KSAMPLER
and an optional sigma override in its live preset table. Inspect that table;
never replace the preset with a fresh stock sampler, which would lose ODE
noise_scaler/s_noise/max_stage and any other configured solver options.

Only the known Bleh preset wrapper is transparent here. Arbitrary wrappers,
adaptive solvers and sigma overrides are NOT implicitly declared compatible.
"""
from functools import partial
import math
import sys


def _options(sampler):
    options = getattr(sampler, "extra_options", {})
    if not isinstance(options, dict):
        raise ValueError("SDA expected a sampler extra_options dictionary.")
    return dict(options)


def _bleh_preset(function):
    """Resolve the installed node's registry, without importing Bleh or guessing its package name."""
    import nodes

    setter = nodes.NODE_CLASS_MAPPINGS.get("BlehSetSamplerPreset")
    module = sys.modules.get(getattr(setter, "__module__", ""))
    wrapper = getattr(module, "bleh_sampler_preset_wrapper", None)
    if (setter is None or getattr(module, "BlehSetSamplerPreset", None) is not setter
            or not isinstance(function, partial) or function.func is not wrapper):
        raise ValueError(
            "SDA cannot verify this custom sampler wrapper. Only the installed "
            "BlehSetSamplerPreset wrapper around a supported stock solver is supported."
        )
    if len(function.args) != 1 or function.keywords:
        raise ValueError("SDA cannot verify this Bleh preset binding; expected one positional preset index.")
    index = function.args[0]
    table = getattr(module, "BLEH_PRESET", None)
    count = getattr(module, "BLEH_PRESET_COUNT", None)
    if (type(index) is not int or type(count) is not int
            or not isinstance(table, (list, tuple))
            or not 0 <= index < min(count, len(table))):
        raise ValueError(f"SDA cannot resolve Bleh preset index {index!r}.")
    label = f"bleh_preset_{index}"
    entry = table[index]
    if entry is None:
        raise ValueError(
            f"SDA: {label} has not been registered. Ensure BlehSetSamplerPreset "
            "runs upstream of the sampler (V4 already wires it through the model)."
        )
    if not isinstance(entry, (list, tuple)) or len(entry) != 2:
        raise ValueError(f"SDA: unexpected {label} registry layout; no sampler was substituted.")
    child, override_sigmas = entry
    if override_sigmas is not None:
        # Bleh replaces sigmas inside its solver call, after Comfy's initial
        # noise scaling and sample_sigmas publication. Our gate would see the
        # wrong schedule. Do not silently ignore or accept this advanced option.
        raise ValueError(
            f"SDA: {label} has override_sigmas_opt connected. This override is not "
            "supported by the SDA gate; keep the selected scheduler and disconnect "
            "only the override. V4's default beta schedule needs no override."
        )
    return label, child


def inspect_sda_sampler(sampler, supported_names):
    """Return a diagnostic route and effective options; leave execution untouched.

    Resolve the registry anew for every run. A preset slot is mutable and may
    later contain another solver. Kernel identity is checked against Comfy's
    installed functions, not a __name__ copied by functools.update_wrapper.
    """
    from comfy import samplers

    kernels = {
        name: getattr(samplers.k_diffusion_sampling, "sample_" + name, None)
        for name in supported_names
    }
    route, seen = [], set()
    effective_options = {}
    current = sampler
    for _ in range(16):
        if id(current) in seen:
            raise ValueError("SDA detected a cycle in the Bleh sampler preset chain.")
        seen.add(id(current))
        options = _options(current)
        duplicates = effective_options.keys() & options.keys()
        if duplicates:
            # Bleh calls f(**child_options, **outer_kwargs); duplicate keys are
            # an error, not an override. Preserve that contract rather than
            # inventing a precedence and running different solver settings.
            raise ValueError(f"SDA: duplicate options in Bleh sampler chain: {sorted(duplicates)}")
        effective_options.update(options)
        function = getattr(current, "sampler_function", None)
        for name, kernel in kernels.items():
            if kernel is not None and function is kernel:
                try:
                    churn = float(effective_options.get("s_churn", 0.0))
                except (TypeError, ValueError) as exc:
                    raise ValueError("SDA requires finite s_churn=0.") from exc
                if not math.isfinite(churn) or churn != 0.0:
                    raise ValueError("SDA does not support Euler churn; use s_churn=0.")
                return " -> ".join((*route, name)), effective_options
        if not isinstance(function, partial):
            received = getattr(function, "__name__", type(function).__name__)
            prefix = " -> ".join(route)
            raise ValueError(
                f"SDA cannot schedule {prefix + ' -> ' if prefix else ''}{received!s}. "
                f"Supported underlying solvers: {', '.join(supported_names)}. "
                "The selected sampler and scheduler were not changed."
            )
        label, current = _bleh_preset(function)
        route.append(label)
    raise ValueError("SDA sampler preset nesting exceeds the supported depth (16).")

"""SDA target routing for Donut's exact model2 forward swaps.

Store module paths, never the loader's physical layers or a source patcher.
At inference the merge injection binds to the sampling clone's retained source;
SDA must resolve the same source. Partial blends remain primary-model targets.
No weights, merge ratios, source attachments or persistent injections are changed.
"""
from dataclasses import dataclass

try:
    from .donut_krea2_merge_serialization import (
        KREA2_MERGE_INJECTION_KEY, get_krea2_merge_bypass_info,
    )
except ImportError:
    from donut_krea2_merge_serialization import (
        KREA2_MERGE_INJECTION_KEY, get_krea2_merge_bypass_info,
    )


def checked_merge_info(model):
    """Validate a real swap plan, including falsey composable injection lists."""
    try:
        info = get_krea2_merge_bypass_info(model)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid SDA module-swap metadata: {error}") from error
    if info is None:
        return None
    source, plans, _ = info
    injections = getattr(model, "injections", {})
    # bool(list) is deliberately False on Donut's _ComposableInjectionList.
    entries = injections.get(KREA2_MERGE_INJECTION_KEY)
    if not isinstance(entries, (list, tuple)) or len(entries) == 0:
        raise ValueError("SDA module-swap plans have no runtime merge injection.")
    if getattr(source, "model", None) is None or source.model is getattr(model, "model", None):
        raise ValueError("SDA module-swap source must be a distinct retained model.")
    if KREA2_MERGE_INJECTION_KEY in getattr(source, "injections", {}):
        # Donut's merge builder rejects nested injected sources too. Do not
        # infer effective ownership from a hand-edited, unsupported merge tree.
        raise ValueError("SDA encountered a nested module-swap source; rebuild this merge with Donut Model Merge Krea2.")
    diffusion = getattr(source.model, "diffusion_model", None)
    if (getattr(diffusion, "txtlayers", None) != 12
            or getattr(diffusion, "txtdim", None) != 2560
            or not hasattr(diffusion, "txtfusion") or not hasattr(diffusion, "blocks")):
        raise ValueError("SDA module-swap source is not a compatible Krea2 model.")
    if hasattr(diffusion, "_orig_mod") or getattr(diffusion, "_compiled_call_impl", None) is not None:
        raise ValueError("Disable torch.compile on the SDA module-swap source as well as the primary model.")
    return info


def _module(root, path):
    value = root
    try:
        for part in path.split("."):
            value = value[int(part)] if part.isdigit() else getattr(value, part)
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise ValueError(f"SDA cannot resolve the live module {path!r}.") from error
    return value


@dataclass(frozen=True)
class SDAMergeTargets:
    """Disjoint adapter maps and the clone-persistent swap plan (no roots)."""
    primary: dict
    source: dict
    plans: tuple = ()

    @classmethod
    def build(cls, model, patches):
        info = checked_merge_info(model)
        if info is None:
            return cls(dict(patches), {})
        source, plans, _ = info
        swapped = {path for path, _key, _ratio in plans}
        primary, retained = {}, {}
        for key, adapter in patches.items():
            # Current scheduled bypass supports whole weight adapters only.
            # Offset/function patch keys must not be silently attached to an
            # entire linear or to the wrong side of a swap.
            if not isinstance(key, str) or not key.endswith(".weight"):
                raise ValueError(f"SDA module-swap routing requires a whole weight target, got {key!r}.")
            path = key[:-7]
            destination = retained if path in swapped else primary
            destination[key] = adapter
            live = _module(source.model if path in swapped else model.model, path)
            if path in swapped:
                unused = _module(model.model, path)
                if live is unused:
                    raise ValueError(f"SDA module-swap target aliases its unused primary layer: {path}")
                for dimension in ("in_features", "out_features"):
                    if getattr(live, dimension, None) != getattr(unused, dimension, None):
                        raise ValueError(f"SDA module-swap target has incompatible {dimension}: {path}")
        return cls(primary, retained, tuple(plans))

    def roots(self, model_root, patcher=None, *, runtime=False):
        """Bind only to this run's primary and retained model2 modules."""
        source_root = None
        if self.plans:
            if patcher is None:
                patcher = getattr(model_root, "current_patcher", None)
            if patcher is None or getattr(patcher, "model", None) is not model_root:
                raise ValueError("SDA cannot resolve the active module-swap sampling patcher.")
            info = checked_merge_info(patcher)
            if info is None or tuple(info[1]) != self.plans:
                raise ValueError("SDA module-swap plan changed after target validation; queue the workflow again.")
            if runtime and not getattr(patcher, "is_injected", False):
                raise ValueError("SDA module-swap injection is not active on the sampling model.")
            source_root = info[0].model
        groups = []
        if self.primary:
            groups.append(("primary", model_root, self.primary))
        if self.source:
            groups.append(("retained model2", source_root, self.source))
        return groups

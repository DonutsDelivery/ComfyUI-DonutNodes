"""Opt-in NAG text-energy experiment. Default is no change.

Two explicit flags on the Donut fusion budget, both default off:

* ``nag_text_energy_compensation`` — blend NAG's two tap streams toward the
  per-item RMS midpoint, weighted by NAG alpha, before ``txtfusion`` runs.
* ``nag_batch_txtfusion`` — run equal-length streams through one
  ``txtfusion`` call; unequal lengths always use the original two calls.

Missing configuration leaves tensors and the upstream NAG forward untouched.
This module must not replace ``krea2_nag_forward`` process-wide; the
experimental forward is attached only to the Donut-created NAG clone by
:func:`install_donut_nag_experiment` (see ``krea2_nag_integration``).
"""
import torch

NAG_TEXT_ENERGY_COMPENSATION = "nag_text_energy_compensation"
NAG_BATCH_TXTFUSION = "nag_batch_txtfusion"
NAG_TEXT_ENERGY_MAX_SCALE = "nag_text_energy_max_scale"
_DEFAULT_MAX_SCALE = 4.0

# Wrapper key used on the Donut NAG clone. Removing the upstream
# krea2-nag wrapper key (nodes.py WRAPPER_KEY) is what keeps the experiment
# model-local: the standalone NAG node and every other graph keep the
# untouched upstream forward.
_EXPERIMENT_WRAPPER_KEY = "donut_nag_text_energy_experiment"
_UPSTREAM_WRAPPER_KEY = "krea2_normalized_attention_guidance"


def _fusion_budget(transformer_options):
    options = transformer_options or {}
    budget = options.get("donut_krea2_fusion_budget")
    return budget if isinstance(budget, dict) else {}


def _item_rms(value):
    dims = tuple(range(1, value.ndim))
    if not dims:
        return value.float().square().sqrt().clamp_min(1e-6)
    return value.float().square().mean(dim=dims, keepdim=True).sqrt().clamp_min(1e-6)


def _nudge_tap_energy(positive_context, negative_context, nag_alpha, transformer_options):
    """Opt-in per-item RMS blend toward the pair midpoint.

    Requires ``nag_text_energy_compensation=True`` on the fusion budget.
    Absent budget, a false flag, or alpha 0 returns the original objects.
    ``nag_match_taps`` does not enable this. Scale is clamped so a near-zero
    stream cannot explode.
    """
    budget = _fusion_budget(transformer_options)
    if budget.get(NAG_TEXT_ENERGY_COMPENSATION) is not True:
        return positive_context, negative_context
    blend = min(1.0, max(0.0, float(nag_alpha)))
    if blend == 0.0:
        return positive_context, negative_context
    pos_rms = _item_rms(positive_context)
    neg_rms = _item_rms(negative_context)
    mid = (pos_rms + neg_rms) * 0.5
    max_scale = float(budget.get(NAG_TEXT_ENERGY_MAX_SCALE, _DEFAULT_MAX_SCALE))
    if not (max_scale > 1.0):
        max_scale = _DEFAULT_MAX_SCALE
    pos_scale = ((1.0 - blend) + blend * (mid / pos_rms)).clamp(1.0 / max_scale, max_scale)
    neg_scale = ((1.0 - blend) + blend * (mid / neg_rms)).clamp(1.0 / max_scale, max_scale)
    return (
        (positive_context.float() * pos_scale).to(dtype=positive_context.dtype),
        (negative_context.float() * neg_scale).to(dtype=negative_context.dtype),
    )


def _fused_text(model, positive_context, negative_context, transformer_options, nag_alpha=0.0):
    """Fuse NAG text. Default is the upstream two-call path.

    Equal-length batching is only used when ``nag_batch_txtfusion`` is True.
    Unequal sequence lengths always use two calls; there is no discarded warmup.
    """
    def run(ctx):
        return model.txtmlp(
            model.txtfusion(ctx, mask=None, transformer_options=transformer_options)
        )

    positive_context, negative_context = _nudge_tap_energy(
        positive_context, negative_context, nag_alpha, transformer_options,
    )
    budget = _fusion_budget(transformer_options)
    if (
        budget.get(NAG_BATCH_TXTFUSION) is True
        and positive_context.shape[1:] == negative_context.shape[1:]
    ):
        stacked = torch.cat((positive_context, negative_context), dim=0)
        return run(stacked).chunk(2, dim=0)
    return run(positive_context), run(negative_context)


def ensure_nag_txtfusion_is_batched():
    """No-op. Do not replace upstream NAG forwards process-wide."""
    return


def _repeat_batch(tensor, batch):
    import comfy.utils

    if tensor.shape[0] == batch:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.expand(batch, *tensor.shape[1:])
    return comfy.utils.repeat_to_batch_size(tensor, batch)


def donut_nag_forward(
    model,
    x,
    timesteps,
    context,
    negative_context,
    transformer_options,
    phi,
    tau,
    alpha,
    upstream,
):
    """Mirror of upstream ``krea2_nag_forward`` with the text stage swapped.

    The only intended difference is ``_fused_text``: tap-energy compensation
    and equal-length batching happen before ``txtfusion``. Everything after
    the two text streams (blocks, ROPE, guidance, output crop) calls the
    upstream helpers, so upstream math is not forked.
    """
    from einops import rearrange
    from comfy.ldm.flux.layers import timestep_embedding

    _nag_block = upstream._nag_block

    temporal = x.ndim == 5
    temporal_shape = None
    if temporal:
        x, temporal_shape = _flatten_temporal_latent(x)
    elif x.ndim != 4:
        raise RuntimeError(f"Krea2 NAG expected a 4D or 5D latent, got rank {x.ndim}.")

    bs, _, h_orig, w_orig = x.shape
    patch_size = model.patch
    positive_context = model._unpack_context(context)
    negative_context = _repeat_batch(negative_context, bs).to(context)
    negative_context = model._unpack_context(negative_context)

    image, image_pos, h_tokens, w_tokens = model.process_img(x)
    image_tokens = image.shape[1]
    image = model.first(image)

    t = model.tmlp(timestep_embedding(timesteps, model.tdim).unsqueeze(1).to(image.dtype))
    tvec = model.tproj(t)

    positive_text, negative_text = _fused_text(
        model, positive_context, negative_context, transformer_options, nag_alpha=alpha,
    )

    positive_len = positive_text.shape[1]
    negative_len = negative_text.shape[1]
    device = image.device
    positive_text_pos = torch.zeros(bs, positive_len, 3, device=device, dtype=torch.float32)
    negative_text_pos = torch.zeros(bs, negative_len, 3, device=device, dtype=torch.float32)
    positive_freqs = model.pe_embedder(torch.cat((positive_text_pos, image_pos), dim=1))
    negative_freqs = model.pe_embedder(torch.cat((negative_text_pos, image_pos), dim=1))

    options = dict(transformer_options)
    options["total_blocks"] = len(model.blocks)
    options["block_type"] = "single"
    options["img_slice"] = [positive_len, positive_len + image.shape[1]]

    for index, block in enumerate(model.blocks):
        options["block_index"] = index
        positive_text, negative_text, image = _nag_block(
            block,
            positive_text,
            negative_text,
            image,
            tvec,
            positive_freqs,
            negative_freqs,
            phi,
            tau,
            alpha,
            options,
        )

    combined = torch.cat((positive_text, image), dim=1)
    final = model.last(combined, t)
    output = final[:, positive_len:positive_len + image_tokens]
    output = rearrange(
        output,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=h_tokens,
        w=w_tokens,
        ph=patch_size,
        pw=patch_size,
        c=model.channels,
    )
    output = output[:, :, :h_orig, :w_orig]
    if temporal:
        output = _restore_temporal_latent(output, temporal_shape)
    return output


def _nag_module():
    """Import the installed krea2-nag ``krea2_nag`` module.

    ComfyUI may register the pack under ``custom_nodes.krea2-nag`` or a bare
    name depending on version, so resolve it from the registered node class's
    parent package instead of guessing the module key.
    """
    import importlib
    import sys

    import nodes

    cls = nodes.NODE_CLASS_MAPPINGS.get("Krea2NormalizedAttentionGuidance")
    if cls is not None and "." in cls.__module__:
        parent = cls.__module__.rsplit(".", 1)[0]
        return importlib.import_module(parent + ".krea2_nag")
    for name in ("custom_nodes.krea2-nag", "krea2-nag"):
        if name in sys.modules:
            return importlib.import_module(name + ".krea2_nag")
    raise RuntimeError(
        "NAG text-energy experiment requires the krea2-nag pack to be loaded."
    )


def _experiment_is_active(transformer_options, state):
    if state.alpha == 0.0 or state.phi == 0.0:
        return False
    sigmas = transformer_options.get("sigmas")
    if sigmas is None:
        return True
    return bool(torch.all((sigmas >= state.sigma_end) & (sigmas <= state.sigma_start)).item())


def _nag_transformer_options(args, kwargs):
    options = kwargs.get("transformer_options")
    if isinstance(options, dict):
        return options
    return next((value for value in reversed(args[3:]) if isinstance(value, dict)), {})


def _attention_hooks(transformer_options):
    patches = (transformer_options or {}).get("patches", {})
    if not isinstance(patches, dict):
        return ()
    return tuple(name for name in ("attn1_patch", "attn1_output_patch") if patches.get(name))


def _reject_unsupported_attention_hooks(transformer_options):
    hooks = _attention_hooks(transformer_options)
    if hooks:
        raise RuntimeError(
            "Krea2 NAG cannot preserve ComfyUI attention hooks "
            f"({', '.join(hooks)}). NAG was stopped instead of silently skipping "
            "those attention patches. Disable the conflicting attention patch or NAG."
        )


def _flatten_temporal_latent(value):
    """Flatten [B,C,F,H,W] in frame-major order and return its inverse shape."""
    if not torch.is_tensor(value) or value.ndim != 5:
        return value, None
    batch, channels, frames, height, width = value.shape
    flat = value.movedim(2, 1).reshape(batch * frames, channels, height, width)
    return flat, (batch, frames)


def _restore_temporal_latent(value, shape):
    if shape is None or not torch.is_tensor(value) or value.ndim != 4:
        return value
    batch, frames = shape
    if value.shape[0] != batch * frames:
        raise RuntimeError(
            "Krea2 NAG changed the flattened frame batch size; refusing to return "
            "a misordered temporal latent."
        )
    return value.reshape(batch, frames, *value.shape[1:]).movedim(1, 2)


def _call_remaining_wrappers(executor, terminal, *args, **kwargs):
    """Run later DIFFUSION_MODEL wrappers around a NAG terminal forward."""
    import comfy.patcher_extension

    remaining = comfy.patcher_extension.WrapperExecutor.new_class_executor(
        terminal,
        executor.class_obj,
        executor.wrappers,
        idx=executor.idx + 1,
    )
    return remaining.execute(*args, **kwargs)


def _state_from_wrapper(wrapper):
    state = getattr(wrapper, "keywords", None)
    if isinstance(state, dict):
        state = state.get("state")
        if state is not None:
            return state
    for cell in getattr(wrapper, "__closure__", ()) or ():
        try:
            candidate = cell.cell_contents
        except ValueError:
            continue
        if all(hasattr(candidate, name) for name in (
            "negative_context", "phi", "tau", "alpha", "sigma_start", "sigma_end",
        )):
            return candidate
    return None


def _composable_nag_wrapper(original, state, upstream):
    """Adapt upstream NAG so its active terminal forward keeps later wrappers."""
    if getattr(original, "_donut_nag_composable", False):
        return original

    from types import SimpleNamespace

    def wrapper(executor, *args, **kwargs):
        if len(args) < 3:
            return original(executor, *args, **kwargs)
        options = _nag_transformer_options(args, kwargs)
        if not upstream._nag_is_active(options, state):
            return executor(*args, **kwargs)
        _reject_unsupported_attention_hooks(options)

        def terminal(*forward_args, **forward_kwargs):
            _reject_unsupported_attention_hooks(
                _nag_transformer_options(forward_args, forward_kwargs),
            )
            terminal_executor = SimpleNamespace(class_obj=executor.class_obj)
            adjusted_args = forward_args
            temporal_shape = None
            if forward_args:
                flattened, temporal_shape = _flatten_temporal_latent(forward_args[0])
                if temporal_shape is not None:
                    adjusted_args = (flattened, *forward_args[1:])
            result = original(terminal_executor, *adjusted_args, **forward_kwargs)
            return _restore_temporal_latent(result, temporal_shape)

        return _call_remaining_wrappers(executor, terminal, *args, **kwargs)

    wrapper._donut_nag_composable = True
    wrapper._donut_original_nag_wrapper = original
    wrapper._donut_nag_state = state
    return wrapper


def install_nag_wrapper_composition(patched, key):
    """Make installed upstream NAG wrappers composable on this model clone."""
    model_options = getattr(patched, "model_options", {}) or {}
    transformer_options = model_options.get("transformer_options", {})
    owners = (getattr(patched, "wrappers", None),
              transformer_options.get("wrappers") if isinstance(transformer_options, dict) else None)
    if not any(isinstance(owner, dict) and owner for owner in owners):
        return patched

    import comfy.patcher_extension

    wrapper_type = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    tables = []
    for owner in owners:
        if not isinstance(owner, dict):
            continue
        table = owner.get(wrapper_type)
        if isinstance(table, dict) and key in table and all(table is not old for old in tables):
            tables.append(table)
    if not tables:
        return patched

    upstream = None
    replacements = {}
    for table in tables:
        current = table[key]
        entries = list(current) if isinstance(current, (list, tuple)) else [current]
        wrapped_entries = []
        for entry in entries:
            if getattr(entry, "_donut_nag_composable", False):
                wrapped_entries.append(entry)
                continue
            state = _state_from_wrapper(entry)
            if state is None:
                # Other versions and test doubles can register opaque callables.
                # Leave those untouched rather than guessing at their contract.
                wrapped_entries.append(entry)
                continue
            if upstream is None:
                upstream = _nag_module()
            wrapped_entries.append(_composable_nag_wrapper(entry, state, upstream))
        if isinstance(current, tuple):
            replacements[id(table)] = (table, tuple(wrapped_entries))
        elif isinstance(current, list):
            replacements[id(table)] = (table, wrapped_entries)
        else:
            replacements[id(table)] = (table, wrapped_entries[0])
    for table, replacement in replacements.values():
        table[key] = replacement
    return patched


def install_donut_nag_experiment(patched, *, nag_negative, phi, tau, alpha,
                                 sigma_start, sigma_end):
    """Attach the experimental forward to this Donut NAG clone only.

    Reads both flags from the clone's fusion budget. With both flags absent
    or false the model object is returned unchanged (upstream wrapper stays).
    """
    options = getattr(patched, "model_options", None) or {}
    budget = (options.get("transformer_options") or {}).get("donut_krea2_fusion_budget") or {}
    wants_energy = budget.get(NAG_TEXT_ENERGY_COMPENSATION) is True
    wants_batch = budget.get(NAG_BATCH_TXTFUSION) is True
    import comfy.patcher_extension

    wrapper_type = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    existing = (getattr(patched, "wrappers", {}) or {}).get(wrapper_type, {})
    if "krea2_edit_normalized_attention_guidance" in existing:
        # The combined edit forward is a different upstream function. Keep it,
        # but adapt its NAG branch to compose with later wrappers.
        return install_nag_wrapper_composition(
            patched, "krea2_edit_normalized_attention_guidance",
        )
    if not (wants_energy or wants_batch) or not nag_negative:
        return install_nag_wrapper_composition(patched, _UPSTREAM_WRAPPER_KEY)
    from types import SimpleNamespace

    state = SimpleNamespace(
        negative_context=nag_negative[0][0],
        phi=float(phi),
        tau=float(tau),
        alpha=float(alpha),
        sigma_start=float(sigma_start),
        sigma_end=float(sigma_end),
    )
    upstream = _nag_module()

    def wrapper(executor, *args, state=state, upstream=upstream, **kwargs):
        if len(args) < 3:
            raise RuntimeError("Krea2 NAG encountered an unexpected diffusion-model signature.")
        x, timesteps, context = args[:3]
        transformer_options = kwargs.get("transformer_options")
        if not isinstance(transformer_options, dict):
            transformer_options = next(
                (value for value in reversed(args[3:]) if isinstance(value, dict)), {}
            )
        if not _experiment_is_active(transformer_options, state):
            return executor(*args, **kwargs)
        _reject_unsupported_attention_hooks(transformer_options)
        cond_or_uncond = transformer_options.get("cond_or_uncond", [0])
        if any(branch != 0 for branch in cond_or_uncond):
            raise RuntimeError(
                "Krea2 NAG currently requires CFG 1.0 (positive branch only). "
                "A mixed CFG batch would violate NAG's shared-image-query requirement."
            )
        ref_latents = kwargs.get("ref_latents")
        if ref_latents is None and len(args) >= 5 and not isinstance(args[4], dict):
            ref_latents = args[4]
        if ref_latents:
            raise RuntimeError(
                "The NAG text-energy experiment does not support reference "
                "latents/Krea2Edit; use the regular NAG path."
            )
        def terminal(*forward_args, **forward_kwargs):
            forward_x, forward_t, forward_context = forward_args[:3]
            forward_options = forward_kwargs.get("transformer_options")
            if not isinstance(forward_options, dict):
                forward_options = next(
                    (value for value in reversed(forward_args[3:]) if isinstance(value, dict)), {}
                )
            _reject_unsupported_attention_hooks(forward_options)
            return donut_nag_forward(
                executor.class_obj,
                forward_x,
                forward_t,
                forward_context,
                state.negative_context,
                forward_options,
                state.phi,
                state.tau,
                state.alpha,
                upstream,
            )

        return _call_remaining_wrappers(executor, terminal, *args, **kwargs)

    wrapper._donut_uses_upstream_nag_module = upstream
    wrapper._donut_experiment_wrapper_key = _EXPERIMENT_WRAPPER_KEY
    wrapper_type = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    patched.remove_wrappers_with_key(wrapper_type, _UPSTREAM_WRAPPER_KEY)
    patched.remove_wrappers_with_key(wrapper_type, _EXPERIMENT_WRAPPER_KEY)
    patched.add_wrapper_with_key(wrapper_type, _EXPERIMENT_WRAPPER_KEY, wrapper)
    return patched

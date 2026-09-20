"""Run NAG's positive and negative txtfusion in one call.

UncensorFix is 33 LoRAs on ``diffusion_model.txtfusion``. Upstream NAG calls
txtfusion once for the want-prompt and once for the don't-want prompt.
Comfy-patch / dynamic-VRAM bake can change those weights between the two
calls, which is the same class of leftover grain as unmatched Rebalance taps.
Concatenating the streams keeps one weight snapshot for both.
"""
import inspect

import torch


_MARK = "_donut_batch_txtfusion"


def _upstream_nag_module():
    import nodes

    cls = nodes.NODE_CLASS_MAPPINGS.get("Krea2NormalizedAttentionGuidance")
    if cls is None:
        return None
    node_mod = inspect.getmodule(cls)
    wrapper = getattr(node_mod, "krea2_nag_wrapper", None)
    return inspect.getmodule(wrapper) if wrapper is not None else node_mod


def _nudge_tap_energy(positive_context, negative_context, nag_alpha, transformer_options):
    """Pull NAG's two tap streams toward shared RMS in proportion to NAG alpha.

    Rebalance itself is unchanged. Alpha 0 leaves energy alone; alpha 1 meets
    in the middle. UncensorFix/TeacherFix then see milder taps as NAG gets stronger.
    """
    options = transformer_options or {}
    budget = options.get("donut_krea2_fusion_budget") or {}
    if not budget.get("nag_match_taps", True):
        return positive_context, negative_context
    blend = min(1.0, max(0.0, float(nag_alpha)))
    if blend == 0.0:
        return positive_context, negative_context
    pos_rms = positive_context.float().square().mean().sqrt().clamp_min(1e-6)
    neg_rms = negative_context.float().square().mean().sqrt().clamp_min(1e-6)
    mid = (pos_rms + neg_rms) * 0.5
    pos_scale = (1.0 - blend) + blend * (mid / pos_rms)
    neg_scale = (1.0 - blend) + blend * (mid / neg_rms)
    return (
        (positive_context.float() * pos_scale).to(dtype=positive_context.dtype),
        (negative_context.float() * neg_scale).to(dtype=negative_context.dtype),
    )


def _fused_text(model, positive_context, negative_context, transformer_options, nag_alpha=0.0):
    def run(ctx):
        return model.txtmlp(
            model.txtfusion(ctx, mask=None, transformer_options=transformer_options)
        )

    positive_context, negative_context = _nudge_tap_energy(
        positive_context, negative_context, nag_alpha, transformer_options,
    )
    if positive_context.shape[1:] == negative_context.shape[1:]:
        stacked = torch.cat((positive_context, negative_context), dim=0)
        return run(stacked).chunk(2, dim=0)
    with torch.no_grad():
        model.txtfusion(positive_context, mask=None, transformer_options=transformer_options)
    return run(positive_context), run(negative_context)


def _batched_t2i(nag, model, x, timesteps, context, negative_context, transformer_options, phi, tau, alpha):
    from einops import rearrange
    from comfy.ldm.flux.layers import timestep_embedding

    temporal = x.ndim == 5
    batch_5d = frames_5d = None
    if temporal:
        batch_5d, channels_5d, frames_5d, height_5d, width_5d = x.shape
        x = x.reshape(batch_5d * frames_5d, channels_5d, height_5d, width_5d)
    elif x.ndim != 4:
        raise RuntimeError(f"Krea2 NAG expected a 4D or 5D latent, got rank {x.ndim}.")

    bs, _, h_orig, w_orig = x.shape
    patch = model.patch
    positive_context = model._unpack_context(context)
    negative_context = nag._repeat_batch(negative_context, bs).to(context)
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
    device = image.device
    positive_text_pos = torch.zeros(bs, positive_len, 3, device=device, dtype=torch.float32)
    negative_text_pos = torch.zeros(bs, negative_text.shape[1], 3, device=device, dtype=torch.float32)
    positive_freqs = model.pe_embedder(torch.cat((positive_text_pos, image_pos), dim=1))
    negative_freqs = model.pe_embedder(torch.cat((negative_text_pos, image_pos), dim=1))

    transformer_options = transformer_options.copy()
    transformer_options["total_blocks"] = len(model.blocks)
    transformer_options["block_type"] = "single"
    transformer_options["img_slice"] = [positive_len, positive_len + image.shape[1]]

    for index, block in enumerate(model.blocks):
        transformer_options["block_index"] = index
        positive_text, negative_text, image = nag._nag_block(
            block, positive_text, negative_text, image, tvec,
            positive_freqs, negative_freqs, phi, tau, alpha, transformer_options,
        )

    combined = torch.cat((positive_text, image), dim=1)
    final = model.last(combined, t)
    output = final[:, positive_len:positive_len + image_tokens]
    output = rearrange(
        output,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=h_tokens, w=w_tokens, ph=patch, pw=patch, c=model.channels,
    )
    output = output[:, :, :h_orig, :w_orig]
    if temporal:
        output = output.reshape(
            batch_5d, frames_5d, model.channels, h_orig, w_orig
        ).movedim(1, 2)
    return output


def ensure_nag_txtfusion_is_batched():
    """Make upstream NAG fuse want/don't-want text under one txtfusion call."""
    import nodes

    cls = nodes.NODE_CLASS_MAPPINGS.get("Krea2NormalizedAttentionGuidance")
    if cls is None:
        return
    node_mod = inspect.getmodule(cls)
    wrapper = getattr(node_mod, "krea2_nag_wrapper", None)
    nag = inspect.getmodule(wrapper) if wrapper is not None else node_mod
    if nag is None:
        return
    forward = getattr(nag, "krea2_nag_forward", None)
    if forward is None or getattr(forward, _MARK, False):
        return

    def krea2_nag_forward(model, x, timesteps, context, negative_context,
                          transformer_options, phi, tau, alpha, _nag=nag):
        return _batched_t2i(
            _nag, model, x, timesteps, context, negative_context,
            transformer_options, phi, tau, alpha,
        )

    setattr(krea2_nag_forward, _MARK, True)
    nag.krea2_nag_forward = krea2_nag_forward
    for mod in (nag, node_mod):
        bound = getattr(mod, "krea2_nag_wrapper", None)
        globals_dict = getattr(bound, "__globals__", None)
        if globals_dict is not None:
            globals_dict["krea2_nag_forward"] = krea2_nag_forward

"""Optional NAG txtfusion helpers. Default is no change.

Experimental RMS compensation and equal-length batching are explicit
budget flags, both default off. Missing configuration leaves tensors and
the upstream NAG forward untouched. This module must not replace
``krea2_nag_forward`` process-wide.
"""
import torch

NAG_RMS_COMPENSATION = "nag_rms_compensation"
NAG_BATCH_TXTFUSION = "nag_batch_txtfusion"
NAG_RMS_MAX_SCALE = "nag_rms_max_scale"
_DEFAULT_MAX_SCALE = 4.0


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

    Requires ``nag_rms_compensation=True`` on the fusion budget. Absent budget,
    a false flag, or alpha 0 returns the original objects. ``nag_match_taps``
    does not enable this. Scale is clamped so a near-zero stream cannot explode.
    """
    budget = _fusion_budget(transformer_options)
    if budget.get(NAG_RMS_COMPENSATION) is not True:
        return positive_context, negative_context
    blend = min(1.0, max(0.0, float(nag_alpha)))
    if blend == 0.0:
        return positive_context, negative_context
    pos_rms = _item_rms(positive_context)
    neg_rms = _item_rms(negative_context)
    mid = (pos_rms + neg_rms) * 0.5
    max_scale = float(budget.get(NAG_RMS_MAX_SCALE, _DEFAULT_MAX_SCALE))
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

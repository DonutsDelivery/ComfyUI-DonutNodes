"""Sampler bridge to the optional krea2-nag node pack."""

import nodes
import comfy.patcher_extension

try:
    from .DonutKrea2FusionControl import prepare_nag_conditioning
except ImportError:
    from DonutKrea2FusionControl import prepare_nag_conditioning


def nag_input_types():
    return {
        "nag_enabled": ("BOOLEAN", {"default": False, "tooltip": "Apply Krea2 NAG inside sampling (requires krea2-nag). Uses CFG 1; Turbo negative conditioning stays zeroed."}),
        "nag_negative": ("CONDITIONING", {"tooltip": "Unzeroed negative prompt for NAG. Defaults to edit_negative_prompt in edit mode, otherwise negative."}),
        "nag_phi": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
        "nag_tau": ("FLOAT", {"default": 2.5, "min": 0.01, "max": 20.0, "step": 0.05}),
        "nag_alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
        "nag_sigma_start": ("FLOAT", {"default": 1000.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
        "nag_sigma_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
        "nag_ref_boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
        "nag_ref_boost_a": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
        "nag_fit_mode": (["fit", "crop (legacy)"], {"default": "fit"}),
        "nag_ref_boost_mask": ("MASK",),
    }


def apply_krea2_nag(model, negative, *, nag_enabled=False, nag_negative=None,
                    nag_phi=4.0, nag_tau=2.5, nag_alpha=0.25,
                    nag_sigma_start=1000.0, nag_sigma_end=0.0,
                    nag_ref_boost=1.0, nag_ref_boost_a=1.0,
                    nag_fit_mode="fit", nag_ref_boost_mask=None,
                    source_latent=None, vae=None, source_image=None,
                    source_image_b=None, target_latent=None):
    if not nag_enabled:
        return model
    node_id = ("Krea2EditNormalizedAttentionGuidance" if source_latent is not None
               else "Krea2NormalizedAttentionGuidance")
    node_class = nodes.NODE_CLASS_MAPPINGS.get(node_id)
    if node_class is None:
        raise RuntimeError(f"NAG requires krea2-nag. Install/enable it and restart ComfyUI (missing {node_id}).")
    # Each combined forward owns edit and NAG. Replace competing forwards on a
    # clone, preserving LoRAs and unrelated wrappers on the caller's model.
    model = model.clone()
    for key in ("donut_krea2_edit", "krea2_edit", "krea2_normalized_attention_guidance",
                "krea2_edit_normalized_attention_guidance"):
        model.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, key)
    kwargs = {}
    if source_latent is not None:
        sources = source_latent if isinstance(source_latent, (list, tuple)) else [source_latent]
        kwargs = dict(source_latent=sources[0],
                      source_latent_b=sources[1] if len(sources) > 1 else None,
                      ref_boost=nag_ref_boost, ref_boost_a=nag_ref_boost_a,
                      ref_boost_mask=nag_ref_boost_mask, fit_mode=nag_fit_mode,
                      vae=vae, source_image=source_image, source_image_b=source_image_b,
                      target_latent=target_latent)
    return node_class().patch(
        model=model, nag_negative=prepare_nag_conditioning(model, negative if nag_negative is None else nag_negative),
        phi=nag_phi, tau=nag_tau, alpha=nag_alpha,
        sigma_start=nag_sigma_start, sigma_end=nag_sigma_end, **kwargs,
    )[0]


def sampler_negative(negative, turbo_mode):
    return nodes.ConditioningZeroOut().zero_out(negative)[0] if turbo_mode else negative

"""
DonutZitLayerBlendEncode

A replacement text-encode node for Z-Image Turbo (ZiT / Lumina2) that pulls
MANY hidden layers out of the Qwen3-4B text encoder in a single forward pass
and blends them along a shallow->deep interpolation curve into one 2560-dim
conditioning the DiT can consume.

Why this exists: the stock encode keeps only one hidden state (layer_idx=-2),
so a post-hoc rebalancer has a single layer to work with. By requesting a LIST
of layers via clip.layer_idx, Comfy's encoder returns every requested layer
stacked as (B, K, seq, 2560) (see comfy/text_encoders/llama.py `only_layers`).
We then collapse K by weighted average - NOT concatenation - so the output stays
2560-dim and feeds the model unchanged.

Unlike a per-channel knob, the layer axis is *ordered*: shallow layers carry
low-level/syntactic features, deep layers carry high-level semantics. So a
2-handle interpolation (weight_shallow -> weight_deep across depth, shaped by a
curve) is a meaningful, low-knob control - the spiritual analog of Krea2's
per-layer sliders, adapted to ZiT's single-layer-output architecture.

Requires a ZiT / Lumina2 style CLIP (Qwen3-4B). With other encoders that emit a
normal 3D conditioning, it falls back to a plain encode (no blend).
"""

import logging

import torch


def _curve(t, kind):
    # t: tensor in [0, 1] from shallow(0) to deep(1)
    if kind == "linear":
        return t
    if kind == "smoothstep":
        return t * t * (3.0 - 2.0 * t)
    if kind == "ease_in":
        return t * t
    if kind == "ease_out":
        return 1.0 - (1.0 - t) * (1.0 - t)
    if kind == "cosine":
        return (1.0 - torch.cos(t * torch.pi)) * 0.5
    return t


class DonutZitLayerBlendEncode:
    CURVES = ["linear", "smoothstep", "ease_in", "ease_out", "cosine"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "weight_shallow": ("FLOAT", {
                    "default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Gain on the shallowest layers (low-level/syntactic features).",
                }),
                "weight_deep": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Gain on the deepest layers (high-level semantics). "
                               "Stock encode is roughly deep-only.",
                }),
                "curve": (cls.CURVES, {"default": "linear"}),
                "multiplier": ("FLOAT", {
                    "default": 1.0, "min": -1000.0, "max": 1000.0, "step": 0.01,
                    "tooltip": "Overall gain applied after the depth blend.",
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "Donut/conditioning"

    @staticmethod
    def _num_layers(clip):
        try:
            csm = clip.cond_stage_model
            inner = getattr(csm, getattr(csm, "clip", ""), None)
            n = getattr(inner, "num_layers", None)
            if isinstance(n, int) and n > 0:
                return n
        except Exception:
            pass
        return None

    def encode(self, clip, text, weight_shallow, weight_deep, curve, multiplier):
        n = self._num_layers(clip)
        # Request every capturable layer; invalid-high indices are simply not
        # captured by the encoder, so a generous range is safe.
        layer_list = list(range(0, (n + 1) if n else 64))

        c = clip.clone()
        c.layer_idx = layer_list
        tokens = c.tokenize(text)
        out = c.encode_from_tokens(tokens, return_pooled=True, return_dict=True)

        cond = out.pop("cond")

        if cond.dim() == 4:
            # (B, K, seq, D) -> weighted average across K (shallow -> deep)
            k = cond.shape[1]
            if k > 1:
                t = torch.linspace(0.0, 1.0, k, device=cond.device, dtype=cond.dtype)
            else:
                t = torch.zeros(1, device=cond.device, dtype=cond.dtype)
            w = weight_shallow + (weight_deep - weight_shallow) * _curve(t, curve)

            denom = w.sum()
            if torch.abs(denom) < 1e-6:
                w = torch.ones_like(w)
                denom = w.sum()

            w = (w / denom).view(1, k, 1, 1)
            blended = (cond * w).sum(dim=1)  # (B, seq, D)
            logging.info(f"[DonutZitLayerBlend] blended {k} layers "
                         f"(shallow={weight_shallow}, deep={weight_deep}, curve={curve})")
        else:
            # Non-ZiT clip returned a normal conditioning; nothing to blend.
            blended = cond
            logging.info("[DonutZitLayerBlend] encoder returned a single layer; "
                         "blend skipped (not a ZiT/Lumina2 clip?)")

        if multiplier != 1.0:
            blended = blended * multiplier

        return ([[blended, out]],)


NODE_CLASS_MAPPINGS = {
    "DonutZitLayerBlendEncode": DonutZitLayerBlendEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutZitLayerBlendEncode": "Donut ZiT Layer-Blend Encode",
}

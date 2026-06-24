"""
DonutZitConditioningRebalance

Input-side conditioning gain for Z-Image Turbo (ZiT / Lumina2) and friends.

Z-Image Turbo conditions on a single Qwen3-4B hidden state (layer_idx=-2),
shape (B, seq, 2560) - a single tensor, NOT a stack of concatenated encoder
layers. So the per-layer reweighting trick used by Krea2 rebalancers has no
axis to act on here. What DOES port is scaling the embedding that feeds
cross-attention: in Comfy's convert_cond, the tensor at index 0 of each
conditioning pair becomes `cross_attn` verbatim, so scaling it is exactly
scaling the cross-attention K/V source.

This is the input-side analog of guidance. It is NOT the same as CFG: CFG
extrapolates in output (noise) space against the unconditional branch, while
this rescales the cross-attention input (a nonlinear attention-temperature +
value-gain knob, positive branch only). It matters most for distilled turbo
models that run at CFG 1, where CFG is pinned and offers no strength control.

  - multiplier          : global gain on the embedding
  - per_channel_weights : optional gains spread as contiguous bands across the
                          feature channels (the closest analog to per-layer
                          weighting available on a single-layer conditioning)
"""

import logging

import torch


class DonutZitConditioningRebalance:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "multiplier": ("FLOAT", {
                    "default": 1.0, "min": -1000.0, "max": 1000.0, "step": 0.01,
                    "tooltip": "Global gain on the text embedding fed to cross-attention. "
                               ">1 strengthens prompt adherence; the input-side analog of "
                               "guidance for CFG-1 turbo models. 1.0 = no change.",
                }),
            },
            "optional": {
                "per_channel_weights": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Optional comma/semicolon list of gains spread as contiguous "
                               "bands across the feature channels (e.g. 2560 dims). Empty = "
                               "uniform. Multiplies on top of multiplier.",
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "rebalance"
    CATEGORY = "Donut/conditioning"

    @staticmethod
    def _parse_weights(s):
        if not s:
            return None
        parts = [p.strip() for chunk in s.split(";") for p in chunk.split(",")]
        vals = [float(p) for p in parts if p]
        return vals if vals else None

    def rebalance(self, conditioning, multiplier, per_channel_weights=""):
        weights = self._parse_weights(per_channel_weights)

        out = []
        for cond, meta in conditioning:
            new_meta = meta.copy() if isinstance(meta, dict) else meta

            if cond is None:
                out.append([cond, new_meta])
                continue

            scaled = cond * multiplier

            if weights:
                d = cond.shape[-1]
                k = len(weights)
                # band i covers a contiguous range of channels; works for any k <= d
                idx = (torch.arange(d, device=cond.device) * k // d).clamp(max=k - 1)
                wvec = torch.tensor(weights, device=cond.device, dtype=cond.dtype)[idx]
                scaled = scaled * wvec

            out.append([scaled, new_meta])

        return (out,)


NODE_CLASS_MAPPINGS = {
    "DonutZitConditioningRebalance": DonutZitConditioningRebalance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutZitConditioningRebalance": "Donut ZiT Conditioning Rebalance",
}

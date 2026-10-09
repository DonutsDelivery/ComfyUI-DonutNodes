"""Krea2 attention weighting, adapted from kijai/ComfyUI-KJNodes.

Upstream: nodes/model_optimization_nodes.py at d3cfe21625e5170126ce06fbfcfe1d88108688c3
https://github.com/kijai/ComfyUI-KJNodes
SPDX-License-Identifier: GPL-3.0-only

Changes: Donut node name/category, optional existing conditioning (retains
fusion/variance metadata), no-op at zero/unit weights, encoder/strength checks,
and lazy Comfy attention imports. Weighting math is copied from upstream.
"""
import logging
import math
import re
import types

_QWEN_IM_START, _QWEN_USER, _QWEN_NL, _QWEN_IM_END = 151644, 872, 198, 151645

def _krea2_user_content_span(ids):
    # token span of the user prompt, between '<|im_start|>user\n' and the next '<|im_end|>'
    for i in range(len(ids) - 2):
        if ids[i] == _QWEN_IM_START and ids[i + 1] == _QWEN_USER and ids[i + 2] == _QWEN_NL:
            start = i + 3
            end = start
            while end < len(ids) and ids[end] != _QWEN_IM_END:
                end += 1
            return start, end
    return None, None

def _krea2_token_ids(clip, text):
    tok = clip.tokenize(text)
    key = "qwen3vl_4b"
    return [t[0] for t in tok[key][0]]

def _find_subsequence(seq, sub, lo, hi):
    out = []
    n = len(sub)
    if n == 0:
        return out
    for i in range(lo, hi - n + 1):
        if seq[i:i + n] == sub:
            out.append(i)
    return out


def krea2_attn_forward_weight(self, x, freqs=None, mask=None, transformer_options={}):
    import torch
    from einops import rearrange
    from comfy.ldm.modules.attention import optimized_attention, attention_pytorch
    from comfy.ldm.flux.math import apply_rope

    q, k, v, gate = self.wq(x), self.wk(x), self.wv(x), self.gate(x)
    q = rearrange(q, "B L (H D) -> B H L D", H=self.heads)
    k = rearrange(k, "B L (H D) -> B H L D", H=self.kvheads)
    v = rearrange(v, "B L (H D) -> B H L D", H=self.kvheads)
    # weights: list of (pos, v_factor, k_bias). v_factor scales the token's VALUE (removal/de-emphasis; <0 flips
    # to subtract the concept). k_bias adds to the token's attention logit (emphasis: more of the image attends
    # to it -> more of the concept), which works far better for amplification than scaling the value.
    weights = transformer_options.get("krea2_token_weights")
    if weights:
        v = v.clone()
        for pos, v_factor, _ in weights:
            if v_factor != 1.0 and pos < v.shape[2]:
                v[:, :, pos] = v[:, :, pos] * v_factor
    q, k = self.qknorm(q, k)
    if freqs is not None:
        q, k = apply_rope(q, k, freqs)
    if self.kvheads != self.heads:
        rep = self.heads // self.kvheads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    bias = None
    if weights and any(kb != 0.0 for _, _, kb in weights):
        bias = q.new_zeros(1, k.shape[2])
        for pos, _, kb in weights:
            if kb != 0.0 and pos < bias.shape[1]:
                bias[:, pos] = kb
    if bias is not None:
        out = attention_pytorch(q, k, v, self.heads, mask=bias, skip_reshape=True)
    else:
        out = optimized_attention(q, k, v, self.heads, mask=mask, skip_reshape=True, transformer_options=transformer_options)
    return self.wo(out * torch.sigmoid(gate))

class Krea2WeightPatch:
    def __get__(self, obj, objtype=None):
        return types.MethodType(krea2_attn_forward_weight, obj)


_WEIGHT_PATTERN = re.compile(r"\(([^():]+):(-?\d*\.?\d+)\)")


def clean_weighted_text(text):
    """Remove weighting markup before the Qwen encoder's literal tokenizer."""
    return _WEIGHT_PATTERN.sub(lambda match: match.group(1), text)


class DonutKrea2PromptWeight:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "model": ("MODEL",),
                "text": ("STRING", {"multiline": True, "default": "", "tooltip": "Prompt with per-token weights in parentheses, e.g. (word:-1) removes/represses a concept, (word:2) emphasizes it, plain text = 1.0. Works on Krea2 where ComfyUI's normal (word:weight) does nothing (the Qwen3-VL LLM encoder ignores it). weight<1 scales the token's attention VALUE (subtracts/removes at <0); weight>1 boosts how much the image ATTENDS to the token (adds more of it). Set sampler CFG to 1.0."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05, "tooltip": "Global multiplier on the weighting effect. Effect compounds over all 28 blocks; lower if results break up, raise for a stronger effect. Removal (weight<0) is the reliable direction; emphasis (weight>1) works but is looser."}),
            },
            "optional": {"conditioning": ("CONDITIONING",)},
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    RETURN_NAMES = ("model", "conditioning")
    FUNCTION = "encode"
    CATEGORY = "Donut/conditioning"
    DESCRIPTION = "Per-token prompt weighting for Krea2 (K2) via attention value scaling. Use (word:-1) to remove a concept, (word:1.5) to emphasize one -- works through the Qwen3-VL encoder where normal weighting doesn't. Outputs the patched model + conditioning; set the sampler CFG to 1.0."
    EXPERIMENTAL = True

    def encode(self, clip, model, text, strength, conditioning=None):
        if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength) or not 0 <= strength <= 4:
            raise ValueError("Prompt weight strength must be between 0 and 4.")
        if strength == 0 and conditioning is not None:
            return model, conditioning
        terms = [(m.group(1).strip(), float(m.group(2))) for m in _WEIGHT_PATTERN.finditer(text)]
        if any(not math.isfinite(weight) for _, weight in terms):
            raise ValueError("Prompt phrase weights must be finite numbers.")
        clean = clean_weighted_text(text)

        tok = clip.tokenize(clean)
        if "qwen3vl_4b" not in tok:
            raise ValueError("Krea2 prompt weights require the Qwen3-VL text encoder.")
        key = "qwen3vl_4b"
        ids = [t[0] for t in tok[key][0]]
        cond = conditioning if conditioning is not None else clip.encode_from_tokens_scheduled(tok)
        cond_len = cond[0][0].shape[1]
        visible_start = len(ids) - cond_len            # conditioning == ids[visible_start:]
        start, end = _krea2_user_content_span(ids)
        if start is None:
            start, end = visible_start, len(ids)

        weight_pairs = []                              # (conditioning position, value multiplier)
        for phrase, w in terms:
            if w == 1.0 or strength == 0:
                continue
            if w > 1.0:
                v_factor, k_bias = 1.0, strength * (w - 1.0) * 2.0   # emphasis via attention boost
            else:
                v_factor, k_bias = 1.0 + strength * (w - 1.0), 0.0   # de-emphasis / removal via value scaling
            positions = []
            for variant in (" " + phrase, phrase):     # words usually carry a leading-space token in context
                sub = _krea2_token_ids(clip, variant)
                ps, pe = _krea2_user_content_span(sub)
                if ps is None:
                    continue
                sub = sub[ps:pe]
                matches = _find_subsequence(ids, sub, start, end)
                if matches:
                    for mi in matches:
                        positions.extend(mi + off - visible_start for off in range(len(sub)))
                    break
            if not positions:
                logging.warning(f"Krea2PromptWeight: phrase '{phrase}' not found in prompt; skipped.")
                continue
            for cp in positions:
                if 0 <= cp < cond_len:
                    weight_pairs.append((cp, v_factor, k_bias))

        if not weight_pairs:
            return (model, cond)
        logging.info(f"Krea2PromptWeight: weighting {weight_pairs}")

        model_clone = model.clone()
        diffusion_model = model_clone.get_model_object("diffusion_model")

        transformer_options = model_clone.model_options.get("transformer_options", {}).copy()
        transformer_options["krea2_token_weights"] = weight_pairs
        model_clone.model_options["transformer_options"] = transformer_options

        for idx, block in enumerate(diffusion_model.blocks):
            patched_attn = Krea2WeightPatch().__get__(block.attn, block.attn.__class__)
            model_clone.add_object_patch(f"diffusion_model.blocks.{idx}.attn.forward", patched_attn)

        return (model_clone, cond)


NODE_CLASS_MAPPINGS = {"DonutKrea2PromptWeight": DonutKrea2PromptWeight}
NODE_DISPLAY_NAME_MAPPINGS = {"DonutKrea2PromptWeight": "Donut Krea2 Prompt Weight"}

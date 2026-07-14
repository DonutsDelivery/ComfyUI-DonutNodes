"""
DonutKrea2ImageConditioning

Krea2 image+text conditioning with per-image strength and multi-image support.

Krea2 uses Qwen3-VL 4B as its CLIP text encoder. Images are embedded as
vision tokens in the prompt — there are no reference_latents consumed by
the model. Image influence comes entirely from the vision tokens.

Per-image strength uses separate CLIP encodes, blended against a text-only
baseline:

  final = text_strength × text_cond + Σ strength_i × (img_i_cond − text_cond)

strength=0 → text-only, strength=1 → full image influence, >1 → overemphasis.
text_strength=0 → no text baseline (pure image deltas), 1 → normal.

An optional VAE path stores reference_latents on the conditioning for
forward-compatibility. An optional global multiplier scales the output.

Pair this with ConditioningKrea2Rebalance for per-layer tap weighting,
and DonutWeightVectorScale for rebalance-normalized weight strings.
"""

import logging
import math

import torch

logger = logging.getLogger(__name__)

# Template from compile_edit in ComfyUI-ConditioningKrea2Rebalance
KREA2_SYS_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the key features of the input image (color, shape, size, texture, "
    "objects, background), then explain how the user's text instruction should "
    "alter or modify the image. Generate a new image that meets the user's "
    "requirements while maintaining consistency with the original input where "
    "appropriate.<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


class DonutKrea2ImageConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "Krea2 CLIP model (Qwen3-VL 4B)"}),
                "prompt": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": True,
                    "tooltip": "Text instruction describing the edit or generation",
                }),
                "text_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "Text influence: 0=no text baseline, 1=full, >1=overemphasis",
                }),
                "rebalance": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Reboost blended conditioning to match the requested "
                               "text_strength magnitude. When lowering image strength, "
                               "text gets proportionally louder so overall CFG stays constant.",
                }),
            },
            "optional": {
                "image1": ("IMAGE", {"tooltip": "First reference image"}),
                "strength1": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "Image1 influence: 0=text-only, 1=full, >1=overemphasis",
                }),
                "image2": ("IMAGE", {"tooltip": "Second reference image"}),
                "strength2": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                }),
                "image3": ("IMAGE", {"tooltip": "Third reference image"}),
                "strength3": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                }),
                "vae": ("VAE", {
                    "tooltip": "Optional VAE. Stores reference_latents on the "
                               "conditioning for forward-compatibility.",
                }),
                "multiplier": ("FLOAT", {
                    "default": 1.0,
                    "min": -1000.0,
                    "max": 1000.0,
                    "step": 0.01,
                    "tooltip": "Global gain on the conditioning tensor. "
                               "1.0 = unchanged.",
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "Donut/conditioning"

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_one(clip, prompt, images):
        """Tokenize + encode prompt + images through Krea2 CLIP."""
        image_prompt = ""
        if images:
            for i in range(len(images)):
                image_prompt += (
                    f"Picture {i+1}: "
                    "<|vision_start|><|image_pad|><|vision_end|>"
                )
        full_prompt = image_prompt + prompt
        tokens = clip.tokenize(
            full_prompt, images=images, llama_template=KREA2_SYS_TEMPLATE,
        )
        return clip.encode_from_tokens_scheduled(tokens)

    @staticmethod
    def _encode_ref_latent(vae, image, strength):
        """VAE-encode an image for reference_latents, scaled by strength."""
        if vae is None:
            return None
        samples = image.movedim(-1, 1)  # NHWC → NCHW
        pixel_budget = 1024 * 1024
        src_pixels = samples.shape[3] * samples.shape[2]
        if src_pixels > pixel_budget:
            scale_by = math.sqrt(pixel_budget / src_pixels)
            width = round(samples.shape[3] * scale_by / 8.0) * 8
            height = round(samples.shape[2] * scale_by / 8.0) * 8
            try:
                import comfy.utils
                samples = comfy.utils.common_upscale(
                    samples, width, height, "area", "disabled",
                )
            except ImportError:
                pass
        try:
            latent = vae.encode(samples.movedim(1, -1)[:, :, :, :3])
            return latent * strength
        except Exception:
            return None

    # ------------------------------------------------------------------
    # encode
    # ------------------------------------------------------------------

    def encode(
        self,
        clip,
        prompt,
        text_strength=1.0,
        rebalance=True,
        image1=None,
        strength1=1.0,
        image2=None,
        strength2=1.0,
        image3=None,
        strength3=1.0,
        vae=None,
        multiplier=1.0,
    ):
        pairs = [
            (image1, strength1),
            (image2, strength2),
            (image3, strength3),
        ]
        active = [(img, s) for img, s in pairs if img is not None and s != 0.0]

        # text-only baseline
        text_cond = self._encode_one(clip, prompt, images=None)

        if not active:
            out = text_cond
            if text_strength != 1.0:
                for i in range(len(out)):
                    if out[i][0] is not None:
                        out[i][0] = out[i][0] * text_strength
            ref_latents = None
        else:
            image_conds = []
            ref_latents = []

            for img, strength in active:
                img_cond = self._encode_one(clip, prompt, images=[img])
                image_conds.append((img_cond, strength))
                rl = self._encode_ref_latent(vae, img, strength)
                if rl is not None:
                    ref_latents.append(rl)

            if not ref_latents:
                ref_latents = None

            # blend: text_strength * text_cond + Σ strength_i × (image_i_cond − text_cond)
            out = []
            for pair_idx in range(len(text_cond)):
                text_tensor, text_meta = text_cond[pair_idx]
                # start from text_strength-scaled baseline
                blended = text_tensor.clone() * text_strength

                for img_cond, strength in image_conds:
                    img_tensor, _ = img_cond[pair_idx]
                    seq_len = min(blended.shape[1], img_tensor.shape[1])
                    diff = img_tensor[:, :seq_len, :] - text_tensor[:, :seq_len, :]
                    blended[:, :seq_len, :] += strength * diff

                # Reboost: scale blend to match text_strength * text_cond RMS
                if rebalance:
                    target_rms = (text_tensor.float() * text_strength).pow(2).mean().sqrt()
                    blend_rms = blended.float().pow(2).mean().sqrt()
                    if blend_rms > 1e-8:
                        blended = blended * (target_rms / blend_rms).to(blended.dtype)

                new_meta = text_meta.copy() if isinstance(text_meta, dict) else text_meta
                out.append([blended, new_meta])

        # global multiplier
        if multiplier != 1.0:
            for i in range(len(out)):
                if out[i][0] is not None:
                    out[i][0] = out[i][0] * multiplier

        # reference_latents (forward-compat)
        if ref_latents:
            try:
                import node_helpers
                out = node_helpers.conditioning_set_values(
                    out, {"reference_latents": ref_latents}, append=True,
                )
            except Exception:
                pass

        return (out,)


NODE_CLASS_MAPPINGS = {
    "DonutKrea2ImageConditioning": DonutKrea2ImageConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutKrea2ImageConditioning": "Donut Krea2 Image Conditioning",
}

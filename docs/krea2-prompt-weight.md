# Krea2 prompt weights

Adapted directly from kijai/ComfyUI-KJNodes, `nodes/model_optimization_nodes.py`
at `d3cfe21625e5170126ce06fbfcfe1d88108688c3`, under GPL-3.0 (see `licenses/krea2-prompt-weight-GPL-3.0.txt`).
The copied helpers and attention implementation retain upstream weighting math.

Use `(smile:1.5)` to emphasize a phrase and `(fog:-1)` to suppress one. Strength
ranges from zero to four; zero disables weighting. Use sampler CFG 1.0.
Connect both model and conditioning outputs. The optional conditioning input
preserves existing fusion/variance transforms instead of re-encoding them.
Only connect conditioning generated from the same cleaned text and encoder.
With DonutPromptConditioning, enable its optional `krea_prompt_weights` input:
it cleans scene encoding before variance/fusion, while output 0 retains resolved
weight markup for this node. The face and negative encoding paths are unchanged.
Text-only Krea2 image generation is supported; image editing and native image
reference conditioning are not supported by this attention replacement.

Give separate prompts separate patched model clones. In particular, do not
reuse scene token offsets for a face prompt, negative prompt or another scene.
DMC routes the copied node into the image sampler/upscale branch only.

Synthetic unit tests exercise token offsets, clone isolation, metadata retention,
strength scaling, standalone encoding and no-op/error behavior. They do not
prove visual quality or GPU compatibility; real generation remains unverified.

## Verification limits

Behavioral fixtures execute the copied attention on CPU, import the real pack
initializer, and exercise the real prompt encoder contract with synthetic CLIP
and variance modules. No user backend, profiles, prompts or model files were used.
Real ComfyUI panel queueing, saved image metadata, reload and GPU generation have
not been exercised. The existing real-Comfy variance integration suite cannot
import `nodes` in this isolated source environment. This is not an end-to-end
image-generation acceptance report. No distributed workflow JSON was changed.

# Finetuned VAE decoder — implementation review

Spacepxl Wan2.1/Qwen 2× support now follows **01 · Models → VAE**. That one
selection supplies both encoder and decoder throughout the connected graph.
The earlier separate decoder choices were removed at the user's request.
Existing VAE selection and correction settings are preserved.

## Reference implementation audit

The [model card](https://huggingface.co/spacepxl/Wan2.1-VAE-upscale2x), installed
[VAE-Utils adapter](https://github.com/spacepxl/ComfyUI-VAE-Utils/blob/4c62ea005897fafbc593d69bedb8308ec9f932fd/vae_patch.py)
and embedded JSON in the author's
[example workflow](https://github.com/spacepxl/ComfyUI-VAE-Utils/blob/4c62ea005897fafbc593d69bedb8308ec9f932fd/workflow/workflow_wan_t2i_upscale2x.png)
were inspected. The encoder is unchanged. The decoder has a 12-channel head,
unpacked into 2× RGB by pixel shuffle. The example sends identical latents to
the original and finetuned decoders and previews their outputs directly; it
does not resize the 2× result.

The author also recommends slight filtering and downsampling when the extra
resolution is unwanted. Donut implements that original-resolution use with
antialiased bilinear reduction immediately after decoding. The exact filter is
our implementation choice, not a filter specified by the reference workflow.
All external canvas dimensions therefore follow the workflow's configured size.

## Source trace

- `DonutVAELoader` inherits the stock loader's single `vae_name` widget and
  loads that checkpoint once with normal ComfyUI management/offloading.
  `prepare_vae` applies the upstream `VAEUtils_PatchWanUpscaleVAE` node only
  when the selected VAE has the Wan 16-channel latent and 12-channel RGB head.
  The adapted VAE flows through the existing connections. Ordinary VAEs pass
  through unchanged. No global ComfyUI patch, additional per-stage model or
  copied VAE architecture is added.
- The prepared VAE wraps its public `decode` and `decode_tiled` methods. After
  the upstream decoder finishes RGB unpacking, Donut checks for the expected
  2× dimensions and reduces to latent width/height times the native spatial
  compression (8). Filtering uses float32 and returns the VAE's output dtype
  on the same device. Encoder, latent layout and model weights are unchanged.
  A flag makes preparation idempotent. The native OOM fallback calls the
  upstream `decode_tiled_3d` before RGB unpacking and reduction, so it also
  reduces once.
- `DonutVAEDecode`, `DonutTiledUpscale` and face refinement also accept a
  core-loaded 2× VAE, applying the same adapter if it is not already prepared.
  Images return at the ordinary size before previews, tile blending, correction
  or face compositing consume them.
- Hires uses its ordinary `rescale_factor` without a VAE-specific divisor.
  Sampling dimensions, tile placement, overlaps, masks and debug geometry use
  the existing configured-size calculations. The tiled solver excludes the old
  unbounded full-input candidate when it exceeds the 1.1 MP tile budget.
- Face edit padding and masked paste retain their ordinary geometry. Sampling
  cycles retain their existing one-final-decode path.
- VAE correction runs the selected VAE's encode/decode and receives the same
  filtered, original-size output as other callers. No separate resize is needed.
  Per-image padding/cropping, same-size checking and float32 subtraction remain;
  Off/zero strength add no correction pass.
- `upgradeV5VaeLoaders` finds the main loader through the Models panel's actual
  `vae_name` control path. On tagged V5 workflows, that stock loader becomes
  `DonutVAELoader` with the same widget, filename, node ID, inputs, outputs and
  links. The migration runs only when the backend advertises that node and
  skips unrelated/custom loaders. The existing base-decode migration remains.
- Old `vae_decode_mode` / `vae_upscale_name` panel controls, named values and
  recognized trailing positional values are removed from the three affected
  stage types. Correction controls and values remain. No override widgets or
  model selection parameters remain in backend stage signatures.
- Model files discovery reads `DonutVAELoader.vae_name`. The catalog retains
  the pinned native checkpoint and now names the required VAE-Utils patch node.
  Distributed workflow JSON files are unchanged.

## Upstream artifacts

- Model repository: `spacepxl/Wan2.1-VAE-upscale2x` at
  `384fb7de682e60bd54b59d6eea810ca9d9993497`.
- File: `Wan2.1_VAE_upscale2x_imageonly_real_v1.safetensors`, 507,684,560 bytes.
- SHA-256: `2413554bbec24215185662d009893cf4666b8e777efece2d895e03e1a6b63e06`.
- Inspected VAE-Utils source: `4c62ea005897fafbc593d69bedb8308ec9f932fd`.

Local setup: cloned VAE-Utils at that revision into
`/home/user/Programs/ComfyUI/custom_nodes/ComfyUI-VAE-Utils` and installed the
checkpoint in that ComfyUI's `models/vae/` after checking its size and SHA-256.
The active `ComfyUI-new/ComfyUI` installation has symlinks to both. These are
local dependencies, separate from the DonutNodes repository/package.

## Validation scope

Reference and source inspection only, plus read-only inspection of the existing
failed run below. No tests, browser interaction, new generation, generated-PNG
metadata comparison, save/reload exercise or GPU quality comparison was run.
The running ComfyUI process was not restarted. Runtime correctness,
panel-to-generation behavior and visual improvement remain unverified.
Publication is recorded separately in [release-3.0.38.md](release-3.0.38.md).
No distributed workflow JSON update was made.

### Reported hires OOM and size correction

Read the user's existing failed execution `c1a732f4-50bc-464a-86be-ac767e6b6e29`
from the local history API. First hires (`1014:989`) used the selected 2× VAE,
rescale 1.5, tiled diffusion Off, NAG On and edit mode Off. Its base preview
(`912`) is 1792×2304. The submitted traceback fails during NAG Q/K normalization
in diffusion sampling. The local GPU is an RTX 4070 with 12,282 MiB VRAM.

The previous integration enlarged the configured 896×1152 base image to
1792×2304, then used a 2688×3456 hires sampling canvas (9,289,728 pixels) and
would decode to 5376×6912. An interim hires-only size adjustment still left the
base image enlarged; it has been removed.

The current integration returns a 896×1152 base image. Hires at 1.5 uses
1344×1728 sampling (2,322,432 pixels), decodes internally to 2688×3456, and
immediately reduces to 1344×1728. The sampling pixel count is reduced by four
relative to the failed run; this is not a claim of fourfold VRAM reduction.
The 2× decoder still needs memory for its own intermediate output. NAG, tiling,
denoise, correction strength and saved workflow choices were not changed.

These dimensions were traced from the recorded run, preview header and source.
No new generation, test suite or GPU benchmark was run after the correction.
Successful execution on this GPU remains unverified.

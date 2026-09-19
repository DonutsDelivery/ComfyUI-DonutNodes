# SeedVR2 in the V4 upscale stages

SeedVR2 is an optional engine in the existing `DonutTiledUpscale` stages, not a
replacement for the Krea2 generation model. Open **Generate & finish** and choose
**Upscale engine: SeedVR2** for the desired upscale stage. The default remains
**Donut**; both disabled stages and existing Donut settings keep their behavior.

After installing the node changes, restart ComfyUI and refresh the browser.
The frontend adds controls beside each stage's existing controls by resolving
its actual graph path. No node IDs, connections, stage order, or saved model
choices are rewritten. Save the workflow to retain the additional controls and
chosen engine. Standalone Donut Tiled Upscale nodes expose the same widgets.

## Models and native ComfyUI support

Use a ComfyUI build containing `SeedVR2Preprocess`, `SeedVR2Conditioning`, and
`SeedVR2PostProcessing`. Missing native support is reported only when SeedVR2
is selected. No SeedVR2 custom-node pack is required.

Install the native [Comfy-Org SeedVR2 weights](https://huggingface.co/Comfy-Org/SeedVR2):

- `models/diffusion_models/seedvr2_3b_int8_convrot.safetensors`, or
  `seedvr2_7b_int8_convrot.safetensors` in that directory.
- `models/vae/seedvr2_ema_vae_fp16.safetensors`.

3B is the initial selection. The file selectors also list installed files,
including renamed/native-compatible models. A GGUF intended for a third-party
SeedVR2 node is not interchangeable with these native loaders. This integration
does not download weights or add unverified entries to Donut's model catalog;
install these files separately. Ordinary Donut generation needs none of them.

## Behavior

The engine lives in `donut_seedvr2.py` and is selected lazily by
`donut_upscale_stage.py`. Only the selected stage's seed, scale, and resize filter
are shared. SeedVR2 uses its own model/VAE and image-derived conditioning, not
Krea LoRAs, reference conditioning, NAG, Turbo scheduling, or the Donut denoise
setting. The default native recipe is one step, CFG 1, Euler/simple, denoise 1,
color correction off, and tiled VAE encode/decode at 512 pixels with 128 overlap.
SeedVR2 steps, denoise, color correction and VAE tile size have separate controls.

The sequence is resize, native preprocessing, VAE encode, native conditioning,
sampling, VAE decode, and native post-processing. Each image in a still-image
batch is processed independently with `stage seed + image index` (uint64 wrap),
not interpreted as adjacent video frames. Dimensions are rounded to even pixels
before preprocessing. Post-processing restores the resized source alpha when
present. The second/debug output repeats the result; no diffusion-tile diagram
is meaningful for this engine.

**VAE tiling is not diffusion tiling.** Diffusion processes the complete output
canvas. There is no fixed minimum-VRAM claim or promise of OOM-free 4K/8K runs.
Models are cached per stage with file size/mtime invalidation and remain subject
to ComfyUI's model manager. Returning to the Donut engine drops that stage's
SeedVR2 resource references.

Existing downstream inpaint restoration, previews, Face Detailer and saving
remain connected. In selected-area editing, the existing composite still restores
A outside the selection; SeedVR2 is not allowed to redefine that preservation rule.
This change does not add a standalone second-pass/refiner stage or video workflow.

## Verification

```sh
python tests/test_seedvr2_stage.py
node --test tests/seedvr2_controls.test.mjs
```

Local validation: 16 CPU Python contract tests and 5 frontend logic tests passed,
plus Python/JavaScript syntax and whitespace checks. Tests cover disabled/default
behavior, lazy branch isolation, native call order, independent still batches,
seed wrapping, geometry, model errors/cache reuse, and idempotent panel controls.
Native nodes are stubbed: real weights, GPU inference, full V4 frontend execution,
and the complete repository suite were not exercised in this environment.

Before merging, run the native 3B template and this engine on the same image and
settings, test each upscale slot separately, a two-image batch, save/reload and
engine-off execution without SeedVR2 weights. Also verify selected-area editing
preserves A outside the mask through finishing. Test 7B and large-output memory
behavior separately; passing 3B is not evidence for those configurations.

Primary contracts inspected: [native SeedVR2 nodes](https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy_extras/nodes_seedvr.py)
and the [official 3B image template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/utility_seedvr2_3b_int8_upscale_image.json).

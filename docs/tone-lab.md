# Tone Lab: learned final-image correction (experimental)

`Donut Tone Lab · Learned Auto Tone` runs the global feature model exported by
Donut Tone Lab v4.0. It is inference-only: the same 169 measurements and one set
of 2,883 shared weights determine each image's gamma and brightness multiplier.
It never looks up filenames, ratings, training history, or per-image targets.
No trained aesthetic checkpoint is bundled. Keep using Tone Lab to finalize the
model; later exports can replace the JSON without changing node code.

## Install a checkpoint

1. In Tone Lab v4, choose **Export model only** (not Export session).
2. Place the JSON in `ComfyUI/models/donut_tone/`, for example
   `ComfyUI/models/donut_tone/my-tone-v4.json`. Subfolders and ComfyUI extra model
   paths configured for `donut_tone` are supported.
3. Restart ComfyUI after installing the node code, refresh model lists after
   adding a JSON, and select it under **Save images → Tone Lab · learned auto tone**.
4. Enable learned auto tone. Strength 1 uses the trained prediction; 0 is an
   exact passthrough. In between, strength scales log-gamma and log-gain toward
   identity; it does not change the feature vector or train the model.

Replacing a selected JSON at the same path invalidates the ComfyUI cache by
SHA-256 of its contents. Keep versioned filenames when reproducibility matters.
Changing the selected filename also selects the new weights. Exporting a model
from the HTML does not automatically copy it into ComfyUI's model directory.

## V5 integration

Opening the standard tagged `workflows/v5/DonutWF_v5.json` with this node pack
adds a **disabled** final-stage node and controls in the existing Save images
panel (also used in App Mode). The frontend recognizes the final preview and
its generation subgraph by structure and semantic labels, not numeric IDs:

```
Generation engine (upscales, Face Detailer, SeedVR2, final composites)
  → Tone Lab
  → existing final preview / Latest result
  → main save and secondary resized save
```

Earlier stage previews remain ungraded. Both final-save branches receive the
same graded image, before any secondary resize. Prompts, seeds, model settings,
nested graphs, and existing filenames are not changed. The existing final
preview ID remains stable. The one-time migration preserves later user choices
and does not re-add a deliberately deleted node.

This PR does not replace the distributed JSON itself: the narrowly scoped
import migration supplies the wiring. Untagged, older V4, ambiguous, malformed,
or custom postprocessing graphs are left alone. Check the browser console when
automatic wiring is skipped. For such graphs, connect the node manually after
the final image/composite and before both final preview and saving.

For a statically wired export, save the migrated graph in ComfyUI, or run:

```
node tools/prepare_tone_lab.cjs workflows/v5/DonutWF_v5.json DonutWF_v5_tone_lab.json
```

The CLI refuses to overwrite files. Before distributing a baked workflow,
perform the panel/Run/PNG/reload checks in the validation document. If publishing
that export to Civitai, manually upload **DonutWF_v5_tone_lab.json** (or the
verified saved equivalent); updating the node pack does not update Civitai's file.

## Defaults and editing

Enable is Off and model is None. Off, Strength 0, and protected editing pass the
incoming tensor through exactly and do not read a model. Explicitly enabling
correction without a valid selected model raises a clear error, not an invisible
fallback to a different algorithm. No-op predictions also leave the frame alone.

The V5 Editing output is wired into `edit_mode`. **Also grade edited images** is
Off by default, so editing/inpainting/outpainting retains its unedited-surroundings
contract. Turning it On deliberately grades the entire final image, including
preserved surroundings. In custom workflows, connect `edit_mode` explicitly if
this protection is needed. No automatic mask-only grading is implied.

The node supports RGB/RGBA IMAGE batches. Each frame is analyzed independently
using the same global model. RGB is transformed in display space as
`clamp(rgb ** gamma * gain, 0, 1)`, not linear-light exposure. Alpha is preserved,
inputs are not mutated, and output dtype/device are retained. No output dither
or 8-bit quantization is introduced. This is a still-image model; frame-to-frame
video stability has not been tested.

## Export contract and diagnostics

The loader accepts model-only JSON with version 4, type `donut-tone-model`,
algorithm `donut_feature_mlp_v4.0`, schema `donut_srgb256_features_v4.0`, analysis
`{longEdge:256, colorSpace:"srgb", opaqueAlphaMinimum:250}`, and the exact ordered
169 feature names. It validates normalization, three dense layer shapes
169→16→8→3, numeric finiteness, and weight limits before inference. Unsupported
schemas, untrained models and sessions are rejected. Only JSON within configured
model directories is read; traversal and escaping symlinks are rejected. Reads
are bounded to 1 MiB. There are no downloads, uploads, pickle loads or trainers.

The `report` STRING output and execution UI text contain the checkpoint digest,
revision, actual per-frame gamma/gain, effective strength, no-op status and
feature coverage. `noOpScore` is an **uncalibrated** score; coverage is the
fraction of features beyond three training standard deviations, not a calibrated
quality or out-of-distribution probability. The original v4 identity gate is
preserved exactly. Fewer than 16 opaque proxy samples are skipped, as in the HTML.
Model-only exports do not contain the user's training images or ratings.

## Numerical compatibility and remaining verification

Feature extraction and network inference are tested against an inference-only
JavaScript reference extracted from the v4 HTML. For identical analysis pixels,
the feature order and results agree within floating-point tolerance.

V4 uses browser canvas to build a 256-long-edge, sRGB, 8-bit analysis proxy.
Python reproduces the tested Chromium CPU PNG mipmap/bilinear path; six synthetic
PNG cases matched its proxy pixels exactly. This is NOT a promise of identical
rasterization across browsers, colour management, odd mip sizes, JPEG/WebP
codecs, or arbitrary float ComfyUI images. The full-resolution output stays
floating point, unlike the browser's 8-bit preview. For reliable final validation,
compare the actual exported checkpoint on the same images in Tone Lab and
ComfyUI, including your preferred browser and typical save formats.

See [validation record](validation/tone-lab-v4-port.md). Live ComfyUI panel → Run
→ saved PNG metadata → reload, CUDA execution, and real trained-checkpoint
image quality remain to be verified. This experimental stage is Off by default
until those checks and checkpoint selection are complete.

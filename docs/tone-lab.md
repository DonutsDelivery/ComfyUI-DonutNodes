# Tone Lab: learned final-image correction (experimental)

`Donut Tone Lab · Learned Auto Tone` applies one global v4 feature model:
169 image measurements → 16 → 8 → gamma, gain and no-change score. Its 2,883
shared weights do not look up filenames, ratings or per-image targets.

This PR now includes a trained checkpoint and the standalone browser trainer.
The node and its V5 stage remain **Off by default**. This is not a registry release.

## Use the included checkpoint

Restart ComfyUI after installing this branch, refresh model lists, and select
**donut-tone-v4-r12.json** under **Save images → Tone Lab · learned auto tone**.
Enable the stage and start with Strength 1. The checkpoint is discovered in the
node pack's `models/donut_tone/`; no manual copying or network download is needed.
Existing saved choices, including None and disabled, are not replaced.

The [checkpoint card](../models/donut_tone/README.md) documents the exact source,
digests and fitting metadata. The uploaded file reports model revision 12; the
HTML's review-round counter is a different value. No training photos or rating
history are bundled.

## Train or install your own model

Open [the standalone HTML trainer](../tools/donut_tone_lab_feature_learner_v4.html)
locally in a modern browser. On GitHub use **Download raw file**, not Save Page
on the GitHub file-view page. No ComfyUI, Python, server or external scripts are
needed for this trainer. Select images explicitly; training and image analysis
run in the browser. See the [training guide](tone-lab-training.md).

Choose **Export model only**, then put your custom JSON in
`ComfyUI/models/donut_tone/`, refresh the model list and select it. Subfolders
and extra model paths configured for `donut_tone` are supported. User-configured
roots take precedence over the bundled root for identical filenames; prefer
unique, versioned filenames to avoid accidental shadowing. Exporting from the
HTML does not automatically install a model into ComfyUI.

The loader hashes file contents, so replacing a checkpoint under the same name
invalidates cached execution. The report records its actual SHA-256. Session
exports are private training backups, not node checkpoints.

## V5 integration

Opening the standard tagged `workflows/v5/DonutWF_v5.json` adds a **disabled**
final-stage node and controls in the existing Save images panel, also used by
App Mode:

```
Generation engine (upscales, Face Detailer, SeedVR2, final composites)
  → Tone Lab
  → existing final preview / Latest result
  → main save and secondary resized save
```

Both final-save branches receive the same graded image before secondary resize.
Earlier stage previews stay ungraded. Prompts, seeds, nested graphs and existing
filenames are untouched; final-preview IDs and later user choices are preserved.
The import migration does not re-add a deliberately deleted node. Untagged,
older V4, custom, malformed or ambiguous graphs are not automatically rewired;
check the browser console, or wire the node manually before preview and saving.

This PR does not replace the distributed workflow JSON. For a statically wired
export, save the migrated graph in ComfyUI, or run:

```
node tools/prepare_tone_lab.cjs workflows/v5/DonutWF_v5.json DonutWF_v5_tone_lab.json
```

The CLI refuses to overwrite files. Before distributing a baked workflow,
perform the panel/Run/PNG/reload checks in the validation record. If publishing
that export on Civitai, manually upload **DonutWF_v5_tone_lab.json** (or the
verified saved equivalent); a node-pack update does not replace Civitai's file.

## Defaults, editing and output

Enable is Off, model is None, Strength is 1, and **Also grade edited images** is
Off. Disabled, Strength 0 and protected editing pass the exact input tensor
through without loading weights. Enabling without a valid model raises an
error rather than silently selecting a different algorithm.

V5's Editing output feeds `edit_mode`. This preserves editing/inpainting/
outpainting surroundings by default. Opting into grading edited images changes
the entire final image, including preserved surroundings. For custom graphs,
wire `edit_mode` explicitly to obtain that protection. There is no mask-only
correction implied.

Each RGB/RGBA batch frame uses the same model independently. At Strength 1 the
transform is `clamp(rgb ** gamma * gain, 0, 1)` in display space, not linear-light
exposure. Intermediate strength scales log-gamma and log-gain toward identity.
Alpha, dtype and device are preserved; inputs are not mutated and full-resolution
outputs are not quantized to 8 bits. No dither is added. Still-image use only;
video temporal stability has not been evaluated.

## Export contract and diagnostics

Model-only JSON must specify version 4, type `donut-tone-model`, algorithm
`donut_feature_mlp_v4.0`, schema `donut_srgb256_features_v4.0`, analysis
`{longEdge:256, colorSpace:"srgb", opaqueAlphaMinimum:250}` and the exact ordered
169 feature names. The loader validates normalization, 169→16→8→3 layer shapes,
finite numeric values and weight bounds. Untrained models, sessions and other
schemas are rejected. Paths are restricted to configured model roots, including
the bundled root, with traversal and escaping symlinks rejected. Reads are
bounded to 1 MiB. Inference has no network, pickle or training dependency.

The STRING report/execution UI records model SHA-256, revision, per-frame gamma/
gain, strength and no-op status. `noOpScore` is **uncalibrated**. `coverage` is
the fraction of features beyond three training standard deviations, not an
accuracy estimate. The original v4 identity gate is retained. Fewer than 16
opaque analysis samples are skipped, as in the HTML.

## Verification limits

The bundled real checkpoint is checked against the shipped HTML on 18 synthetic
RGBA proxies. Feature and prediction parity for identical proxy pixels does
not establish pixel-identical browser/ComfyUI preprocessing on arbitrary images.
The original port tests covered a specific Chromium CPU PNG resize path;
browser colour management, odd mip dimensions and JPEG/WebP decoding can differ.
ComfyUI also keeps float output, whereas the HTML preview is 8-bit.

Live ComfyUI **panel → Run → PNG metadata → reload**, CUDA and aesthetic quality
on real checkpoint images remain unverified. See the original
[port validation](validation/tone-lab-v4-port.md) and the
[bundle follow-up](validation/tone-lab-bundle.md). Keep the stage experimental
until these checks are performed with the actual generation environment.

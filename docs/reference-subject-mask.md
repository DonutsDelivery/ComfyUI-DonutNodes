# Smart subject masking for Edit Studio Reference B

Open **Edit Studio > B (Subject / identity)**. The new **Smart subject mask · B**
controls belong to that reference, not to A's existing **Paint area** selection.
Masking defaults to **Off**, so existing workflows retain their original behavior.
Restart ComfyUI and refresh the browser after installing the node changes.

## Select and refine

**Auto select subject** queues a small, output-only BiRefNet job. It does not run
the Krea generation or finishing pipeline. When complete, its soft foreground
mask becomes a **Saved mask**. Open **Refine / paint…** to paint, erase, draw
rectangles, undo, invert, clear, and switch between source, mask and cutout views.
The editor renders a reduced preview but saves operations at the original B
resolution. An empty mask shows the source so manual painting remains usable.

Grow/shrink and feather are measured in B's original pixels, before its crop.
The mask preview is the raw selection, before those adjustments. Outside the
selection, choose neutral gray, white, or black. Save the workflow after making
changes. The original B image is never destructively replaced.

Modes:

- **Off:** the original B reference path, with no mask/model requirements.
- **Auto subject:** compute/cache the native foreground mask during a normal
  queued generation. Successful runs also populate the preview for refinement.
- **Saved mask:** use the persistent selection from Auto select or the painter;
  no segmentation model is needed on subsequent runs.
- **External mask:** connect `mask_b` on Edit Studio to a mask from Grounding/SAM,
  YOLO-based segmentation, or another node. It must contain exactly one mask at
  the ORIGINAL B width/height, not A's dimensions or the cropped output size.
  White keeps the subject. ComfyUI Load Image's alpha MASK may need inversion.

This PR implements native auto foreground selection and manual refinement, not
built-in text-prompted detection, click-to-SAM inference, or automatic selection
of one named object among several. Those pipelines can supply the external MASK
input without becoming mandatory dependencies of DonutNodes.

## Native model

Auto selection requires core `LoadBackgroundRemovalModel` and `RemoveBackground`
nodes and the native [Comfy-Org BiRefNet weights](https://huggingface.co/Comfy-Org/BiRefNet):
`models/background_removal/birefnet.safetensors`. No additional custom-node pack
is required. Install the weights separately; this PR performs no automatic model
download and does not add an unverified model-catalog entry. Missing native support
or weights affects Auto only; saved, painted and external masks remain usable.

BiRefNet selects foreground and may retain multiple objects. Review the result,
especially hair, thin props, transparent materials and overlapping people. A
segmentation matte cannot guarantee Photoshop-quality edges in every image.

## Integration and persistence

`donut_reference_mask.py` extends the existing `DonutEditStudio` registration,
using the pack's existing isolated-override pattern. New widgets append after
old ones; all twelve outputs and existing graph connections retain their indices.
The new UI attaches to the existing B card. No workflow JSON rewrite is necessary.

B is neutral-composited BEFORE the existing crop/resampling and before it reaches
either the Krea grounded encoder or appearance-token/VAE reference path. A,
its inpainting selection, LoRA application and grounding schedule are untouched.
This is reference-image isolation, **not an attention mask on the LoRA**. It
removes B's background pixels; it does not guarantee that pose/lighting within
the subject cannot influence generation, or remove context still supplied by A.

Masks are content-addressed 8-bit grayscale PNGs under
`ComfyUI/user/donut/edit_subject_masks/`, with a reference ID and RGB-source
fingerprint saved in the workflow. Copy that folder AND `edit_references/` when
moving a workflow. Missing, stale, corrupt or empty selections fail explicitly;
they never silently fall back to the full unmasked B. Auto cache files can be
regenerated. No persistent GPU segmentation-model cache is retained.

The native selection node runs in ComfyUI's execution queue. HTTP endpoints only
validate/store/serve masks, never invoke GPU inference. Mask uploads accept only
Edit Studio's content-addressed references and exact source dimensions. Replacing
B while a selection is queued does not apply the stale result to the new image.
References for this feature are limited to 32 megapixels; mask uploads to 32 MiB.

## Verification

```sh
python tests/test_reference_mask.py
node --test tests/subject_mask_state.test.mjs
# Optional test dependency: Playwright and Chromium, not a runtime dependency.
python tests/subject_mask_editor_browser.py --chromium /usr/bin/chromium
```

Local validation: 21 Python CPU tests and 4 frontend state tests passed. A real
headless Chromium smoke test verified mask/invert/undo, full-resolution PNG output
from the reduced preview, source visibility when painting from scratch, Apply,
Cancel and no browser errors. Syntax/whitespace checks also passed.

ComfyUI node APIs and the parent studio are stubbed in unit tests. Actual
BiRefNet/Krea GPU inference, full V4 queue/UI execution and the full repository
suite have NOT been validated here. Before merging, test Auto select and normal
Auto mode, A+B editing, source replacement mid-queue, external SAM masks,
paint/apply/cancel, save/reload, and moved/missing mask files in a real V4 install.
Compare masking Off against the unchanged baseline with the same seed.

Primary contract inspected: [native background-removal nodes](https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy_extras/nodes_bg_removal.py).

# Edit Studio selected-area editing — development validation

Implemented locally on 2026-09-13. Not a Registry release or a GPU-quality claim.

## Research and implementation choice

- [Krea2Edit upstream](https://github.com/lbouaraba/comfyui-krea2edit): clean VAE
  reference tokens plus image-grounded instructions; optional second reference.
  Preserve this conditioning for masked edits instead of replacing it with a
  text-only inpaint path.
- [LanPaint Krea 2 example](https://github.com/scraed/LanPaint/blob/master/example_workflows/Krea2_EncodeDecode_Inpaint.json):
  inspected the actual JSON, including `LanPaint_ImageEncode`,
  `LanPaint_KSampler`, and `LanPaint_ImageDecode`. This is an alternative sampler
  approach, not the algorithm implemented here. No LanPaint code was copied and
  no new sampler/node-pack dependency was added.

Donut reuses A's already encoded reference latent as the inpaint target and sends
the painted `noise_mask` through its existing ComfyUI sampler lifecycle. A stays
available as full cropped image context, with optional B unchanged. Four explicit
composite nodes preserve A outside the selection after base decode, both upscales,
and Face Detailer. This includes the inputs to stage previews and saved results.
Mask softness is inward; at larger output sizes the preserved pixels are resized
from A, rather than newly generated upscale detail. Lossy saving can alter them.

## Validation

- Python mask, Edit Studio, and edit geometry tests: 41 passed.
- Dynamic sampler tests using installed ComfyUI imports: 12 passed, including
  forwarding the encoded A latent and mask across a target batch.
- Workflow streamlining tests: 44 passed, 1 fixture-dependent skip.
- Frontend Node suite: 125 passed, 2 fixture-dependent skips.
- Updated V4 graph passed reciprocal links, types, subgraph boundaries, cycle
  checks, and strict frontend import/export repair tests.
- Browser UI exercised in a separate local harness using the actual Edit Studio
  and mask-editor modules: opening the painter, drawing, undo, erase, clear,
  empty-selection prevention, applying, reopening, and cancel preservation.
- `git diff --check` passed.

ComfyUI had an active user generation when checked, so it was not restarted and
no GPU generation was queued. The browser harness validates controls independently
of the ComfyUI graph renderer. A real Krea 2 inpaint run and human review of edit
quality/seams remain to be performed after restart with the updated V4 JSON.

## Single-frame VAE correction — 2026-09-14

The first user run failed at `masked_edit_target`: the initial implementation
accepted only 4D latents, while image VAEs can encode stills as B,C,1,H,W.
The helper now removes only that singleton frame axis before constructing the
4D sampler target. Multi-frame latents and multiple base images remain rejected,
with the received shape included in the error. Reference tensors stay unchanged.

Mask/Studio/geometry tests: 43 passed. Installed-ComfyUI sampler tests: 12 passed,
including the actual single-frame shape through the edit sampler dispatch.
GPU generation after this correction remains unverified.

## Painter controls — 2026-09-14

Added an output-pixel seam slider with a live amber blend preview, selection
inversion, a brush/eraser cursor ring, and filled rectangle strokes. The seam
preview uses browser Gaussian blur in output-crop coordinates; it is a visual
approximation of the PIL feather, not a generated-result preview. Inversion and
rectangle strokes are persisted in the existing mask JSON. Older brush-only
masks still load. The backend applies inversion before cropping and feathering.

Browser harness checks confirmed rectangle drawing, live 48-pixel seam preview,
inversion, saved seam/rectangle/inversion values, the cursor ring while no stroke
is active, and the footer fitting in the dialog. CPU tests cover complementary
masks, preservation of the protected inverted region, both rectangle drag
directions, and erasing a rectangle. New shape/inversion execution needs a backend
restart; no workflow wiring changes are required for these painter additions.

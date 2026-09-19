# Donut Workflow changelog

## V5 · workflow redesign, outpainting and visible finishing stages

V5 is the new name for the workflow assembled after V4 Beta. It keeps V4
import compatibility while giving the substantially expanded workflow its own
release identity.

- Fixed selected-area editing becoming stuck after clearing a mask. An empty
  editor now applies as **Clear & turn off**, while enabling the main control
  without a saved selection opens the editor instead of enabling invalid state.
- Added directional outpainting canvas presets for left/right side-by-side and
  above/below layouts. They derive a grid-aligned resolution from A while
  remaining within the current megapixel budget, and apply it as Custom sizing.

## Unreleased · individual finishing previews

- Latest result now selects Base generation, First upscale, Second upscale, Face Detailer, or SeedVR2 / final image. Existing tagged V4 workflows gain the missing preview branches on import.
- Preserved inpaint/outpaint surroundings in the Face Detailer preview. Always latest retains the furthest completed stage even if preview events arrive out of order.

## Unreleased · outpainting inside the mask editor

- Added Outpaint mode with image sizing, drag placement, alignment shortcuts, automatic selection of uncovered canvas, and adjustable overlap. The complete canvas uses the existing output dimensions and megapixel budget.
- Saved placements travel with workflow/image metadata through the existing mask field. Supports legacy and independent crop modes; no widget-order or workflow rewiring changes.

## Unreleased · reliable setup and separate finishing panels

- Fixed SeedVR2 post-upscale imports omitting ComfyUI’s extra seed control, which shifted the model, VAE, denoise and color-correction values. The bundled JSON now contains the complete widget order; compatible older exports are repaired on import.
- Split Generate & finish into Generate, First upscale, Second upscale, Face detail and SeedVR2 panels, including App Mode entries. Existing generation controls and wiring are preserved.
- Reordered panels by frequency of use: setup first, finishing/save next, then editing, prompts and seed/guidance beside the result. App Mode follows the same order; the ordering migration runs once.
- Defaulted the fresh-install workflow to public Krea2 in Single model mode. Existing personal workflows keep their selected models.
- Restored ComfyUI’s native searchable LoRA dropdown on the stack node and workflow panel, replacing the datalist that initially showed only the selected filename.
- Clarified download failure counts and provider-specific authentication errors. Verified the actual Download missing flow installs SeedVR2’s model and VAE in their correct folders.

## Unreleased · isolate native reference guidance from face refinement

- Fixed native **Reference guidance** leaking into `face_positive`. The full
  generation/upscale prompt retains its reference images and vision tokens;
  Face Detailer's normal refinement prompt is now encoded from face text only.
- Kept explicit identity editing through `face_reference` / `face_reference_b`,
  face seed variance, negatives, and prompt-set selection unchanged.
- The inspected V4 generation subgraph sends the upscale result to the
  detailer's working `image` input, not its explicit reference inputs. The bug
  was in prompt conditioning, so existing V4 workflow JSONs need no rewiring.
  Install the fixed DonutNodes code and restart ComfyUI to use the change.
- Added coverage for identical prompt text, A/B and B-only references, reference
  changes, enable/disable transitions, seed variance, and prompt variants.
  Nine conditioning tests passed with mocked ComfyUI/vision boundaries; seven
  failed against the original implementation. Syntax checks passed. Full
  ComfyUI execution, Edit Studio integration, and GPU image quality were not
  validated in this run. This change has not been published to the registry.

## DonutNodes 3.0.19 · promoted inpaint control compatibility

- Fixed a blank `mask_feather` value on the V4 Image setup & editing subgraph
  that could overwrite Edit Studio's repaired value during prompt queueing.
- Normalized promoted and nested inpaint controls before serialization while
  preserving the selected mask, prompt, references, and seam width.

## DonutNodes 3.0.18 · Edit Studio metadata compatibility

- Fixed legacy Edit Studio workflow metadata that left the optional inpaint
  `mask_feather` value blank. Load, save, and queue serialization now normalize
  the inpaint controls to the backend types, and the DOM-only panel no longer
  adds a trailing positional value to PNG workflows.
- Kept existing masks, prompts, references, and seam-width settings intact
  while repairing the malformed metadata.

## DonutNodes 3.0.17 · selected-area editing

- Added **Paint area…** to Edit Studio, with brush size, erase, undo, clear,
  a crop preview, and edge softness. Applying a selection enables inpainting.
- Added a live brush/eraser size circle, **Rectangle** selections, and
  **Invert selection**. The painter's **Seam width** slider shows an amber
  preview of the inward blend and saves to the existing edge-softness setting.
- Saved masks follow A's crop and output size. Optional B remains available
  for identity guidance; whole-image editing is still a toggle away.
- Added masked sampling and preservation of A outside the selection after
  base decode, both upscale stages, and Face Detailer. Requires this updated
  workflow JSON and node code.

## V4 Beta — compared with the original V3

This comparison uses the original V3 JSON supplied by the author, not an
intermediate development file that was also named V3. The source fingerprints
and structural comparison are recorded in [comparison.json](comparison.json).

### Redesigned interface

- Replaced the spread-out loader/settings layout and rgthree labels with
  numbered Donut control cards and collapsible Advanced sections.
- Moved source loaders and generation internals into inspectable subgraphs.
- Added direct block-weight slider controls, automatic card sizing and shared
  controls across Graph and App Mode. LoRA stacking and ordering existed in V3.
- Added a consolidated latest-result panel, stage progress and final expanded
  prompt display while retaining the individual stage outputs.

### Images and editing

- Replaced the separate Load Image, editing toggle, edit-LoRA loader and
  resolution selector with **Edit Studio**.
- Added two image slots: A for the base/scene and optional B for subject/identity,
  including B-based face-reference routing.
- Added paste/drop/upload, visual crop controls and integrated preset/custom,
  reference-aspect and crop-aligned sizing.
- Added **Reference guidance** as a separate native image-conditioning path for
  generation, without an edit LoRA. It pauses while Editing is enabled.

V3 already had a single-image edit path and a face detailer. These changes are
editing/control additions, not a claim of universally smarter face detailing.

### Prompts and guidance

- Added a persistent **Wildcard library**, picker and expanded-text preview;
  explicit wildcard tokens replace the old automatic prompt-addition controls.
- Prompt editors now grow with their content, and each prompt card uses one
  shared wildcard picker that can target any of its text fields.
- Added **Prompt variants** as additional sets using the same three fields as
  Prompt 1, with blank and duplicate actions. Choose a 1-based active set and
  keep it fixed or increment it after each generation.
- Added shared seed-variance controls for general and face positives, including
  reapplication to freshly encoded edit conditioning.
- Integrated NAG controls into the sampler, upscale and face-detail paths,
  including edit-mode negative conditioning. V3 already used a standalone NAG
  node; NAG itself is not new to the workflow.

### Installation and models

- Added **Download missing** with a reviewed model catalog, size/hash checks,
  reuse of matching local files and cancellation.
- Added a collapsed **Required node packs** subgraph so Manager can discover
  internally called Krea2 Edit, NAG and seed-variance dependencies.
- Fixed DonutFaceDetailer registration depending on Impact Pack loading first.
  Dependency loading failures are distinct from image-quality improvements.
- Removed WAS and rgthree label requirements from this workflow. Seven companion
  packs remain discoverable through the normal missing-node installer.

### Saving

- Replaced WAS Image Save with **Donut Image Save**, without adding dependencies.
- Preserved the active Final folder, seed-based filenames, numbering, WebP
  quality, overwrite setting and secondary core save output.
- Retained MIT attribution for the adapted WAS filename numbering code.
- Removed unused WAS-only history, color-profile and high-bit-depth/EXR controls.

### Defaults and compatibility

- Existing V3 concepts remain: model merging/Fusion Control, LoRA stacking,
  Turbo sampling, face detailing, two upscale stages and seed planning.
- The beta is a new preset as well as a new layout; saved settings differ from
  the original V3. Carry personal settings over explicitly rather than assuming
  the same prompt/seed will reproduce a V3 image.
- The distributed configuration has editing off, first upscale on and second
  upscale off. It includes optional seed variance and integrated NAG controls.
- Requires the updated DonutNodes code shipped with this workflow. A node-pack
  release number is not the workflow's V4 version number.

### Validation

- Fresh default missing-node installation, restart and full generation passed
  with the development DonutNodes code; no WAS selection or repair override.
- Verified base generation, first upscale, two face refinements and a 1728 × 1344
  WebP save. Output matched the preceding WAS save for the same tested preset.
- Six saver tests and 34 registration/dependency-isolation tests passed.
- Editing, the disabled second upscale, model downloads and Windows/macOS were
  not exercised by that final fresh-install run. These remain beta test areas.

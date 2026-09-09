# Donut Workflow changelog

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

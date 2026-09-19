# Independent source crops

## What changes

Edit Studio and Reference Guidance keep their existing node IDs and outputs.
Each image card gains **Crop image…**, **Reset crop**, and a source-pixel crop
readout. The shared editor supports moving/resizing the rectangle and Original,
Free, or preset aspect ratios. Apply saves normalized source coordinates;
Cancel does not change the workflow. Crops are associated with an image ID and
its source dimensions. Source files are never overwritten.

Choose **Reference crop geometry → Independent crops** to use this preparation.
**Legacy output-linked** remains the schema default, so existing saved/API
workflows retain their previous geometry. The bundled V4 starter explicitly
selects Independent crops; it still starts with Editing off. Switching geometry
modes does not erase the old crop-position settings or the new rectangles.

A crop's pixel dimensions describe the retained SOURCE region. They are not a
new reference-encoder resolution. No misleading per-reference Auto/Custom
encoding-resolution controls are added in this change.

## Output canvas and reference preparation

**Generate & finish → Image size → Output canvas** owns the output relationship:

- **Follow A crop:** use A's selected aspect and the global megapixel budget.
  The existing Reference A crop-only mode instead uses the selected source
  dimensions, snapped to the grid. The entire selected region is still fitted,
  not trimmed a second time.
- **Independent output:** fit both chosen crops into the global Preset/Custom
  canvas. Reference-only sizing modes fall back to the preset in this mode.
  Auto A/B aspect choices use the corresponding selected crop's aspect.

Ordinary generation while Editing is off uses the existing output settings and
requires no images/crops. A standalone Edit Studio without a matching Generate
panel keeps an output-canvas selector locally. Follow A hides the independent
aspect/width/height controls but leaves its pixel budget available as applicable.

Example: A can retain a wide scene and B a full portrait, while the output is
landscape. B is proportionally fitted onto a neutral canvas rather than having
its head/feet center-cropped or its proportions stretched. Pixel-grid rounding
can introduce the usual subpixel-size approximation, not arbitrary anisotropic
stretching. Subject-mask background choices are respected for B's initial fit.

The edit model carries an explicit `donut_reference_fit` flag. The existing
model-resolution bridge preserves it when combining edit-LoRA and sampler
branches. Krea edit preparation then uses pixel-space fit instead of another
center crop for BOTH references before VAE encoding. Grounded conditioning
receives the same already-cropped/fitted references. The legacy path does not
change when that flag is absent.

Appearance tokens still use the existing target-sized latent grids. This does
not implement independent token-grid sizes, reduce reference memory costs, or
guarantee identity-only conditioning. Independent model-encoding resolutions
remain out of scope.

Reference Guidance has no generated canvas of its own: it supplies each crop at
its retained source dimensions to the existing native conditioning path. It
retains its existing pause behavior while Editing is enabled.

## Masks

A's selected-area mask is rasterized in original A coordinates, cropped using
A's rectangle, resized to A's fitted content dimensions, then padded using the
same offset as A. Padding is unselected, and feathering operates inward in
output pixels. The painter's seam preview uses fitted content dimensions so its
pixel scale agrees with this transform. The final preservation composite uses
this same fitted A/mask context, including after SeedVR2. Neutral letterboxing
is not an outpainting feature.

B's existing Auto/Saved/External masks stay in original B coordinates. Subject
isolation occurs before B's independent crop and fit. **Crop to subject** uses
the saved grayscale mask and an adjustable percentage padding margin. It uses
foreground values above 16/255 to avoid tiny background probabilities; review
hair and soft edges and increase padding where needed. It does not run an extra
segmentation model. Stale asynchronous crop-to-mask results are discarded if
B, the mask, or the crop changes before completion.

## Implementation ownership

- `donut_reference_geometry.py`: strict crop parsing, shared fit geometry,
  pixel/mask transforms and output sizing; no ComfyUI import.
- `donut_crop_studio.py`: appended inputs on the existing Edit/Reference Studio
  registrations. Reuses original prompt/LoRA/grounding and subject-mask logic.
- `krea2_edit_integration.py`: preserve the fit flag across model selection and
  select fit before A/B VAE encoding; no transformer/token-grid rewrite.
- `web/donut_reference_crop_geometry.js`, `..._editor.js`, `donut_reference_crops.js`:
  shared crop mathematics/editor and image-card integration.
- Existing Edit/Reference Studio UI and categorized-panel sizing adapters:
  small hooks to the new geometry; existing workflow state remains authoritative.
- `tools/prepare_independent_crops.cjs`: bake prior V4 repairs/categories and the
  new starter defaults into the actual bundled JSON, with bidirectional-link
  validation. Authoring only; it is not a runtime migration of user settings.

No new model weights, package dependencies, download mechanism, or Registry
publication is included. Existing Download missing discovery keeps using the
same node IDs and model selectors.

## Validation

Preparation ran 38 Python CPU geometry/bridge tests, 22 JavaScript geometry/
visibility tests, and 17 isolated Chromium editor/card checks. The browser
checks inline local ESM imports in a blank page and use a small fake Comfy graph;
they are not a full ComfyUI module-loading test. Original Studio/model APIs are
stubbed in the bridge tests. Source-crop math and image/mask transforms use real
PIL/NumPy/PyTorch tensors. Ten additional bundle patch-construction checks ran.

```sh
python tests/test_independent_crops.py
node --test tests/reference_crop_geometry.test.cjs
python tests/reference_crop_browser.py --chromium /path/to/chromium
```

No real-model GPU render, complete ComfyUI session, full repository regression
suite, model download, or Registry package/review was run. The assistant's
connection did not allow cloning or writing the remote repository; the complete
bundled workflow bake/validation is therefore a LOCAL preflight of the supplied
application script, not a claimed test already performed in that session.

Before approval: run landscape A/portrait B edits, source/output-size changes,
Auto and manual masks, crop-to-subject, save/reload, disabled/legacy comparisons,
Reference Guidance, and both hires passes plus SeedVR2. Check the saved pixels
outside A's selection and the source crop shown by each panel.

# V4 panel categorization and PR #66 review

Reviewed PR #66 at `5b3cf76bbb91bf76cea87d6ed408819a3d9d3bcb`.
This follow-up preserves the standalone SeedVR2 node, native LoRA picker work,
model-download discovery, existing models/settings, and CommonJS test fixes.
It does not merge the PR, change package versions, or publish a Registry build.

## Panel ownership

| Panel | Controls |
| --- | --- |
| 01 Models | Model selection, encoders/VAE, supporting models, Fusion, and a **standard UncensorFix** section; advanced merge/tap/projector controls remain separately grouped. |
| 02 LoRAs | Existing searchable installed-LoRA rows, LoRA setup, global block weights, safety/fusion settings. |
| 03 Edit Studio | A/B references, crop previews, B subject masking, A's edit selection, grounding and edit-LoRA settings. The effective output-size readout remains here. |
| 04 Prompts | The existing shared prompt/wildcard editors and variants, plus prompt-composition settings previously mixed into variance. |
| 05 Seed & guidance | Shared seed, NAG, seed variance and its advanced controls. |
| 06 Generate & finish | Global image size/aspect, batch, base sampling, AuraFlow, first Donut hires pass, face detail, second Donut hires pass, then SeedVR2 post upscale. |
| 07 Latest result | Existing previews, unchanged. |
| 08 Save images | Destination/filenames, format/compression, metadata/previews, secondary output resize/save. |

The standard panels use the same section-grid layout as panel 01. Advanced
settings remain under the relevant feature rather than a mixed collection.
Prompt editors, prompt variants, LoRA rows and shared wildcard controls retain
their existing stateful renderers. Large row/prompt/vector sections span the
panel width rather than being squeezed into narrow columns.

SeedVR2's **post upscale** enable, scale, model/VAE and color correction are
standard finishing controls. Sampling/tiling/seed details have their own
Advanced section. Existing per-stage replacement-engine controls are retained
under **alternative engine** and **SeedVR2 replacement settings**; they are not
relabeled as the post-pass. No engine or saved enable state is changed.

Global sizing controls proxy the original Edit Studio widgets, with the same
callbacks and values. They are exposed in Generate & finish for both ordinary
generation and editing. Edit Studio hides its duplicate sizing rows only after
a unique matching destination actually renders all six controls, and keeps its
live effective-size readout. Standalone studios or ambiguous copied layouts
retain their local sizing controls. No guessed association across panel families.

## Review findings corrected

1. PR #66 changed `web/donut_native_lora.js`, but the visible panel's separate
   row renderer in `web/donut_app_controls.js` still used a plain select. The
   new panel adapter makes that actual Installed LoRA control searchable too.
   It delegates commits to the original select/change handler, so catalog
   refresh, row state, hash invalidation and metadata behavior stay owned by the
   existing LoRA module. Partial/unknown filenames are not committed.
2. The shipped post-node's named settings were correct, but its positional
   `widgets_values` started with enabled/seed instead of the required
   scale/filter widgets. The narrow compatibility repair reconstructs the
   ten-element order from complete, valid named settings without changing them.
3. The final preservation composite still listed output link `1079868`, while
   the PR's link table used `1115408` for composite-to-SeedVR2. The reciprocal
   links were inconsistent, despite the earlier validation summary.
4. SeedVR2 was after the last inpaint-preservation composite. The supported V4
   shape is now repaired to **regular finishing -> SeedVR2 -> restore A outside
   the selection -> existing final consumers**. SeedVR2 remains a post-hires
   pass, but cannot be the final generative operation after preservation.

The graph repair runs before the existing generic import/link validator. It
keeps link IDs, subgraph interfaces and the inpaint cable intact, handles array
and object link formats, and runs only on tagged Donut V4 Beta workflows.
Custom/shared composite branches are reported rather than silently rewired.
Missing/invalid named settings are not coerced into new defaults.

## Module placement and migration

- `web/donut_panel_categories_model.js`: pure category/ownership definitions.
- `web/donut_panel_categories_dom.js`: panel search and sizing presentation.
- `web/donut_panel_categories.js`: panel lifecycle, live refresh and layout.
- `web/donut_seedvr2_workflow_repair.js`: narrow serialized-graph compatibility.
- `web/donut_workflow.js`: calls that repair before its existing import guards.

There is no new sampler, segmentation model, model-file downloader, or duplicate
LoRA state implementation. No GPU work is performed by these UI helpers.

**The bundled workflow JSON is not replaced by this follow-up.** The repairs
and categorization are applied when opening it in the updated frontend. Refresh
the browser, reopen the V4 workflow, then save it to persist the repaired wiring
and panel metadata. A backend-only API client does not execute frontend import
migrations; use a workflow/API prompt exported after the update.

For maintainers who need to bake the same changes into a workflow file:

```sh
node tools/organize_v4_panels.cjs workflows/v4-beta/DonutWF_v4_beta.json /tmp/DonutWF_v4_categorized.json
```

The authoring helper uses the same pure migrations and writes a NEW output file.
It refuses existing destinations, untagged workflows and ambiguous graph repair.
It is a development tool, not a runtime or Registry dependency.

## Validation executed

```sh
node --test tests/panel_categories.test.cjs
python tests/panel_categories_browser.py --chromium /usr/bin/chromium
```

**25 Node tests and 12 isolated Chromium DOM checks passed.** Node tests cover
preservation of controls/settings, live and serialized paths, idempotence,
ambiguous ownership, standard UncensorFix, per-feature grouping, legacy versus
post SeedVR2 controls, positional widget repair, reciprocal links, inpaint
ordering, the actual workflow hook's repair order, and the authoring helper.
Browser checks exercise the real DOM-adapter source: searchable row commits,
invalid input rejection, unique datalist IDs, size-control visibility and
restoration when the destination panel disappears. No page errors were observed.

These tests use representative graph fixtures and an isolated DOM, not a full
ComfyUI installation. The full shipped workflow, the entire existing repository
suite, live model downloads, real GPU rendering, and Registry packaging/review
have NOT been validated by this follow-up. Do not interpret these results as
GPU acceptance or as proof that every third-party frontend extension is compatible.

Before merging, reopen/save the actual V4 workflow in ComfyUI, check all panels
at normal and narrow widths, exercise LoRA search/refresh/reorder, and compare
editing off/on sizing. Run masked editing with SeedVR2 enabled and verify the
saved output outside A's selection (resized when output dimensions change).
Also check post-disabled baseline behavior and custom/copied subgraphs.

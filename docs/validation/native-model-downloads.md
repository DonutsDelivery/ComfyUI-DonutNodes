# Native model setup validation — 2026-09-19

Follow-up to merged PRs #63 and #64. Source baseline:
`05cc2c5fa02e007a5703e22b6df493ec92d79010`.

## Scope and module ownership

- `model_sources.json`: four pinned Comfy-Org files, upstream byte sizes and
  SHA-256 values, plus explicit native-node prerequisites.
- `web/donut_model_requirements.js`: discover the existing SeedVR2 and Auto
  subject selectors; shared pure link/location presentation for Registry.
- `donut_model_downloads.py`: accept the registered `background_removal` category
  and reject missing native core support before downloading affected models.
- `web/donut_model_downloads.js`: keep the existing explicit-click download flow
  and refresh the original Edit Studio/mask controls after model resolution.
- `distribution/registry/donut_model_downloads.js` and `tools/prepare_registry.py`:
  static catalog links, locations and integrity metadata, with no Registry
  downloader backend. Catalog generation uses the packed release, not a remote
  request or a second hand-maintained model list.

No sampler, segmentation algorithm, workflow connections, package-install
mechanism, version number, or publication status is changed.

## Tests executed

```sh
python tests/test_native_model_downloads.py
node --test tests/native_model_requirements.test.mjs
python -m py_compile donut_model_downloads.py tools/prepare_registry.py
node --check web/donut_model_requirements.js
node --check web/donut_model_downloads.js
node --check distribution/registry/donut_model_downloads.js
git diff --check
```

**21 Python tests and 18 JavaScript tests passed.** The Python tests execute the
actual downloader's verification/publication path against temporary folders and
tiny, mocked HTTP payloads for each of the four catalog file types. Tests cover
registered/custom destinations, reuse and renamed-file results, rejected hashes,
existing-file preservation, native-node preflight, HTTPS/host restrictions,
Hugging Face Xet redirect credential isolation, and duplicate requests.

Registry tests exercise the staging script with synthetic ZIP fixtures, compare
the generated static catalog with the archive's catalog, verify downloader
exclusion/manual panel replacement, and reject unsafe or stale inputs. The
fixtures do not constitute a complete release ZIP or Registry approval.

JavaScript tests cover selected engine/mask modes, nested modules, old/default
workflows, muted/bypassed graphs, 3B versus 7B, shared VAE deduplication in the
manual list, filenames in subfolders, renamed-file rebinding and stale-choice
protection. The actual Git button handler and Registry panel handler are run
with a minimal DOM/API harness. They verify names-only download requests,
control refresh, and user-clicked HTTPS links with correct save paths. This is
not a full ComfyUI browser test.

## Upstream metadata verification

The exact SHA-256 values and integer sizes were read from Comfy-Org's file pages
and Git LFS pointers, cross-checked with the commits linked in
[model setup](../native-model-setup.md). No placeholder or locally invented hash
was added. Only upstream metadata, not the complete weight files, was fetched.

## Remaining acceptance checks

No real SeedVR2/BiRefNet downloads, full V4 UI/queue session, model-weight loading,
GPU inference, complete repository suite or Registry publication was run here.
Before declaring plug-and-play acceptance:

1. On a current ComfyUI Git install with these weights absent, select SeedVR2
   3B and Auto subject, click Download missing, and confirm verified files and
   ready selectors. Run both upscaling and a mask-only selection. Repeat with 7B.
2. Repeat with shared/custom model roots, a matching renamed file, missing native
   core nodes, a pre-existing different file and an interrupted download.
3. Stage a real Registry archive using `prepare_registry.py`; confirm the
   manual panel imports its generated catalog, shows every selected link/path,
   and the automatic downloader backend/routes are absent. Install files from
   the links and run the two features. Registry review remains a separate step.

The download button prepares model files; it cannot supply unavailable native
core nodes or guarantee sufficient memory for arbitrary resolutions.

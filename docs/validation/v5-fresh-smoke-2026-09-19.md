# V5 smoke tests — 2026-09-19

Local accumulated changes, not a publication or full GPU quality certification.

## Clean environment

Cloned upstream ComfyUI commit `3c80da7f87ee359b2d06f107cb3c0797079dfbbb`
into `/tmp/donut-fresh-20260919/ComfyUI`. Created a separate Python 3.12.7
environment and installed upstream ComfyUI requirements, CPU PyTorch, and
DonutNodes requirements. ComfyUI reports 0.36.0, frontend 1.53.6.
No personal Python environment or settings were copied.

All 68 Donut nodes registered without import failures. Installed fresh upstream
copies of the seven companion packs documented in the V5 README and their
requirements. Every required workflow node type registered, including native
BiRefNet and SeedVR2 nodes. Models were symlinked to the user's existing
ComfyUI models directory at their request; no large weights were downloaded.

Full-source V5 loaded in real Chromium with zero page errors: 25 outer nodes,
47 serialized execution nodes. ComfyUI's actual `execution.validate_prompt`
accepted the complete prompt and all eight outputs with an empty error map.

## Feature checks

- 167 frontend tests passed: prompt variants, shared wildcards, clipboard
  routing, inpaint/outpaint geometry and clearing, panel ordering, native LoRA
  picker, stage previews, model discovery, SeedVR2 controls and metadata.
- Additional workflow reload/streamlining JavaScript suite: 63 passed, two skipped.
- Relevant Python suites passed: Edit Studio, inpainting, face detailer, NAG
  integration and Fusion taps, grounding schedule, independent crops, subject
  masks, SeedVR2 stages and post-pass, FP8 merge, wildcard library, model
  downloads, standalone installer, streamlining. Streamlining has one explicit
  skip. CPU/model-mock checks are not GPU inference evidence.
- Real Chromium component checks: 14 panel/picker/layout checks, 17 reference
  crop checks, and subject-mask invert/undo roundtrip, full-resolution PNG,
  Apply, Cancel and empty-mask behavior. No page errors.
- Extracted auxiliary installer ZIP and ran its Linux shell launcher with
  `--list`: correct 14-entry shared catalog, 60.91 GB total. Installer tests
  cover matching-file reuse, conflict preservation, failed checksum cleanup,
  root detection and authorization stripping on cross-host redirects.

Two obsolete test expectations were corrected: Registry tutorial list items
are separate from model list items, and the streamlining folder-path mock now
supplies the filename-list API required by SeedVR2's schema. Initial import-path
failures passed when rerun with ComfyUI on PYTHONPATH.

## Registry build

Packed a temporary Git snapshot containing all current source additions with
Comfy CLI, then ran `tools/prepare_registry.py`. The 203-entry input archive had
no standalone installer or downloader backend. The staged manual UI loaded in
the same clean ComfyUI environment, with zero browser page errors and the same
25 outer / 47 execution nodes. Registry exclusion regression tests pass.
The standalone ZIP download link still requires pushing its artifact to GitHub.
Nothing was published and Registry approval was not checked.

## Limits and observations

No full GPU generation, image-quality comparison, or repeated-run VRAM test was
performed. Windows/macOS launcher execution and real OS clipboard permissions
were not tested. Companion packs were installed from GitHub, not through Manager's
installation UI. Existing weights were reused, so this is not a full network
download test of every model or authentication provider. Custom model-root YAML
handling remains outside the standalone installer's supported behavior.

Impact/Ultralytics dependencies installed both OpenCV distributions; Donut's
checker reported that conflict. Startup and validation still passed. Optional
Florence-2 support reported absent; it is not used by this workflow. The frontend
also reported legacy API deprecation warnings without page errors. A concurrent
test instance reported database locking; the Registry instance was given its own
database. These observations do not establish full production compatibility.

Detailed logs, serialized prompts and screenshots are under
`/tmp/donut-smoke-20260919/` (temporary test artifacts).

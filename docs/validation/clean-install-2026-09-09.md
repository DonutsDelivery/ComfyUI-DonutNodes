# Clean installation test — 2026-09-09

Status: historical investigation below. The current workflow no longer requires
WAS. Default missing-node installation and full generation passed with the
current development DonutNodes code; see the
[WAS-free fresh-install report](no-was-fresh-install-2026-09-09.md).

## Environment and method

- Fresh official ComfyUI git checkout: `672ba9e5e388bd6bfac5ceef61f89ffdd9467200`.
- Standard `python -m venv`, Python 3.12.7; no system site-packages.
- Installed official `requirements.txt` and `manager_requirements.txt` with pip, without version overrides.
- ComfyUI frontend 1.51.10; Manager 4.2.2.
- Test root: `/tmp/donut-clean-install/ComfyUI`; interpreter: `/tmp/donut-clean-install/user-venv/bin/python`.
- Started with `--enable-manager --cpu --port 8189`. CPU mode is an environment limitation: the test host has no working NVIDIA driver.
- Loaded the existing `DonutWF_v3_Modular.json` via the browser's normal Open file picker.
- Used Manager's Missing Nodes → Install All UI with its default selected versions.
- No copying of dependencies or custom nodes from the development installation; no repair or constraint overrides before recording the baseline.
- An earlier uv-created environment is excluded from these results. Manager itself chooses uv for node dependencies by default; that is its normal behavior.

## Baseline observations

1. Official base dependencies install successfully; `pip check` reports no broken requirements.
2. The fresh ComfyUI server and browser start successfully.
3. Loading the workflow reports 7 missing pack groups and 5 missing models. Six groups resolve to installable packs; one unknown group contains frontend-only Donut nodes.
4. Manager offers these default versions: DonutNodes 2.0.3, Impact Pack 8.28.3, Impact Subpack 1.3.5, WAS Node Suite 1.0.1, Derfuu ModdedNodes 1.0.1, and ComfyUI-bleh nightly.
5. Manager cannot identify `DonutLatestPreview`, `DonutModelDownloads`, or `DonutWorkflowPanel` before Donut's frontend extension is installed.
6. The missing-node UI does not list the internally called Krea2 edit, NAG, or seed variance packs.

## Not covered yet

- Installation completion and restart, registration failures, post-install package conflicts.
- Full workflow validation and model acquisition.
- CUDA inference and image parity.
- ComfyUI Desktop installer on Windows/macOS. Linux manual install is not evidence for that route.

## Logs

- `/tmp/donut-clean-install/user-base-install.log`
- `/tmp/donut-clean-install/clean-startup.log`

Official instructions: https://github.com/Comfy-Org/ComfyUI#manual-install-windows-linux

## Exact target and post-install generation attempt

The initial baseline above used a different workflow. On resuming, the user clarified the target as `/home/user/DonutWF_v3_versions/DonutWF_v3_censored.json` (268249 bytes). That exact file was loaded via the normal browser Open action in the isolated server on port 8189.

- Restarted the existing isolated server after Manager finished installation, then refreshed the browser.
- `pip check` passed before restart, but this does not detect the observed binary incompatibility.
- WAS Node Suite's requirements specify `opencv-python-headless[ffmpeg]<=4.7.0.72`; Manager installed 4.7.0.72 alongside NumPy 2.5.3.
- On restart, OpenCV fails with `_ARRAY_API not found` / `numpy.core.multiarray failed to import`.
- DonutNodes, Impact Pack, Impact Subpack, and WAS Node Suite all report `IMPORT FAILED`. WAS also attempts an automatic uninstall that fails when its interactive pip prompt receives EOF.
- Manager's DonutNodes 2.0.3 package does not contain `DonutEditStudio.py`. The current development workflow therefore also needs a release-content compatibility check after resolving imports.
- Clicked Run on the exact censored workflow after restart. UI reports 11 errors: five missing pack groups, five missing models, and one missing-node-type error. Sampling never started.
- Missing models reported: `qwen-image/qwen_image_vae.safetensors`, `qwen3vl_4b_fp8_scaled.safetensors`, `finepornV4INT8NVFP4BF16_v4.safetensors`, `krea2_turbo_bf16.safetensors`, and `4x_NickelbackFS_72000_G.pth`.
- No dependency repairs or development-source substitutions have been applied to this baseline.
- Post-install startup evidence: `/tmp/donut-clean-install/post-install-startup.log`.

The default Manager installation path does not currently run this exact workflow out of the box in the tested environment. GPU inference remains untested; the test server is still configured for CPU.

## Packaged repair verification

The follow-up used `/tmp/donut-fresh-install/ComfyUI` and its isolated
`/tmp/donut-fresh-install/venv`, on port 8189. At the user's request, its DonutNodes
directory points to the local development source because the registry release
does not yet contain the workflow's new nodes. The working ComfyUI installation
and its venv were not used as the repair target.

- `install.py` registers Donut's existing OpenCV minimum in Manager's own repair
  list. WAS remains installed. The hook is part of the pack, rather than an
  undocumented local pip override.
- Disposable venv tests used Manager 4.2.2's real `PIPFixer`, with both WAS-first
  and Donut-first installation orders. Both ended with OpenCV 5.0.0 and NumPy
  2.5.3, and passed a real `cv2.cvtColor` operation. Those tests isolate the
  conflicting requirements; they are not complete workflow installations.
- The real test-server restart changed `opencv-python-headless` from 4.7.0.72 to
  5.0.0.93 through Manager. WAS loaded 217 nodes and both Impact packs imported.
- This exposed DonutFaceDetailer's import-order dependency on Impact. The node
  now registers before Impact is loaded and resolves Impact when executed.
  After another restart, `DonutFaceDetailer`, `DonutEditStudio`,
  `UltralyticsDetectorProvider`, `SAMLoader`, and WAS `Image Save` all appear in
  `/object_info`. No required backend type from the exact workflow is missing.
- Twenty installer and face-detailer unit tests pass, including Impact loading
  after DonutNodes and a missing-Impact execution error. `pip check` passes.
  The test venv's actual OpenCV image-conversion probe also passes.
- A fresh browser origin on `http://localhost:8189` loads the exact workflow
  without the stale failed-load state from the earlier test tab. Face detector
  and SAM controls are present. The UI reports only two missing-model errors,
  for `finepornV4INT8NVFP4BF16_v4.safetensors` and
  `krea2_turbo_bf16.safetensors`, with no missing custom nodes.

The [registration receipt](fresh-install-node-registration-2026-09-09.json)
records the exact workflow and installer hashes. Startup evidence is retained
in the test server's `user/comfyui_8189.log` and `.prev.log`.

The installer must be included in a new published DonutNodes release before
ordinary registry users receive it. Two OpenCV distributions remain installed
because the retained packs request different variants. Their imports currently
work, but this does not remove the existing shared-file packaging warning.
Model downloads and full GPU generation remain separate acceptance checks.

## Unconnected dependency discovery test

In the still-missing-pack test installation, adding an unconnected
`KreaSeedVarianceEnhancer` node with `properties.cnr_id` set to
`krea-seed-variance-enhancer` makes Manager's Missing Nodes list offer the pack
and an Install button. Removing that pack ID leaves the node unresolved in the
tested Manager cache. The normal saved registry metadata must be retained.

A separate three-node workflow, with no links, also exposes
`comfyui-krea2edit` (Krea2EditModelPatch), `krea2-nag`
(Krea2NormalizedAttentionGuidance), and seed variance to Missing Nodes / Install
All after a browser reload. Merely switching workflow tabs initially retained
incomplete Manager results; the clean reload showed all three packs. No Krea
pack was installed during this discovery experiment. Test files are
`/tmp/DonutWF_missing_dependency_probe.json`,
`/tmp/DonutWF_dependency_no_pack_id.json`, and
`/tmp/Krea_dependency_detection_only.json`.

The workflow's WAS metadata names `was-node-suite-comfyui`, version 1.0.2, while
the actual installed package reports `cnr_id`
`pr-was-node-suite-comfyui-47064894`, version 1.0.1. Manager therefore offers a
WAS missing entry although `/object_info` registers `Image Save`. Aligning the
ID/version in a temporary workflow copy removes that entry. This is a pack
identity mismatch, not another demonstrated WAS import failure. The production
workflow's metadata has not been changed by this experiment.

The first GPU generation attempt reached text encoding and failed in
DonutPromptConditioning because seed variance was absent. No image completed.
The test server currently runs CUDA on an RTX 4070; the CPU limitation recorded
for the earlier baseline no longer applies to this later runtime attempt.

## WAS v3 migration in the existing test environment

At the user's request, Manager uninstalled WAS 1.0.1 and explicitly installed
WAS 3.0.2 in the same `/tmp/donut-fresh-install` environment on port 8189.
Both Manager tasks completed successfully. No additional environment was made.
After restart, the v3 Image Save schema registers.

The live registry versions response explains the default old selection:
3.0.0–3.0.2 are `NodeVersionStatusFlagged`, 1.0.2 is
`NodeVersionStatusBanned`, and 1.0.1 is `NodeVersionStatusActive` and is returned
as `latest_version`. No reason for the flag was supplied. Explicit installation
of 3.0.2 succeeds through the normal registry install API and Manager, without
changing security settings. This supersedes the earlier speculation about
legacy-node compatibility causing the version selection. The 1.0.1 download
is a registry-provided `.tar.gz` archive, despite Manager treating it through
its common archive extraction path.

`/home/user/DonutWF_v3_versions/DonutWF_v3_censored.json` now records WAS 3.0.2.
Image Save uses `root=output`, native boolean values and `bit_depth=8-bit`.
A core StringConcatenate node joins the existing Final folder with the existing
seed filename, preserving the original output destination. The save panel's
folder control now edits that string input; root and bit-depth controls are
exposed. The existing model, prompt and sampling subgraphs are unchanged.

The browser loaded the migrated workflow and showed output / Final / webp /
overwrite off / 8-bit. The apparent delay during verification was a Codex
permission wait, not evidence of a ComfyUI stall.

A real small-image save-path execution passed:
`c011d729-2af0-4882-8763-4697f04f56b0`, output
`Final/was3_runtime_probe1.webp`. This verifies v3 execution with the migrated
settings and folder join; it is not a diffusion-generation result. Full image
generation still needs the previously identified internal Krea packs.

Current workflow SHA256:
`d317eddfac47c98f8e3d8c97482967812e76d6b8c5d5d2d115955d5821e7f029`.
The pre-migration workflow and old WAS pack are backed up under
`/tmp/donut-was-v3-backup` (about 32 MB); the old pack is no longer under
ComfyUI's custom_nodes directory. The existing test environment is on tmpfs.

## Dependency subgraph and full-generation rerun

The saved production workflow now contains a collapsed `Required node packs`
subgraph with three unconnected nodes and their normal `cnr_id` metadata.
The browser still lists all three packs as missing from inside this subgraph.
Using the workflow's Errors panel → Install All installed seed variance 1.2.0,
Krea2 NAG 1.0.2 and Krea2 Edit 1.2.3. Apply Changes restarted the same server.
All three imported successfully. No manual dependency installation or new
environment was used for this step.

The rerun ID is recorded in `was3-runtime-2026-09-09.json`; its final status
is recorded there when available. The saved sampling settings use editing off,
base generation, first upscale enabled, second upscale disabled and face
detailing with NAG. Passing this configuration does not certify the edit path
or the disabled second upscale.

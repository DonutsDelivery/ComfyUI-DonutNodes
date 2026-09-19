# DonutNodes 3.0.23

## User-facing changes since 3.0.22

- V5 workflow with separate setup, generation and finishing panels.
- Outpainting canvas presets within the selected megapixel budget, overlap and
  seam controls, and clearing a mask reliably disables selected-area editing.
- Smart masks: text-prompt selection with SAM3.1 alongside BiRefNet foreground
  selection, saved masks and manual refinement.
- Individual previews for base, each upscale, Face Detailer and final output.
- SeedVR2 finishing controls and model requirements, searchable LoRA selection,
  improved clipboard handling and reliable A/B reference-card selection.
- Complete supporting-model catalog and separate Windows/Linux/macOS installer
  bundle. Registry UI links to the auxiliary installer; executable downloader
  and standalone installers remain excluded from the Registry archive.
- Bounded Krea2 edit inference temporaries for LoRA, MLP and normalization.

## Validation and limits

Actual GPU prompt-mask and outpainting runs are documented in
`prompt-mask-gpu-2026-09-19.md`. Correct full-resolution-source outpainting passed
at 1408x704. The two-reference NAG first-upscale stress test at 2112x1056 still
exceeded 12 GB GPU memory. No claim of universal memory fit or seamless output.
User reviewed the result and authorized release with the current behavior.
Browser A/B selection checks passed, with no page errors.

Publication status: uploaded successfully; Registry approval remains unverified (Pending).

Release checks: 168 frontend tests, 24 subject-mask tests, 39 independent-crop
checks, 22 native downloader/staging tests and 4 standalone installer tests pass.
CLI packed archive staged through `tools/prepare_registry.py` successfully.


## Publication verification

Git release commit: `6d656ec`, pushed to `main`.
Exact version `3.0.23` checked at 2026-09-19 20:17:11 UTC:
`NodeVersionStatusPending`. Downloaded ZIP matches the staged upload byte for
byte per member (207 files); SHA-256
`7d2c093aff61422e5b27cc99f40b93f7310ed6e7ff88c91c13c662f139ce8346`.
Downloader backend and standalone installer files are absent from the Registry
ZIP. Auxiliary GitHub installer URL verified against the local artifact.
Hourly monitoring offered; no monitor created without the user's answer.

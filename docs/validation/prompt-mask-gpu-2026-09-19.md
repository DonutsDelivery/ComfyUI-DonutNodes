# Prompt masking: actual GPU execution, 2026-09-19

Implemented native SAM3.1 prompt selection in Reference B's Smart Mask controls.
Both independent-crop and legacy Edit Studio paths call the same implementation.
New widgets append after existing crop fields to preserve positional workflows.
Existing BiRefNet, manual and external-mask modes remain available.

Downloaded `sam3.1_multiplex_fp16.safetensors` using the live Donut downloader:
1,745,546,848 bytes, SHA-256
`9ba99c92703c2e8b4f47de2d34a539bb8e18923049e238b780d70dbe6368eb03`.
Pinned Comfy-Org revision `f38cd62b71494b53ac2b56ca36e24f3c8d565581`.
Installer catalogs include the checkpoint; runtime masking never downloads it.

## Executed tests

Used an isolated GPU server on port 8201 with the user's matching ComfyUI core,
GPU Python environment, current Donut code, and symlinked existing models. This
is not the clean CPU dependency environment from the earlier smoke report.

- `hat`: job `c5873bee-7f9e-4487-83e9-1c8dbd72621b`, success, 3.14 s.
- `shirt`: job `3a924100-69f9-401c-94b8-12d3f7b72116`, success, 1.95 s.
- Visually inspected both masks on the same 288 × 224 reference: the first
  selected the central hat, the second the central shirt. These were actual
  native SAM3 GPU inference calls through DonutSubjectMaskPreview.
- Real Chromium loaded V5, exposed the Mask prompt and Select from prompt
  controls, and serialized Prompt selection with `hat` and threshold 0.5.
  No browser page errors. Preview jobs above were submitted via API, not clicks.
- Combined Edit Studio Prompt selection + outpainting + NAG completed eight base
  sampling steps at 1408 × 704. First upscale then failed with CUDA OOM in a
  LoRA operation. Job `c141ba42-2592-4c5f-bbd1-de233e23ef2f`, 217.35 s, **failed**.
- Submitted base-only save job `d95bf68f-698a-4b31-bd46-fd81bec72257`; it successfully
  saved the cached, actually generated base image. This was not a second sampling
  pass. Output: `/tmp/donut-fresh-20260919/ComfyUI/output/DonutFeatureTests/prompt_hat_base_00001_.png`.
- Compared preserved pixels against the source resized by the outpainting
  placement: maximum difference 0 across the 889 × 704 protected region.
  New content appeared to the right. The join was visibly imperfect; no seamless
  outpainting quality claim. Original source was small and enlarged for this test.

The generation test used the installed Krea2 FP8 variant, dynamic grounding,
and the workflow's edit/NAG code. Face detail and SeedVR2 were excluded from this
targeted test. The upscale failure remains unresolved; this report is not a claim
that all enabled stages pass. No publication was performed.

## Regression checks

168 frontend tests passed. Subject-mask tests (24), independent-crop tests (39),
and native download/Registry-staging tests (22) passed. New tests cover empty
prompt rejection, cache invalidation on prompt/threshold change, and downloading
the prompt checkpoint while masking is disabled. Installer ZIPs rebuilt.

Source guide: `docs/prompt-subject-mask.md`. Temporary masks and execution records
are `/tmp/donut-sam-{hat,shirt}*` and `/tmp/donut-{base,upscale}-history.json`.

## Follow-up: real upscale stress test

Retried the unchanged graph after unloading models: failed again. Added bounded
native linear LoRA contribution calculation, applied the existing MLP chunking
to base edit sampling as well as upscale, and bounded per-token Q/K RMSNorm
FP32 temporaries on the attention token axis. These preserve full-frame attention;
they do not tile the image or reduce reference resolution.

- Native bypass/parity suite: 15 tests pass, including stacked LoRA numerical
  comparison across chunk boundaries.
- Memory suite: 3 tests pass, including hooked MLP and B,H,T,D RMSNorm parity.
- Combined job `c1fe38a8-17f4-48e9-9a27-563978e01d01`: base completed;
  upscale failed, 253.76 s.
- Combined job `7b25e4b5-84df-4829-b109-d0879546ed75`: base completed;
  upscale failed, 263.17 s. Saved base `prompt_hat_base_00003_.png` inspected:
  new content on the right, but a clear join remains. Protected 889x704 region
  matches the source resized to 905x704 with Lanczos exactly (max/mean error 0).
- Isolated upscale retry `57a4a2ff-e9ae-42ec-8421-ce92b63c1ade` loads that saved
  base and uses identical upscale inputs: failed in positional encoding, 26.85 s.
- Diagnostic `23a05cab-c66a-4381-9132-70c12da75058` with server `--reserve-vram 4`:
  still failed, 26.32 s. This setting was only applied to the isolated server.

The upscale is full-frame 2112x1056 (1.5x), two edit references, NAG enabled,
FP8 Krea2 and bypass LoRAs. Turbo converts the requested eight steps to three
sampling steps at effective denoise 0.375. This workload remains an unresolved
12 GB GPU OOM; the local memory changes are not a verified cure for it.
Neither seamless outpainting quality nor complete fresh-install execution is
certified by these tests. No release was published.

## Corrected full-resolution source

The user identified the earlier 288x224 reference as a quarter-scale preview.
It was unsuitable for judging outpainting quality. Repeated base-only outpainting
with `output/Final/4539680068827512.webp` (3456x2688) in both reference slots,
keeping the graph's other image-generation settings unchanged and the canvas at
1408x704. Job `df97e2cd-12a8-4d71-aaa2-f09f9da3e948` succeeded in 216.21 s.
Output: `/tmp/donut-fresh-20260919/ComfyUI/output/DonutFeatureTests/full_resolution_outpaint_00001_.png`.
Visually sharper source and a more continuous extension; human review pending.
This was base outpainting only, not an upscale retry.

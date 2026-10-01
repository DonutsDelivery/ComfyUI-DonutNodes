# Release validation: 3.1.1 — 2026-10-01

## Included changes

PR #77 adds bounded edit-mode MLP and normalization operations, including
64-row text-fusion layerwise chunks and 1024-token diffusion/refiner chunks.
Explicit MLP and normalization opt-outs remain independent.

Commit `e378c66` also makes the existing LoRA chunking option cover a lone plain
LoRA in Experimental bypass, including the identity-edit LoRA. Runtime wrapping
preserves strength, canonical adapter data, and the native disabled path.

No panel bindings, defaults, attention geometry, reference sizing, or distributed
workflow JSON changes are included. No replacement Civitai JSON is required.

## Validation and limits

- 73 focused checks passed, including two synthetic CUDA allocation checks on
  an RTX 4070, using PyTorch 2.5.1+cu121.
- Compilation and whitespace checks passed.
- Synthetic incremental peak reductions were 55.56 MiB for layerwise MLP and
  4.00 MiB for the full rank-128 LoRA bypass hook. These are operation-specific
  allocated-memory measurements, not total Krea2 memory savings.
- No full Krea2 generation, checkpoint/reference encoding, hires/detailer
  workflow comparison, output PNG, or latency benchmark was run. The reported
  two-reference 12 GB OOM remains unverified.
- See [edit memory validation](edit-mode-memory-2026-09-30.md) and
  [single-LoRA validation](edit-lora-bypass-memory-2026-10-01.md) for scope,
  control requirements, transitions, and the remaining GPU procedure.

The user explicitly requested GitHub main and Comfy Registry publication after
reviewing these limits. Unrelated local panel/workflow changes are preserved and
excluded from this release. The restricted Registry distribution uses the
manual model-files panel, as documented in `docs/publishing.md`.

## Publication verification

- PR #77 was merged into GitHub main at
  `36e788974c1f47a2fb1d784104d8a5981f9cc188`. Local main was fast-forwarded to
  that revision while preserving unrelated uncommitted edits.
- `comfy-cli 1.20.0` packed the clean release checkout, and
  `tools/prepare_registry.py` prepared a fresh restricted staging directory.
  Configuration/security checks passed. Publication used the documented non-Git
  staging-directory fallback.
- Comfy Registry **3.1.1** upload succeeded.
- At **2026-10-01T19:45:43.397783Z**, both the version listing with status
  reasons and exact-version endpoint reported **NodeVersionStatusPending**.
  `status_reason` was null. Registry approval and normal update discovery remain
  unverified; upload success is reported separately.
- The user declined hourly review checks for 3.1.1. No recurring monitor is
  scheduled. Registry approval remains unverified.

## Downloaded package verification

- URL: `https://cdn.comfy.org/donutsdelivery/donutnodes/3.1.1/node.zip`.
- **228 files**, **15,775,578 bytes**.
- SHA-256:
  `c18d2289a94246615118011e6f1983604e8788128ab0bccff80c0981ec658f38`.
- The downloaded ZIP is byte-identical to the inspected prepublication ZIP and
  the CLI upload ZIP. Every extracted file matches the staging tree; version is
  3.1.1. Changed backend code and the unchanged workflow JSON match the clean
  release source.
- Required runtime assets, model catalog, and manual model-files panel are
  present. Credentials, caches, tests/tools, validation reports, automatic
  downloader backend, and standalone installers are absent.

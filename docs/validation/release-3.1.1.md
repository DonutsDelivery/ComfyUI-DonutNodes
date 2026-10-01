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

Preparation is in progress. Exact GitHub commit, Registry review status and UTC
check time, and published-package verification will be recorded after upload.
Upload success alone does not establish Registry approval.

# Release validation: 3.0.36 — 2026-09-23

- Corrective workflow-only release after 3.0.35 changed user-selected V5 defaults. Restores the original loader choices (`1120`: `krea2_turbo_bf16.safetensors`; `1122`: `finepornV4INT8NVFP4BF16_v4.safetensors`) and `DonutModelMergeKrea2` mode `Merge two models`, preserving the intended text-fusion path.
- Updates the regression test to assert those workflow defaults remain intact through panel organization.
- Source release commit pushed to `origin/main`: `a107f7b`.
- Validation: all 208 Node.js tests passed.
- Published with comfy-cli 1.20.0. Upload succeeded. CLI emitted the existing E702 semicolon security warning at `donut_txtfusion_guard.py:207`; it did not prevent upload.
- Published CDN ZIP downloaded and checked: 15,754,196 bytes; SHA-256 `75ceaeff767384e8f7a22e11d9b2537d2d43f8bead31046c5ee02b6b1becd057`. Archive version 3.0.36 was verified to contain the original two loader selections and `Merge two models` mode.
- Registry API check at `2026-09-23T22:24:22Z`: `NodeVersionStatusPending`; no status reason reported. Approval is unverified. Recheck only if the user requests recurring checks.

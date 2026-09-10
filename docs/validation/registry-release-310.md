# 3.0.10 release validation

## Scope

Routes Bypass 2 and Bypass 3 projector diffs to the active model-2 projector
when Experimental-bypass merging selects `txtfusion` from model 2. The bundled
V4 Beta workflow also exposes a **Face detail** checkbox in **06 · Generate &
finish**. It bypasses or re-enables the existing `DonutFaceDetailer` node.

## Validation

The focused bypass-projector and Fusion-preset test suites passed (30 tests).
The published workflow declares the Face detail control at `[1014, 984]`, its
`DonutFaceDetailer` target is present and active, and the published projector
implementation uses `add_model_patch_components` for the merge source.

The exact published `3.0.10` ZIP was downloaded and tested at
2026-09-10T15:41:20Z. It has SHA-256
`1d0ad1db30c0fef1f3e782afe41bf60f67ce9368cd89897a62c5fa6fe7c4d282`.
`pyproject.toml`, `DonutKrea2FusionControl.py`, and the V4 workflow match the
validated source files byte-for-byte.

The exact-version Registry check at 2026-09-10T15:41:20Z returned
`NodeVersionStatusActive` with `Passed automated checks`. Upload and Registry
approval are verified.

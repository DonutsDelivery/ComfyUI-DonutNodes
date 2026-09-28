# NAG leftover-noise audit fixes

Audit input: `nag_audit_reproducers.zip`, dated 2026-09-28. The findings were
conditional behavior defects, not proof that they caused any specific output.

## Changes

- Fresh positive conditioning from edit and dynamic-grounding encodes now gets
  the same Fusion tap transform after seed variance is reapplied. Positive taps
  are independent of the negative-only `nag_match_taps` opt-out, and the
  existing metadata signature keeps the operation idempotent.
- Dynamic alpha no longer replaces an explicitly supplied NAG negative with a
  grounding-generated edit negative. Implicit negatives still follow the
  grounding schedule.
- Donut wraps registered upstream T2I and edit NAG forwards so wrappers after
  NAG stay in the diffusion executor chain. The Fusion node also adapts an
  already-registered NAG wrapper on its output clone, covering either node
  order.
- Active NAG now raises a clear error when ComfyUI has `attn1_patch` or
  `attn1_output_patch` callbacks. The current dual-text NAG forward cannot
  preserve those hooks safely, so it no longer silently bypasses them.
- Five-dimensional `[B,C,F,H,W]` latents now flatten in batch/frame order and
  restore with the inverse mapping in Donut's experimental forward and the
  upstream T2I/edit wrapper adapter.

## Validation

- `python -m unittest test_nag_fusion_taps test_donut_grounding_schedule test_donut_grounding_nag` — 70 passed.
- `test_donut_nag_txtfusion` — 22 passed with a minimal Comfy wrapper-executor stub.
- Python compilation passed for every changed Python source and test file.
- `git diff --check` passed.

The checkout at `/home/user/Programs/ComfyUI` contains `custom_nodes` and
`models`, but not ComfyUI's Python core or the separate `ComfyUI-Krea2-NAG`
pack. Therefore these changes were not exercised in a live ComfyUI Run-button
workflow, against the installed upstream NAG implementation, or with a
generated PNG/image-quality comparison. The executor-chain regression uses a
small test double. The attention-hook behavior is tested as an explicit
rejection path.

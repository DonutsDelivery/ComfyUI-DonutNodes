# Release validation: 3.0.44 — 2026-09-28

## Changes

Includes reviewed fix commit `b32f1aac2c60585b6e4276944cfc1a2a9491b8b6`:

- Restore positive Fusion taps after edit encoding and variance in hires and Face Detailer.
- Preserve reference-aware edit forwarding when NAG becomes inactive, while retaining downstream wrappers.
- Bind standalone T2I/edit callback IDs so each installs its correct wrapper adapter.

## Verification and limits

The [follow-up validation](nag-edit-followup-2026-09-28.md) records 123 passing focused tests and 36 upstream routing checks, plus 25 tests rerun with the actual ComfyUI executor. The user supplied an independent review reporting 55 additional passing CPU checks with no blocker in these changes; that review was not rerun as part of publication.

No full ComfyUI/GPU generation or residual-grain improvement has been established. Alpha remains stage-local, Turbo denoise zero still selects a nonzero step, and active-NAG attention-hook conflicts raise an error. No distributed workflow JSON changed.

Registry packaging uses the documented restricted distribution: automatic model downloading is replaced with the manual model-files panel. Publication and exact-version review results will be recorded below.

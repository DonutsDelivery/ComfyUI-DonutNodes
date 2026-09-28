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

## Publication and replacement

GitHub release commit `6dcf321` was pushed. Registry upload succeeded; exact-version and listing checks at `2026-09-28T19:06:34Z` returned `NodeVersionStatusPending`, with an empty listing `status_reason`. Approval remains unverified; hourly monitoring was offered and has not been enabled.

The downloaded ZIP matched the staged files, but inspection found a pre-existing packaging omission: root-level `test_donut_grounding_controls.mjs` was included because only Python tests were excluded at the root. This violates the documented runtime-only packaging requirement. Version 3.0.45 replaces this package with the JavaScript test excluded and a staging guard against recurrence. No cause of any Registry scan result is inferred from this finding.

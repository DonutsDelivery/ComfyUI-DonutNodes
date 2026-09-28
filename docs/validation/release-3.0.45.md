# Release validation: 3.0.45 — 2026-09-28

Contains the reviewed NAG fixes and CPU validation described in [3.0.44](release-3.0.44.md) and [the follow-up report](nag-edit-followup-2026-09-28.md).

This replacement excludes root JavaScript development tests from the Registry ZIP and adds a corresponding staging rejection. Version 3.0.44 uploaded successfully but its ZIP included `test_donut_grounding_controls.mjs`. Runtime Python and frontend behavior are unchanged between these two versions. No workflow JSON changed.

GPU generation and residual-grain improvement remain unverified. Publication and exact-version approval status are recorded below.

# NAG auto-phi audit — 2026-09-27

## Backend calculation

`resolve_nag_phi` has not changed since commit `624e38c` on September 21.
With auto enabled and positive alpha/scale, it returns `nag_phi_scale / nag_alpha`.
The calculated value is forwarded to both the upstream NAG node and Donut's
optional text-fusion wrapper. Alpha or scale at zero disables this guidance.
With auto disabled, the explicit manual phi is used.

At scale 1, alpha 0.25 gives phi 4, alpha 0.15 gives approximately 6.667, and
alpha 0.1 gives phi 10. This preserves alpha × phi before tau normalization.
The installed NAG implementation normalizes/clips the guided attention before
blending with alpha, so auto mode does not imply identical outputs at different
alpha values. Recent DOM and VRAM patches did not change this calculation.

The latest completed execution inspected during this audit had auto enabled,
alpha 0.25, scale 1 and tau 2.5 for generation, first upscale and FaceDetailer.
The second upscale had auto disabled, but that stage itself was disabled.
Its stale auto setting therefore did not affect that execution. The execution
metadata supports the queued values; no new generation was submitted.

## Panel synchronization defect

Commit `d4192b7` on September 21 consolidated shared NAG controls onto the base
sampling panel and removed their per-stage duplicates. Its commit handler tried
to mirror changes by searching only that same panel for duplicate controls with
the same widget name. After consolidation, there were no such duplicates.
Changes to auto phi, guidance scale, tau and the other shared fields could
therefore update the base sampler while leaving hidden stage values stale.

The local fix finds panels with the same explicit seed-path family identity and
uses their remaining `nag_enabled` bindings to locate the stage nodes. A shared
field edit updates that field on each bound stage. Deduplication uses node
objects, avoiding collisions between repeated node IDs in different subgraphs.
The shared-field list is imported from the panel model instead of duplicated.
Per-stage enable toggles and the separately wired alpha control are preserved.

Saved values are not overwritten on page load. After a hard refresh, the user
can reapply the shared auto-phi and scale controls to synchronize existing
stages. No workflow JSON replacement or backend restart is required.

## Scope

Source/history and existing execution metadata inspection only. No tests,
browser interaction, workflow submission or generation were run. The panel fix
is included in the [3.0.40 release](release-3.0.40.md). Runtime interaction
remains unverified.

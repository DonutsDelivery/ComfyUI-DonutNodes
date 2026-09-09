# 3.0.8 release validation

## Scope

Renames the current composition choice to `Fusion + UncensorFix weights`, hides
obsolete legacy choices from the Workflow Panel, and adds an independent
UncensorFix weight-strength control. Tap strength now affects fusion without
also changing the embedded weight strength.

## Validation

22 backend parity/composition tests and the Fusion/Workflow Panel UI suites
passed before packaging.

Published successfully on 2026-09-09 using Comfy CLI 1.15.0 from a fresh
Registry-specific staging directory. The downloaded archive's 159 entries
matched the validated upload byte-for-byte.

Exact-version check at 2026-09-09T11:56:04Z: NodeVersionStatusPending with an
empty status reason. Upload is verified; Registry approval remains unverified.

Follow-up 2026-09-09T12:08:29Z: NodeVersionStatusPending, empty status_reason.

Follow-up 2026-09-09T12:14:29Z: NodeVersionStatusPending, empty status_reason.

Follow-up 2026-09-09T12:25:59Z: NodeVersionStatusPending, empty status_reason.

Follow-up 2026-09-09T12:19:59Z: NodeVersionStatusPending, empty status_reason.

Follow-up 2026-09-09T12:32:29Z: NodeVersionStatusPending, empty status_reason.

Follow-up 2026-09-09T12:38:29Z: NodeVersionStatusPending, empty status_reason.

Follow-up 2026-09-09T12:57:59Z: NodeVersionStatusPending, empty status_reason.

Exact-version follow-up 2026-09-09T13:03:59Z: NodeVersionStatusActive with
status reason "Passed automated checks". Version 3.0.8 is approved and
available through normal Registry discovery.

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

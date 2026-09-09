# 3.0.6 release validation

## Scope

Fixes the v4 Donut Workflow Panel so that selecting a compatibility preset and
changing tap strength updates the nested Krea2 Fusion Control and its visible
advanced controls. The release also moves the AuraFlow enable control from the
Prompts panel to the Generate & finish panel.

## Package and publication checks

Published successfully on 2026-09-09. At 11:14:39 UTC the exact version was
NodeVersionStatusPending with an empty status_reason. Approval is unverified.
Downloaded the published ZIP and verified all 159 entries match the upload
archive byte-for-byte. Five-minute review follow-up scheduled in this task.
Targeted panel/Fusion/layout tests and 28 dependency-isolation tests passed.
Packaging used the installed Comfy CLI 1.15.0 environment; CLI 1.3.8 ignored
the existing exclusions and its archive was rejected before publication.

Follow-up 2026-09-09T11:20:59Z could not reach api.comfy.org because DNS
resolution failed; registry approval remains unverified.

Follow-up 2026-09-09T11:48:29Z: NodeVersionStatusPending, empty status_reason.
Sandbox DNS failed; the read-only check succeeded outside the sandbox.

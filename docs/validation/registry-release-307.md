# 3.0.7 release validation

Fusion only / Fusion + LoRA now controls backend composition independently
of preset labels. Preset selection populates composition as well as fusion
settings; legacy saved control values retain their previous behavior.

21 backend parity/composition tests and the Fusion/panel UI suites passed.
Published using Comfy CLI 1.15.0 from registry-specific staging, retaining
the manual downloader UI. Downloaded ZIP: all 159 entries match the upload.

2026-09-09T11:37:26Z: exact 3.0.7 status NodeVersionStatusPending, empty reason.
Upload verified; approval unverified. User requested publishing and Git push
without waiting for registry review.

Follow-up 2026-09-09T11:48:29Z: NodeVersionStatusPending, empty status_reason.
Sandbox DNS failed; the read-only check succeeded outside the sandbox.

Follow-up 2026-09-09T12:08:29Z: NodeVersionStatusPending, empty status_reason.

Follow-up 2026-09-09T12:25:59Z: NodeVersionStatusPending, empty status_reason.

Follow-up 2026-09-09T12:14:29Z: NodeVersionStatusPending, empty status_reason.

Follow-up 2026-09-09T12:19:59Z: NodeVersionStatusPending, empty status_reason.

Follow-up 2026-09-09T12:32:29Z: NodeVersionStatusPending, empty status_reason.

Follow-up 2026-09-09T12:38:29Z: NodeVersionStatusPending, empty status_reason.

Follow-up 2026-09-09T12:57:59Z: NodeVersionStatusPending, empty status_reason.

Exact-version follow-up 2026-09-09T13:03:59Z: NodeVersionStatusActive with
status reason "Passed automated checks". Version 3.0.7 is approved and
available through normal Registry discovery.

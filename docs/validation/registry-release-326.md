# DonutNodes 3.0.26

Fixes compatibility Off handling, panel/serialized composition mismatch,
experimental preset composition transitions, preservation of experimental
adjustments on reload, and preservation of saved Fixed seed policies.
Adds panel verification requirements to AGENTS.md.

Includes the V5 workflow composition correction. Upload the updated
`workflows/v5/DonutWF_v5.json` manually to Civitai.

Validation: see panel-regressions-2026-09-20.md for UI-queued PNG evidence and
test results. The backend Off guard is unit-tested; runtime Python verification
requires a ComfyUI restart. Unconfirmed Edit Studio switch changes are excluded.

Release commit `bb1c535` pushed to main. Registry upload succeeded.
Exact version checked at 2026-09-20 01:05:42 UTC: `NodeVersionStatusPending`;
status reason empty. Approval is unverified. Hourly checks offered; no monitor
scheduled without the user's yes.

Published ZIP verified against staged upload: every member matches. Standalone
installers, downloader backend, credentials and development-only files excluded.
SHA-256: `d2b43de8c3b1078cb6a1d0aee2440920ef33cc5a733ce1c1744a40dbfdcba0f6`.

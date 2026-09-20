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

## Earlier V5 review findings

Rechecked 2026-09-20 01:07:31 UTC: 3.0.23 and 3.0.24 are Flagged;
3.0.22 is Active. 3.0.26 remains Pending. The listing has no 3.0.25 entry.
Both flagged versions report the same YARA `$socket4` / `any-network-requests`
findings on `.bind(` in `donut_reference_mask.py`, `donut_grounding_schedule.py`,
and `donut_crop_studio.py`. These calls bind Python function arguments through
`inspect.signature`, not network sockets. All three reported lines also exist
in the verified 3.0.26 published ZIP, so approval must not be assumed.
Resolve through Registry review with the exact findings; do not obscure or
rename equivalent operations merely to evade the scanner.

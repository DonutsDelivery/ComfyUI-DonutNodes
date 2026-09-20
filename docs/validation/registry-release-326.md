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

Registry publication and approval: not yet verified.

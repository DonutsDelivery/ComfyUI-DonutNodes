# Release validation: 3.0.37 — 2026-09-25

- Adds opt-in one-pass VAE damage subtraction to the V5 base decode, Donut hires
  upscales, and face detail. Defaults remain Off; each stage uses its connected
  VAE and has an independent 0–4 strength control.
- No distributed workflow JSON changed. The V5 frontend upgrades only the
  directly connected stock base decoder in memory when the backend registers
  `DonutVAEDecode`; the user's JSON file does not need a manual upload.
- `git diff --check` and Python bytecode compilation passed for the four
  changed backend modules. The existing E702 warning in
  `donut_txtfusion_guard.py` was cleared by splitting adjacent digest updates
  onto separate lines; the computed digest is unchanged.
- UI end-to-end validation is blocked in this environment: CUA reports no
  available browser, so the changed controls could not be set in the panels or
  queued with ComfyUI's Run button. No audit PNG, metadata/prompt comparison,
  save/reload transition check, or GPU output comparison was produced. The
  running backend's `/object_info` confirms registration of the new decoder and
  updated upscale node, but does not verify panel bindings or generation.
- Registry package validation, publication, published ZIP inspection, and the
  exact-version review status are recorded below after they are completed.

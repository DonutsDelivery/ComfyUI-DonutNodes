# Release validation: 3.0.35 — 2026-09-23

- Source release commit pushed to `origin/main`: `4858fe3665786902b9a8fe83870d28718729c19f`.
- Includes Edit Studio checkbox re-render race fix and slot-A clear behavior from `8172a8f`.
- Restores the fresh-install V5 workflow's public `krea2_turbo_bf16.safetensors` loader selections and `Single model` mode. Existing personal workflow choices are not modified by this release.
- Validation: all 208 Node.js tests passed. Python unittest with the ComfyUI venv ran 323 tests; 9 skipped; the Playwright browser suite could not import because Playwright is unavailable in that environment.
- Prepared the registry-specific staging directory with `python tools/prepare_registry.py`; verified staged `pyproject.toml` is 3.0.35, V5 defaults are public Krea2/Single model, workflow is present, and registry credentials are absent.
- Published with comfy-cli 1.20.0. Upload succeeded. CLI emitted the existing E702 semicolon security warning at `donut_txtfusion_guard.py:207`; it did not prevent upload.
- Published CDN ZIP downloaded and checked: 15,754,372 bytes; SHA-256 `742f6317b6e1c3b8ec7dab3e1c61acbd5436c2c9f5feee000f6a386bdb0c9a34`. The archive reports version 3.0.35 and contains the public Krea2 defaults and `Single model` mode.
- Registry API check at `2026-09-23T22:10:51Z`: `NodeVersionStatusPending`; no status reason reported. Approval is unverified. Recheck only if the user requests recurring checks.

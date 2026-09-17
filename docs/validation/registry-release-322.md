# DonutNodes 3.0.22 release validation

Ships PR #60: fix V4 reference guidance leaking into face refinement.
`donut_prompt.py` previously embedded the reference image into both the
main generation prompt and the face-detailer prompt when Reference
guidance was enabled, so the face pass inherited whole-image reference
conditioning instead of its face text only. Generation and upscaling
keep their reference guidance; normal face refinement now gets
text-only face conditioning; explicit identity editing (A/B face
references), seed variance, negative prompts, and prompt variants are
preserved. No workflow rewiring is needed for existing V4 JSON.

Validation: all 11 tests in `test_native_reference.py` pass (9 targeted
conditioning tests for reference-change/toggle/A-B/identical-text/seed
variance/prompt-variant coverage plus 2 others), run in the real
ComfyUI venv at
`/home/user/Programs/ComfyUI-new/ComfyUI/venv`. `py_compile` and
`git diff --check` clean. No GPU generation was run.

Git commit: `64f5779` (`main`, on top of PR #60 merge `6ae735f`).
Published from that exact pushed tree via local CLI (comfy-cli 1.20.0;
no publish workflow in the repository). Upload receipt:
`✓ All validation checks passed successfully` + `Upload successful.`
The linter E702 note about `tools/streamline_workflow.py:329` is
pre-existing and did not block upload.

Published ZIP SHA-256:
`368f00c8fd1f70d487370e93b522eba12a0eeb93d1756fbed747d40244d25f74`
(11,995,579 bytes; contains the updated `donut_prompt.py`,
`assets/uncensorfix.f32`, `model_sources.json`; no tests, `.env`, or
nested archives). ZIP:
https://cdn.comfy.org/donutsdelivery/donutnodes/3.0.22/node.zip

Exact version checked through
`/nodes/donutnodes/versions?include_status_reason=true`:
`3.0.22` → `NodeVersionStatusPending`. Older versions remain Active and
are served until approval; approval is unverified, no monitor created.

# DonutNodes 3.0.17 release validation

Selected-area inpainting with seam slider/preview, inversion, brush-size cursor,
rectangle tools, and single-frame VAE support. Includes the updated V4 JSON.

Tests: 45 mask/Studio/geometry and 12 installed-ComfyUI sampler tests passed.
Frontend: 128 passed, 2 fixture skips. Workflow streamlining: 44 passed, 1 skip.
Browser harness verified mask tools and persistence. GPU quality remains unverified.

Comfy CLI 1.15.0 packed and validated the staged Registry distribution with the
manual model-files UI. Source archive SHA-256:
`d45566ea1f03d473fd258be8b2459ab0aa499d61ba6e81fc4897aadc244207fe`.

Upload succeeded. Published ZIP integrity passed and all 162 payload files
matched staging byte-for-byte. Validation-created .ruff_cache files were excluded
from comparison and are absent from the published archive.
Published ZIP SHA-256: `6c80d53f739014998e12a218f351d09c3c89ba588df765cd7d6896fb8c342f35`.
ZIP: https://cdn.comfy.org/donutsdelivery/donutnodes/3.0.17/node.zip

Exact version checked at `2026-09-13T22:25:12.329137+00:00`: `NodeVersionStatusPending`; status reason empty.
Approval is unverified. No recurring monitor created; hourly follow-ups require
explicit user opt-in.

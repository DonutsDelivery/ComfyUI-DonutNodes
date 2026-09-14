# DonutNodes 3.0.18 release validation

Fixes legacy Edit Studio workflow metadata that left `mask_feather` blank at
queue time. Load, save, queue, and widget serialization now normalize the
inpaint controls; the DOM-only panel no longer adds a trailing positional
value. Existing masks, prompts, references, and seam-width settings are kept.

Validation: 133 frontend tests passed with 2 fixture skips; 57 installed-
ComfyUI Python tests passed. JavaScript syntax, Python compilation, and
whitespace checks passed. The supplied `ComfyUI_00001_.png` reproduced the old
22-entry metadata and repaired to 21 entries with `inpaint_enabled=false` and
`mask_feather=8`.

Git commit: `037d3af` (`main`). Clean source package SHA-256:
`08939aea2ed003e8abd854b63afc7586682b5a3e912dc81ba5d871cf3af4d58d`.

Upload succeeded. The exact published ZIP was downloaded and reports version
3.0.18. Published ZIP SHA-256:
`ce47a0033f2163d6a47d3846ed71f4c9df142dd175b42c1cb59cce8a0dc67189`.
ZIP: https://cdn.comfy.org/donutsdelivery/donutnodes/3.0.18/node.zip

The available Comfy CLI repacked the non-Git staging directory after
validation and included three temporary `.ruff_cache` files; the clean Git
archive and runtime payload exclude them. No credentials, tests, tools, or
automatic downloader backend were included.

Exact version checked at `2026-09-14T04:28:41Z` through
`/nodes/donutnodes/versions?include_status_reason=true`:
`NodeVersionStatusPending`, with an empty status reason. Approval is unverified;
no recurring monitor was created. Hourly follow-ups require explicit user
opt-in.

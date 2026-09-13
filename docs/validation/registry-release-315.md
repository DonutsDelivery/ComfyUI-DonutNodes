# 3.0.15 release validation

## Scope

The V4 Beta prompt panel now autosizes its text editors and uses one shared
wildcard control across the Prompt fields. Prompt 1 remains the connected
Prompt card; additional variants use the same face/style, subject/scene and
negative fields, with blank, duplicate, reorder and remove actions. The active
prompt is highlighted and can stay fixed or increment after each generation.
The conditioning node selects the active set before encoding, wraps the index
across Prompt 1 and its variants, and preserves the existing connected-prompt
behavior when no variants are configured.

## Validation

The focused Python suites passed (57 tests). The frontend suites passed (62
tests), including active-prompt highlighting and duplicate-variant behavior.
JavaScript syntax checks, workflow JSON validation and `git diff --check` also
passed. The repository-wide Python run was not collectible because the local
ComfyUI checkout is missing the optional `ComfyUI-Krea2T-Enhancer` package
imported by `test_krea2_fusion_control.py`.

The source archive was packed with Comfy CLI 1.15.0, passed `unzip -t`, and has
SHA-256
`35955b14655e1bd881f46644c7c0d96d06f1131ccb04d0c3c7e40ced4f774d6f`.
It contains 160 files, includes the required runtime assets and manual model
sources, and excludes tests, development tools, validation documents, the
automatic downloader backend and publishing guide. The fresh registry staging
directory `/tmp/donut-registry-release-315` passed `comfy node validate`.

The exact published ZIP was downloaded from
`https://cdn.comfy.org/donutsdelivery/donutnodes/3.0.15/node.zip`, passed
`unzip -t`, and contained the same 160 payload files as the validated staging
directory byte-for-byte. Its downloaded ZIP SHA-256 is
`5977d499742b08813a526e74c6681712274e16e90fbf9234e03e582a6646b10a`.

At `2026-09-13T00:19:11Z`, the exact version endpoint
`https://api.comfy.org/nodes/donutnodes/versions?include_status_reason=true`
returned `3.0.15` as `NodeVersionStatusPending` with an empty status reason.
Registry upload and ZIP verification are complete; Registry approval remains
pending and will be checked until the version becomes Active or Flagged.

The direct exact-version readback at `2026-09-13T00:20:26Z` returned the same
`NodeVersionStatusPending` status, a null status reason, and the published ZIP
URL above.

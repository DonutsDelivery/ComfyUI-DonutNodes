# 3.0.14 release validation

## Scope

Edit Studio clipboard paste now captures image paste before ComfyUI document
handlers, accepts both browser clipboard image representations, and keeps the
selected A/B slot active when scripted clipboard access is unavailable. The
backend continues to normalize PNG and JPEG uploads to the same stored PNG
reference format.

The release also preserves LoRA, UncensorFix, and Fusion model state through
Krea2 edit branches, including legacy workflows whose model roots differ.

## Validation

The affected Python suites passed (109 tests), the frontend clipboard,
geometry, layout, and workflow-reload suites passed, and `git diff --check`
passed before packaging.

The release archive was packed with Comfy CLI 1.15.0, passed `unzip -t`, and
has SHA-256
`857cd58efae084163515a2f5847795efe2a3ed6b76833d5107269a3fb45c1ce6`.
It contains 160 files, includes the required runtime assets, excludes tests,
development tools, validation documents, and the automatic downloader backend,
and passes `comfy node validate` from the fresh staging directory
`/tmp/donut-registry-release-314`.

The exact version endpoint was checked before publication at
`https://api.comfy.org/nodes/donutnodes/versions?include_status_reason=true`;
version `3.0.14` was not yet listed. Registry upload and the post-upload ZIP
and status checks remain pending because this environment has no Registry API
key configured.

Upload success and Registry approval will be recorded separately. The exact
version endpoint will be checked with `include_status_reason=true` after upload.

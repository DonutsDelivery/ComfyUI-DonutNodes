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
passed before packaging. The exact packed archive, fresh registry staging
directory, and published ZIP will be recorded below after publication.

Upload success and Registry approval will be recorded separately. The exact
version endpoint will be checked with `include_status_reason=true` after upload.

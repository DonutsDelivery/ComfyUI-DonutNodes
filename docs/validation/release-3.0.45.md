# Release validation: 3.0.45 — 2026-09-28

Contains the reviewed NAG fixes and CPU validation described in [3.0.44](release-3.0.44.md) and [the follow-up report](nag-edit-followup-2026-09-28.md).

This replacement excludes root JavaScript development tests from the Registry ZIP and adds a corresponding staging rejection. Version 3.0.44 uploaded successfully but its ZIP included `test_donut_grounding_controls.mjs`. Runtime Python and frontend behavior are unchanged between these two versions. No workflow JSON changed.

GPU generation and residual-grain improvement remain unverified. Publication and exact-version approval status are recorded below.

## Publication verification

- GitHub `main` release commit `a779a8f` was pushed before publication.
- `comfy-cli 1.20.0` packed the source; `tools/prepare_registry.py` produced a new staging directory. Configuration and security validation passed. The new staging guard correctly rejects the downloaded 3.0.44 archive containing the development test.
- Registry upload succeeded. Publishing used the documented staging-directory fallback (no Git repository in staging).
- At `2026-09-28T19:07:36Z`, both the version listing with status reasons and the exact-version endpoint reported **NodeVersionStatusPending** for **3.0.45**. The listing's `status_reason` was empty. Approval and normal update discovery remain unverified.
- The user declined hourly status checks for 3.0.45 (and the superseded 3.0.44 release). No recurring monitor is scheduled. Registry approval remains unverified.

## Downloaded package

- URL: `https://cdn.comfy.org/donutsdelivery/donutnodes/3.0.45/node.zip`
- **228 files**, **15,773,975 bytes**.
- SHA-256: `70ce289fe09c8e987f54be308a25fd7f42d498874e50e6a405e114a97390fe25`.
- Every downloaded file matches the validated staging tree byte for byte. The ZIP declares 3.0.45 and contains all three reviewed fixes, including positive-tap restoration in both refinement stages.
- Required assets and the generated manual model catalog are present. Development tests, tools, caches, credentials, the automatic downloader backend, and standalone installers are absent.

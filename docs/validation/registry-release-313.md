# 3.0.13 release validation

## Scope

Edit mode now preserves the Edit Studio LoRA/UncensorFix branch through Krea2
Fusion, sampler, tiled-upscale, and face-detail stages. Legacy V3 workflows
also repair untagged Fusion model routing when they reload in the frontend.

## Validation

Python compilation and the focused edit-model, Edit Studio, tiled-upscale, and
face-detailer suites passed (47 tests). The frontend workflow reload and
streamlining suites passed (62 tests, with two intentional skips), and
`git diff --check` passed.

The release archive was packed with Comfy CLI 1.15.0 from the working source,
passed `unzip -t`, and has SHA-256
`84502c071a16821799b9fc845e6ccc22f8cbeaf88c8497039ca00766866ef8a4`.
The fresh registry staging directory contains 159 files, includes the manual
model-download panel and required runtime assets, excludes tests, development
tools, validation documents, and the automatic downloader backend, and passes
`comfy node validate`.

The exact published version will be checked at the Registry versions endpoint
with `include_status_reason=true` after upload. Upload success and Registry
approval will be recorded separately below.

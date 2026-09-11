# 3.0.12 release validation

## Scope

The V4 App Mode LoRA panel now persists a newly added row even when the native
LoRA editor has not initialized yet. It also uses a fallback row ID for browser
contexts without `crypto.randomUUID`. The V4 Latest result panel ignores empty
execution events instead of raising an `output` property error. Workflow reload
repair preserves absent empty metadata on helper subgraphs.

## Validation

The App Mode, native LoRA, result preview, workflow reload and dependency UI
suites passed (112 frontend tests, with two intentional skips), and the Python
dependency-isolation suite passed (28 tests).

The release was packed with Comfy CLI 1.15.0 from the working source and
prepared in a fresh registry-specific staging directory. The staged archive
contains 159 files, including the registry manual-download panel, and excludes
tests, validation reports, development tools and the automatic downloader
backend. The packed archive passed `unzip -t` and has SHA-256
`8a4e495cba4784da7078e6eff5c88f20d1c7818cbe2205a1ca6b91f43d479a0f`.

The exact published version was checked at 2026-09-10T23:45:53Z through the
Registry versions endpoint with `include_status_reason=true`. It returned
`NodeVersionStatusActive` with status reason `Passed automated checks`. The
published ZIP was downloaded again, passed `unzip -t`, matched the validated
staging archive byte-for-byte, and retained the SHA-256 above. Upload and
Registry approval are verified.

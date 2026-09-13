# 3.0.16 release validation

## Scope

Workflow repair validation remains strict and atomic, but a rejected repair no
longer aborts ComfyUI's browser-side workflow load or serialization. DonutNodes
now leaves the original workflow data untouched, continues the operation, and
logs the exact validation error to the browser console. The frontend import
cache key was advanced so updated installations load the new repair module.

The exact V4 Beta workflow currently distributed through CivitAI, Git and the
3.0.15 Registry package was checked during investigation. Those copies were
byte-identical and the workflow passed the reload validator on ComfyUI 0.35.1.
The `donut_streamlining.source_sha256` field is provenance for the source before
streamlining; it is not the checksum of the distributed workflow file.

## Validation

The frontend suite passed 122 tests with two fixture-dependent skips, including
new coverage for failed import and export repairs. The Python streamlining suite
passed 44 tests with one fixture-dependent skip. `git diff --check` passed.

The source archive was packed with Comfy CLI 1.15.0 and passed `unzip -t`. Its
SHA-256 is
`fad8423b49c3d37511f580c485420afb76355e885806073c9298948c38cc41af`.
It contains 160 payload files, includes the required runtime assets, and excludes
tests, development tools, validation documents, credentials, the automatic
downloader backend and publishing guide. The fresh Registry staging directory
`/tmp/donut-registry-release-316` substitutes the manual model-download UI and
passed `comfy node validate`.

Registry upload, published ZIP verification and exact-version review status are
not yet recorded.

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

The exact published ZIP was downloaded from
`https://cdn.comfy.org/donutsdelivery/donutnodes/3.0.16/node.zip`, passed
`unzip -t`, and contained the same 160 payload files as the validated staging
directory byte-for-byte. Its downloaded ZIP SHA-256 is
`b42d0e08d98c12eb5611945708e301ace73ccc075f750b72149eec92187596c7`.

At `2026-09-13T20:17:01Z`, the exact versions listing returned `3.0.16` as
`NodeVersionStatusPending` with an empty status reason and the published ZIP URL
above. Registry upload and ZIP verification are complete; Registry approval is
pending. A five-minute follow-up is active until the version becomes Active or
Flagged.

Follow-up exact-version check at `2026-09-13T20:25:00Z` returned the same
`NodeVersionStatusPending` status with an empty status reason. Approval remains
pending.

Follow-up exact-version check at `2026-09-13T20:30:54Z` returned the same
`NodeVersionStatusPending` status with an empty status reason. Approval remains
pending.

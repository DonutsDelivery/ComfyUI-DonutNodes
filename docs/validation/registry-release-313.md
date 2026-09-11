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

The source archive has SHA-256
`84502c071a16821799b9fc845e6ccc22f8cbeaf88c8497039ca00766866ef8a4`.
Publishing from the fresh staging directory regenerated the same 159-file
payload with SHA-256
`c6cc3503fbae6fdd750e05450559ff652d4fb595ba394b3a41c0188b35c62d42`.

Follow-up check at 2026-09-11T21:26:31Z returned the same
`NodeVersionStatusPending` status and empty reason. The published ZIP passed
`unzip -t` and retained SHA-256
`c6cc3503fbae6fdd750e05450559ff652d4fb595ba394b3a41c0188b35c62d42`.

At 2026-09-11T21:12:58Z, the exact version endpoint returned
`NodeVersionStatusPending` with an empty status reason. The published ZIP at
the returned download URL was downloaded, passed `unzip -t`, and matched the
staging archive byte-for-byte (`c6cc3503fbae6fdd750e05450559ff652d4fb595ba394b3a41c0188b35c62d42`).
Upload is verified; Registry approval is pending. Because this release changes
runtime package contents, follow-up checks continue every five minutes until
the exact version becomes Active or Flagged.

Follow-up check at 2026-09-11T21:20:31Z returned the same
`NodeVersionStatusPending` status and empty reason. The published ZIP again
passed `unzip -t` and retained SHA-256
`c6cc3503fbae6fdd750e05450559ff652d4fb595ba394b3a41c0188b35c62d42`.

Follow-up check at 2026-09-11T21:32:01Z returned the same
`NodeVersionStatusPending` status and empty reason. The published ZIP passed
`unzip -t` and retained SHA-256
`c6cc3503fbae6fdd750e05450559ff652d4fb595ba394b3a41c0188b35c62d42`.

Follow-up check at 2026-09-11T21:37:31Z returned the same
`NodeVersionStatusPending` status and empty reason. The published ZIP passed
`unzip -t` and retained SHA-256
`c6cc3503fbae6fdd750e05450559ff652d4fb595ba394b3a41c0188b35c62d42`.

Follow-up check at 2026-09-11T21:43:01Z returned the same
`NodeVersionStatusPending` status and empty reason. The published ZIP passed
`unzip -t` and retained SHA-256
`c6cc3503fbae6fdd750e05450559ff652d4fb595ba394b3a41c0188b35c62d42`.

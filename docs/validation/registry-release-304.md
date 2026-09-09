# 3.0.4 release validation

Checked 2026-09-09T08:17:43.964809+00:00: NodeVersionStatusPending.

Published archive downloaded and all file bytes compared with the validated registry staging directory. Required weight asset matches original SHA-256. No optional downloader backend, tests, authoring tools or distribution templates shipped. Manual registry UI verified; full downloader restored in local GitHub source, not pushed to GitHub by this release operation.

110 Python tests passed (97 weight/LoRA/merge/dependency tests plus 13 downloader tests); 67 JavaScript tests passed and one optional external-fixture test skipped. Both optional-backend present/absent registration paths tested. Weight size, corruption, checksum, missing asset, tensor parity, and bypass behavior tested. Full generation/browser testing was not repeated for these packaging/import/data-representation changes.

Reports addressed: normal lazy OpenCV import; encoded numerical strings replaced by raw verified float32 asset; development-only graph authoring tools excluded. Restored unflagged model catalog and manual source documentation. Subprocess probes remain retired; automatic downloader is only in full source distribution.

Approval remains pending until the exact version is Active. Follow-up checks request include_status_reason=true.

## Full workflow generation

Clean published 3.0.4 installation on isolated ComfyUI port 8190. Existing workflow API graph queued without missing nodes or node errors. Prompt 47d9c9d2-54ef-4397-8418-c0d9366b154a completed successfully in 113.41 seconds with no cached execution nodes. Final output: /tmp/donut-registry-fresh/ComfyUI/output/Final/4539680068827512.webp; dimensions (1728, 1344). Model merge, LoRA bypass, base generation and enabled finish stages completed. This test used the existing preset settings (UncensorFix Off); exact bundled-weight parity was covered by the earlier tests.

Follow-up 2026-09-09T08:30:23.362724+00:00: 3.0.4 NodeVersionStatusPending.

Follow-up 2026-09-09T08:38:11.000940+00:00: 3.0.4 NodeVersionStatusPending.

Follow-up 2026-09-09T08:43:41.434829+00:00: 3.0.4 NodeVersionStatusPending.

## User-requested review — 2026-09-09T09:06:08Z

Exact version 3.0.4: NodeVersionStatusFlagged, confirmed through the versions API
with include_status_reason=true. The report contains two info-level YARA findings,
both in docs/publishing.md:

- Line 25: python_command_injection_risk, matching the subprocess example in the
  table documenting previous scan triggers.
- Line 28 (also line 26 in match metadata): python_network_operations, matching
  the graph connection and HTTP request examples in that same Markdown table.

Verified the cited lines in the previously downloaded published 3.0.4 ZIP. These
are documentation examples, not executable Python. The current report lists no
runtime Python findings. The maintainer publishing guide was inadvertently
included in the runtime archive; existing exclusions cover validation notes but
not this guide. A suitable packaging change is to retain maintainer documentation
in source and exclude it from the registry runtime ZIP, or request review of these
false positives. No packaging/runtime changes or replacement publication were
made during this check. Approval remains unconfirmed because the version is
Flagged. Automatic checking remains stopped at the user's request.

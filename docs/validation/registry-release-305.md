# 3.0.5 release validation

2026-09-09T09:10:56Z: upload succeeded. Registry approval status deliberately not
queried yet, following the user's instruction to check after the noon cron job.

Packaging-only replacement built from the verified published 3.0.4 archive.
Removed docs/publishing.md, updated version to 3.0.5 and package exclusions.
All runtime code, UI, workflow and weight bytes are identical to 3.0.4. The
maintainer guide remains in repository source. The packaging guard now rejects
the guide, AGENTS.md and validation notes in a registry input archive.

Comfy CLI configuration and local security checks passed. Downloaded the newly
published 3.0.5 ZIP and compared all 157 files with the validated staging content
(excluding the CLI's local lint cache). Compared against published 3.0.4: only
docs/publishing.md removed; only pyproject.toml and .comfyignore changed.
No GPU generation was needed for this packaging-only change.

Published ZIP SHA-256:
365ec144533e3284bf88e24c8da414b3ace9913bbf00ba30cec5e815301d6a4c

Local node.zip is the verified downloaded 3.0.5 release archive.

One-time follow-up scheduled in the current task for 2026-09-09 at 12:05
Europe/Copenhagen (10:05 UTC), automation check-donutnodes-3-0-5-after-noon-review.
The user clarified noon, not midnight. No review checks before that time and no
five-minute polling. Report that requested check even if Pending, then delete
the one-time automation. Upload success does not establish registry approval.

## Approval — 2026-09-09T10:04:20Z

Exact version 3.0.5 verified as NodeVersionStatusActive with status reason
"Passed automated checks" through the versions API with include_status_reason=true.
The published download URL remains https://cdn.comfy.org/donutsdelivery/donutnodes/3.0.5/node.zip.
The registry release is approved and available through normal discovery.

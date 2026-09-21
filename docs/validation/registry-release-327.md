# DonutNodes 3.0.27

Replaces reflective argument binding in subject-mask, independent-crop and
dynamic-grounding wrappers with explicit method signatures and keyword
forwarding. This removes the runtime reflection reported as socket operations
by Registry review on 3.0.23/3.0.24. It does not rename or hide the same calls.
All existing positional parameters and defaults match the parent implementations;
SDA/NAG keyword forwarding and mask/crop/grounding behavior are retained.

157 focused tests pass: subject masks (24), independent crops (39), grounding
schedules (31), grounding/NAG (16), inpainting (18), Fusion preset (27), and
explicit-wrapper signature contracts (2). Syntax compilation and diff checks
pass. No full GPU generation was run for this backend-only refactor; restart
ComfyUI to load it. No panel binding or workflow JSON changes in this release.
The updated 3.0.26 V5 JSON remains the current Civitai upload.

Package is built from committed source, excluding unrelated unconfirmed Edit
Studio switch changes. Release commit `de04eb8` pushed to main. Registry upload
succeeded and local CLI security checks passed.

Exact version checked 2026-09-20 02:27:49 UTC: `NodeVersionStatusActive`,
with status reason `Passed automated checks`. The exact published version is
approved by Registry review.

Published ZIP matches staging per member (207 entries). No `.bind(` calls in
packaged Python; installer scripts, downloader backend, credentials and development
files are excluded. SHA-256:
`c19b2a2eb732611a03a03bd1dd5d7660d9253315b3be3fbdc548c22b678a71d5`.

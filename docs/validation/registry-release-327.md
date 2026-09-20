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
Studio switch changes. Publication and approval not yet verified.

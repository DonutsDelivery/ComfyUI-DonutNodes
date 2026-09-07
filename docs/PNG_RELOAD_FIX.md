# PNG workflow reload and Add LoRA repair

Use the existing **main** installation, restart ComfyUI after updating, and
hard-refresh the browser. No feature branch, replacement node, or manual
rewiring is required. This fix retains the native dropdowns and CivitAI panels
from PR48; it does not install the separate, unmerged DOM-editor redesign.

## Constructor failure

ComfyUI frontend 1.51.9 validates `options.values` synchronously inside
`LGraphNode.addWidget`. The dynamic filename and block-preset combos previously
added their live values getters *after* calling `addWidget`, which was too late.
Both now receive real arrays at construction, before their live catalog getters
are installed. This covers loading nonempty stacks and adding the first row to
an empty stack, even while the installed-model catalog is still loading.

The frontend tests now enforce this exact constructor requirement. The previous
mock accepted empty options and missed the production failure. Native widget
types are also left intact when hiding disabled-row or connected controls,
including widgets with getter-only types on their prototypes. Detaching and
re-adding a loader to a graph resumes its metadata lifecycle.

## Subgraph boundary persistence

The supplied PNG contained a schema-2 workflow whose stage instance saved 30
input sockets while its definition still had 37 and its value array still had
25 promoted controls. The stored cable endpoints were internally consistent
with that shorter list, so they must be remapped by **socket name**, not copied
to the same numerical index in a longer list.

`repairStreamlinedWorkflow` now normalizes tagged schemas 1, 2, and 3 at import
and at root-graph serialization. Schema 3 is a format marker, not a reason to
skip subsequent checks. This supersedes the earlier schema-2 skip behavior.
The export guard works on detached serialized data, not live socket arrays,
and therefore covers workflow JSON and PNG metadata without rewiring a live
canvas during serialization.

The repair supports nested instances, array- and object-form links, and input
and output remapping. It does not depend on the original node IDs or layout.
Unknown socket names, contradictory link records, missing values, and
conflicting named/positional values abort before any mutation. Other workflows
and unknown future schema versions are left alone. Sampling algorithms,
model settings, prompts, row data, positions, and dimensions are not reset.

## Validation

Run the public tests with Node.js:

```sh
node --test tests/test_streamlining_frontend.cjs tests/test_workflow_reload.cjs
```

There are 43 tests: 41 run without private fixtures; two explicitly skip.
To include both private regression cases, set `DONUT_BROKEN_WORKFLOW` and
`DONUT_FIXED_WORKFLOW` to the original PR47 workflow pair, and
`DONUT_RELOAD_WORKFLOW` to the JSON extracted from the user's later PNG.
All 43 passed in preparation. The new constructor test reproduces the exact
reported exception against the old PR48 implementation and passes with this
fix. Tests retain metadata, preview, async-race, disabled-row, serialization,
and native-control coverage; they also test nested scopes and repeated saves.

The separately delivered recovered workflow retained all 52 serialized nodes
and 173 semantic cable connections. An offline traversal compared all 301
recorded execution input values/connections with the PNG's `prompt` metadata;
there were no differences. The private PNG/workflows/prompts are not committed.

These are strict frontend contracts and offline graph checks, not a complete
live ComfyUI browser/server or GPU test. CivitAI responses in unit tests are
fixtures. No output-image parity or GPU-performance claim is made. The separate
Fusion Control/LoRA ordering warning is unchanged by this frontend repair.

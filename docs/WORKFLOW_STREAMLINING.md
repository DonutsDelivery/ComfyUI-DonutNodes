# Workflow streamlining

This targets the supplied `ComfyUI_00016_.json`. It consolidates glue nodes,
not the sampling algorithms: model loading/merging, text authoring,
conditioning, sampling, each upscale, face detailing and saving remain distinct.
The original reduction is 68 to 40 serialized nodes (including subgraph proxies),
and six fewer external node packs. This is not a measured GPU speedup or a claim
of pixel-identical output.

## Update and load

Use **main** in the existing DonutNodes checkout:

```sh
git pull --ff-only origin main
```

Restart ComfyUI, then hard-refresh the browser to load the changed JavaScript
and its new modules. Existing Donut LoRA Loader and Dynamic LoRA Stack nodes
update in place; do not delete/recreate them. Their inputs and outputs did not
change. Continue using `ComfyUI_00016_Donut_Streamlined_Fixed.json` supplied
separately. No further workflow changes are required for the native LoRA UI.
Model files, wildcard files, the optional source image and retained packs must
still be present on the local installation.

PR47's original schema-1 migration kept obsolete promoted text values, shifting
later parent-subgraph controls on frontend 1.51.9. The extension now repairs that
specific migration before graph configuration: it rebuilds sockets by name and
serializes only actual promoted widgets, producing 37 sockets and 25 widget
values for the supplied graph. Missing original named values cause a clear
error instead of guessing. Schema-2 saves are left untouched, so later user
edits are not overwritten with old defaults. Workflow files are not committed
to the public repository.

## Native LoRA rows

The editor uses ComfyUI's native `combo`, `toggle`, `number`, `text` and `button`
widgets. There is no HTML datalist, select element or custom dropdown menu.
Installed filenames come directly from `/models/loras`, with the old Donut
LoRA Stack's standard node schema as a fallback. That schema also supplies the
block presets. The editor does not depend on custom STRING metadata surviving
frontend schema normalization. A refresh button reloads the installed list
without replacing any saved selections. Loading/failure states are visible.

Block preset menus contain short names rather than full comma-separated
vectors; selecting one still writes its full vector to that row. Model-type
filtering does not silently alter the saved preset or block vector.

Each disabled row keeps its filename picker, Enabled toggle and Row actions.
Strengths, advanced block controls and information panels are hidden and the
node refits its height. Re-enabling restores their values unchanged. Row
actions provide move up/down, removal, metadata retry and explicit application
of a suggested model weight. Add LoRA has no fixed three/six-slot limit.

All rows remain in one canonical `slots_json` backend input with stable IDs.
Reordering moves the whole row, including disabled state, separate model/CLIP
strengths, vector/preset, global inheritance, hash and unknown saved fields.
Native row controls and information panels are UI-only: they are excluded from
both positional workflow arrays and API prompt inputs. Malformed JSON is
retained for repair instead of being replaced by an empty stack. The original
fixed three-slot node remains registered for existing workflows.

### Information is displayed, not just fetched

Enabled, selected rows have a bounded, scrollable information panel independent
of the advanced block-settings toggle. It shows local weight composition
(model/text-encoder components, populated block indices and tensor count),
current strengths, CivitAI model/version, base model and author, suggested
weight, trigger words, description, hash, a clickable CivitAI version link and
cached preview images. Clicking a preview opens its full-sized local image.
Descriptions are inserted as text; links are built from model IDs or hashes.

Local composition uses the existing safetensors-header analysis endpoint.
Turning CivitAI lookup On or selecting an enabled LoRA requests its metadata
immediately; no image generation is needed just to populate the panel. Off
prevents new UI CivitAI lookups, while local analysis remains available and
previously cached information may still be displayed. The existing server's
CivitAI configuration/cache and preview handling are reused; no new credentials
or dependencies are introduced. Loading, unsupported files, HTTP errors,
not-found results and unavailable previews are shown explicitly. Row actions
includes Retry metadata. Requests are deduplicated and limited to two in flight.

Suggested weights are existing CivitAI example/cache hints, not proof of an
optimal value (the cache can default to 1). They never overwrite strengths
without an explicit row action, and applying one leaves CLIP/text-fusion strength
unchanged. Filename edits clear stale hashes; late responses cannot attach to
another selection. A ten-character lookup hash does not overwrite a full hash.

The backend builder/applier is unchanged. It still resolves the entire dynamic
stack through existing Donut code and calls the safe applier once, preserving
whole-stack budgeting, duplicates, fused-text strengths and block behavior.

## Other boundaries

Three Donut Text editors replace eleven wildcard processors. Each editor
resolves nested TXT wildcards and choices with bounded recursion, local RNG,
cycle/path/size guards, file cache invalidation and explicit missing-file policy.
Resolved text is a read-only, collapsible preview; advanced text settings can
also be collapsed without changing their values. Prompt conditioning keeps
full versus face text and raw versus zeroed negative conditioning separate.

The seed plan keeps independently controlled text, sampler-master and filename
seeds. Sampling stage offsets are 0, 2, 3 and 4 modulo 2**53, intentionally fixing
the old second-upscale/face collision. Grouped body/fusion merge controls retain
all per-block values. Upscale enable switches retain the existing algorithms
and lazy pass-through when disabled.

Removed workflow packs: mikey_nodes, RES4LYF, cg-use-everywhere,
derfuu_comfyui_moddednodes, ComfyUI_Comfyroll_CustomNodes and rgthree-comfy.
Retained: ComfyUI-bleh (configured sampler), WAS Node Suite (saving), Impact Pack
and Subpack (SAM/detection/detailing), and krea2-nag (adjustable NAG).

The separate fusion-aware warning is not suppressed: the safe applier needs a
model carrying Fusion Control metadata upstream. This UI fix does not change
that model execution order.

## Validation

```sh
node --test tests/test_streamlining_frontend.cjs
# Also test the supplied private graph pair (not committed):
DONUT_BROKEN_WORKFLOW=/path/to/ComfyUI_00016_Donut_Streamlined.json \
DONUT_FIXED_WORKFLOW=/path/to/ComfyUI_00016_Donut_Streamlined_Fixed.json \
node --test tests/test_streamlining_frontend.cjs
```

The native-UI update has 25 passing regression tests with the graph pair
provided (24 pass and one skips without it). They exercise the actual extension
against a small native-widget/DOM harness with mocked HTTP responses, including
catalog fallback, native widget types, collapsed rows, JSON/positional/API
serialization, metadata content/links/previews, stale results, request limiting
and the original shifted-widget failure. Existing Python backend tests remain
unchanged. These tests do not constitute a complete ComfyUI browser session,
a live CivitAI API test, GPU inference, output-image parity or a benchmark.

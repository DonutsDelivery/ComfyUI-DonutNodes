# Workflow streamlining

This change targets the supplied `ComfyUI_00016_.json`, not arbitrary graphs.
It removes glue nodes while keeping model loading/merging, prompt authoring,
conditioning, sampling, each upscale stage, face detailing and saving separate.

## Result

Across the root and both original subgraphs: **68 -> 40 nodes**, including
subgraph proxies. Six external node packs are no longer referenced by the
migrated workflow. This is a graph/dependency reduction, not a measured GPU
speedup or a claim of pixel-identical output.

| Original | Replacement |
| --- | --- |
| Eleven wildcard processors | Three Donut Text editors, each recursively resolves its own text |
| Two three-slot LoRA stacks + apply + global-vector primitive | One Donut LoRA Loader with six preserved, editable rows and add/remove/reorder |
| Five seed generators + filename seed string | One Donut Seed Plan with independent text, sampler and filename controls |
| Concatenation, edit-negative text, three encoders, negative zeroing and display | One Donut Prompt Conditioning node with distinct full/face and raw/zeroed outputs |
| Two Anything Everywhere controllers | Four explicit, checked links |
| Group bypass controller | First/second-upscale enable switches, with lazy disabled execution |
| Merge subgraph with two primitive fan-outs | Grouped body/fusion controls on the existing merge node; per-block controls remain available |

Removed packs: `mikey_nodes`, `RES4LYF`, `cg-use-everywhere`,
`derfuu_comfyui_moddednodes`, `ComfyUI_Comfyroll_CustomNodes`, `rgthree-comfy`.

Intentionally retained: `ComfyUI-bleh` for the exact configured sampler preset,
`was-node-suite-comfyui` for the configured WebP/history/naming/metadata saver,
`comfyui-impact-pack` and `comfyui-impact-subpack` for SAM/detection/detailing,
and `krea2-nag` for adjustable NAG. Hiding these imports in wrappers would not
remove a dependency. Replacing these algorithms without validating parity
would conflict with the requirement to retain functionality.

## Install and load

Use the PR branch `feat/workflow-streamlining` in the existing DonutNodes
installation, restart ComfyUI, then refresh the browser to load the new
extension. Load the separately supplied `ComfyUI_00016_Donut_Streamlined.json`; no manual rewiring is
required. The ordinary model files, source image, wildcard text files and the
five retained packs must still be present. Do not replace the whole DonutNodes
installation with only the files changed in this PR.

For reproducibility, the guarded migration can regenerate the graph from the
original uploaded JSON:

```sh
python tools/streamline_workflow.py /path/to/ComfyUI_00016_.json workflows/ComfyUI_00016_Donut_Streamlined.json
```

The source is never overwritten. Unknown layouts or incompatible link sources
fail rather than silently dropping them. The original workflow remains usable
with its original packs; the old three-slot LoRA nodes remain registered.

## Text and seed behavior

Donut Text supports TXT wildcards in the existing ComfyUI user/root wildcard
locations, registered wildcard directories and this pack's `wildcards` folder.
It supports nested `__file__` references, `{a|b}` choices, `<random:min:max>`,
`N$$__file__`, repeated-name `!`, `+`, `-`, `*` modifiers, whole-word OR filters,
subfolders, date macros and node/widget macros. File-line selection and
per-pass seed restarts follow the source processor's convention. Blank lines
and comment-looking lines still count. The default nesting limit is 128,
configurable up to 1024, with cycle, expansion-count and output-size guards.
Missing files default to a clear error rather than disappearing; `keep` and
legacy-style `empty` policies are available. Paths cannot escape wildcard
roots. RNG state is local, and file changes invalidate cached results.

All authoring text editors, prompt injection and the editable negative text in
conditioning receive the same **text seed**. Already-expanded face/scene text
is not expanded again in conditioning. The face prompt is not silently changed
to the combined scene prompt. Raw negative conditioning still reaches NAG;
zeroed negative conditioning still reaches the sampling stages.

Sampling uses a separate master seed. The four stage seeds are master plus
`0, 2, 3, 4`, modulo `2**53`, so they remain distinct and lossless in JavaScript.
The old second-upscale/face collision is intentionally fixed. This changes the
face stage's seed from the previous master+3 to master+4. Text and sampler
controls start with the source's current seed but can now be frozen/randomized
independently. Filename randomness has its own retained control. Per-face and
per-model variation inside the existing samplers is not rewritten.

## LoRA behavior

Rows have stable IDs and are serialized as one canonical JSON value, not as a
variable list of positional widgets. Reordering/removal therefore moves the
entire row, including model/CLIP strengths, block preset/vector, global-vector
inheritance, disabled state and hash. There is no fixed three/six-slot ceiling;
a two-megabyte configuration limit prevents unreasonable inputs. The editor
preserves malformed saved JSON for repair rather than replacing it with `[]`.
Filename edits clear stale hashes; late execution results cannot attach a hash
to a different filename. CivitAI per-row information/previews are retained.

The existing Donut stack builder handles resolution and metadata in chunks;
**the existing safe applier is called only once, on the complete stack**. This
preserves whole-stack energy budgeting, duplicate handling, text-fusion weights,
block weighting, safety options and experimental bypass behavior. The separate
Donut Dynamic LoRA Stack node remains available for workflows that need a
reusable stack without loading a model at the same boundary.

## Compatibility and validation

Existing Tiled Upscale defaults to enabled, and existing Krea2 merge defaults to
Per block. Their original algorithms remain the implementation. Prompt
Injection retains its original type and style controls, with recursive output
processing added. Registration uses Donut's existing isolated-import mechanism;
a failed override is reported rather than silently exposing an incompatible
older implementation.

Run the dependency-free tests from the repository root:

```sh
DONUT_WORKFLOW_SOURCE=/path/to/ComfyUI_00016_.json python -m unittest discover -s tests -p test_streamlining.py -v
node --test tests/test_streamlining_frontend.cjs
```

At preparation: **45 Python tests and 11 JavaScript contract tests pass**.
Python checks reciprocal links/types, subgraph boundaries, acyclicity,
parameter and bypass preservation against the original upload, seed routing,
recursive wildcard safety and mocked legacy delegation. JavaScript executes
the extension against a small DOM/Comfy harness, covering row operations,
serialization, repair, hash races, presets, seed controls and grouped widgets.
The eight real-workflow tests require `DONUT_WORKFLOW_SOURCE`; without it they
are explicitly skipped and the 37 other Python tests still run. No workflow
fixture is committed; the workflow files are delivered separately.

Not performed: a real ComfyUI browser import/queue, model loading, CivitAI
network access, GPU inference or image-output parity testing. The PR is a draft
until those are checked. In particular, test both upscale toggles, editing on
and off, NAG enabled, LoRA add/reorder/save/reload, and a fixed-seed queue with
real nested wildcard files. No claim is made that CPU mocks prove GPU parity.

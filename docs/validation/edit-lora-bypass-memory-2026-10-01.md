# Single edit-LoRA bypass chunking, 2026-10-01

Follow-up to draft PR #77, head `12f2ab1` before this change. The earlier
MLP/normalization optimization is documented in
`edit-mode-memory-2026-09-30.md`.

## Scope and control path

`DonutEditStudio._load_edit_lora` and
`krea2_edit_integration.apply_krea2_edit_lora` already resolve the inherited
global execution mode and call `_apply_bypass_applications` for Experimental
bypass. The resulting injection plan is shared with ordinary Donut LoRAs.
This change introduces no edit-only execution mode.

Previously, `_register_bypass_adapters` left one adapter per target native and
wrapped overlapping adapters in `_CompositeBypassAdapter`. Runtime injection
applied `donut_chunk_lora` only to that composite. A lone plain LoRA therefore
ignored the existing chunking option, including an identity-edit LoRA without
overlapping adapters on its target layer.

Runtime injection now wraps a copied, exact native `LoRAAdapter` in a
one-component composite when `donut_chunk_lora` is true. The child strength is
one; the native hook retains the original strength, including negative and zero
strengths. Canonical adapter registration, attachment data, and save/extraction
inputs stay native. Runtime weight conversions remain separate from those
canonical inputs.

The existing inference policy applies: nonconvolutional `B,T,C` inputs longer
than 1024 tokens use 1024-token chunks. Small, empty, two-dimensional, and
autograd-enabled inputs retain a complete adapter call. Convolutions retain
their native spatial operation. Other adapter types are not newly wrapped;
existing LoKr and overlapping-adapter behavior remains intact. Disabling or
omitting the option keeps a singleton native. No panel, default, workflow JSON,
reference resolution, attention operation, or precision setting changes.

## Executed checks

Local environment: PyTorch `2.5.1+cu121`, NVIDIA GeForce RTX 4070 (12 GB).
Native adapter and bypass-hook source:
`/home/user/Programs/ComfyUI-new/ComfyUI`, revision
`856a922befab9d94cb66f36a3dce17234d7a6e31`.
Tests load those actual source files with explicit ModelPatcher ownership and
device-selection doubles; they do not start a full ComfyUI application.

Each command ran with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`:

```sh
python -m unittest test_lora_bypass_memory -v
python -m unittest test_krea2_memory test_krea2_edit_memory -v
python -m unittest test_lokr_bypass_parity -v
python -m unittest test_safe_lora_stack test_donut_lora_execution -v
```

Results: **73 passed, none skipped**, including two synthetic CUDA allocation
checks. Compilation and `git diff --check` also passed.

The 11 new single-LoRA checks cover actual native hook output versus materialized
weights, strength/alpha variants, noncontiguous input, chunk and base-output slice
sizes, native disabled behavior, on/off/on transitions on a shared root, repeat
injection, a different sampling root, canonical weight/strength preservation,
gradient fallback, convolution geometry, small/empty/2D calls, other adapters,
and automatic ejection when the sampling owner is collected.

Replacing only the runtime injection function with its pre-fix body from
`12f2ab1` and running the new tests produces seven failed assertions (including
subtests) and one error. The error expects a runtime composite where the old
code returns a native singleton. The CUDA comparison then has equal peaks,
confirming the old flag makes no difference for a singleton. No production
source was changed during that baseline comparison.

## CUDA measurements

Both paths were warmed and synchronized. Values below are incremental peak
**allocated** bytes above each call's baseline, not total or reserved VRAM.
Numerical output comparisons passed.

| Synthetic operation | Native bytes | Chunked bytes | Reduction |
| --- | ---: | ---: | ---: |
| Layerwise SwiGLU, input `1025,12,256`, hidden width 768, 64-row chunks | 76,619,776 | 18,362,368 | 55.56 MiB |
| Full linear bypass hook, input `1,8193,256`, output width 2048, rank-128 LoRA, 1024-token chunks | 205,545,984 | 201,351,168 | 4.00 MiB |

The bypass result is modest because the base output, full adapter output, and
final sum still require full-size buffers. This patch bounds adapter
intermediates; it does not eliminate those outputs or bound attention memory.
The numbers are operation-specific and must not be extrapolated into total
Krea2 VRAM savings.

## Remaining validation

No full Krea2 checkpoint/reference encoding, base edit, hires edit, or face
detail generation was run. There is no new output PNG or Run-button/reload
evidence, no latency benchmark, and no verified fix for the reported two-reference
12 GB OOM. The prior unchanged-workflow GPU procedure still applies. Both
`donut_chunk_edit_mlp` and `donut_chunk_edit_norm` must be absent or true to
exercise their corresponding MLP/norm patches; explicit false remains an opt-out.
The edit-LoRA check additionally requires Experimental bypass and
`donut_chunk_lora=true`, with logs confirming its targets did not fall back to
regular patches.

Work took place in an isolated worktree, preserving local main and the user's
uncommitted panel/workflow changes. PR #77 remains draft. No merge, release, or
Registry publication was performed.

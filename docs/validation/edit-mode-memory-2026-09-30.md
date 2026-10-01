# Edit-mode memory optimization, 2026-09-30

Base Donut revision: `2806ebe12caada993e6305e3f441ce61ab22a11c`.
Upstream layout inspected: ComfyUI `fb2315f11db0ebfaafa9099a5df5227dc6bb42bc`,
`comfy/ldm/krea2/model.py` (`TextFusionTransformer`, `SwiGLU`, and `RMSNorm`).

## Changes

The existing sampler, edit upscale, and detailer call sites already call
`patch_krea2_upscale_memory`. Previously, the helper defaulted both memory flags
to false and only covered diffusion-block MLPs and Q/K norms when opted in.

The helper now recognizes a nonempty `donut_krea2_edit` or legacy `krea2_edit`
wrapper in either Comfy wrapper store. Only an active edit forward automatically
enables chunking; ordinary generation and an unused Edit Studio branch do not.
Explicit `donut_chunk_edit_mlp` and `donut_chunk_edit_norm` options remain
independent. False restores the corresponding owned wrappers on a clone;
re-enabling does not nest them. Repeated application returns the same model when
no changes are needed. Existing custom forwards and linear-layer hooks remain
inside the wrapper.

Coverage now includes diffusion and text-fusion MLPs, pre/post RMSNorm, and Q/K
RMSNorm. Diffusion/refiner pointwise operations retain 1024-token chunks.
Text-fusion layerwise operations use 64 rows of the flattened batch axis: upstream
uses `(B*T),layers,C`, not `B,T,C`. At 12 encoder layers, an MLP call therefore
handles at most 768 layer tokens. Layerwise Q/K norms also split axis 0, retaining
all heads and all encoder layers for each row.

Attention itself is not split, replaced, or tiled. There is no reference
resizing, grounding reduction, new cache, precision switch, workflow migration,
or global model mutation. Autograd-enabled calls retain the original forward.
This bounds pointwise intermediates, not total model/attention memory.

## Executed validation

Environment: PyTorch `2.10.0+cpu`; CUDA unavailable. The local-machine connector
was offline. The helper and original memory test were reconstructed from pinned
GitHub reads and verified against their Git blob SHA-1 values before testing.
Only this focused test subset was materialized, not the entire repository.

Command:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest test_krea2_memory test_krea2_edit_memory -v
python -m py_compile krea2_memory.py test_krea2_memory.py test_krea2_edit_memory.py
git diff --check
```

Results: **19 tests passed; one CUDA-only test skipped**. Compilation and diff
checks passed. Running the same test set against the original helper produced
8 failures and 1 error, confirming that the new activation/coverage assertions
exercise behavior absent from the baseline.

Tests cover both wrapper stores and edit keys, non-edit no-op behavior, independent
opt-outs, on/off/on transitions, clone isolation, idempotence, upstream-forward
restoration, gradient fallback, noncontiguous BF16 normalization, and prompt
adapter hooks. A small text-fusion model with upstream-compatible tensor layouts
matches the unchunked output within `rtol=1e-5, atol=1e-5`; its layerwise MLP hook
observes 64/64/2 rows instead of 130. Attention still observes complete
`(130,12,8)` layerwise and `(2,65,8)` refiner inputs. Weak-reference checks verify
that individual chunk results are released before the next forward call.

The CUDA-only synthetic MLP test warms both paths, synchronizes the device,
compares outputs, and asserts lower incremental peak allocated bytes. It has not
been executed here and is not a full Krea2 workload benchmark.

## Verification limits and GPU follow-through

No real ComfyUI generation, Run-button UI test, output PNG, peak VRAM measurement,
or latency measurement was possible in this environment. The full repository
suite and installed LoRA/NAG/Fusion combinations were not executed. Chunking may
change floating-point rounding and increase kernel-launch overhead; numerical
unit parity is not a bit-identical image guarantee.

Before marking GPU validation complete, benchmark a separate workflow/session
with identical weights, seed, resolution, references, grounding, and LoRA/NAG
settings. Compare both options explicitly false against the automatic edit path.
Warm each variant, synchronize around timing, and record peak allocated and
reserved VRAM for base sampling, full-frame upscale, and face detail separately.
Exercise one/two references and NAG off/on, then queue through Run and inspect
saved output and metadata. Restart the backend to load the changed Python.

In particular, retry the documented 2112x1056, two-reference, NAG, FP8/bypass-LoRA
workload from `prompt-mask-gpu-2026-09-19.md`. This change is **not** a verified cure
for that 12 GB OOM. The separately observed edit-forward projection lifetimes and
positional-encoding allocations were not changed in this patch.

No release, registry publication, or distributed workflow JSON change was made.

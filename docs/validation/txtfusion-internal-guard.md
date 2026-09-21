# Checkpoint-referenced internal txtfusion guard (experimental)

Base: `aa2571e3fc4d13380eee4ff2baea6ae84b6c8f51`.
This PR does not merge either parked branch, publish a release, or claim a fix
for Rebalance + UncensorFix + NAG blotching.

## Intervention

For an adapter-modified txtfusion attention or MLP component, evaluate both the
live component and an isolated checkpoint component on the SAME incoming
activation `h`. Compute one RMS ratio per original sample, clamp to `[0.25, 4]`,
and scale the live component's contribution BEFORE adding it to the residual:

```
u0 = checkpoint_component(h)
u1 = live_component(h)
x_next = x + u1 * clamp(RMS(u0) / RMS(u1), 0.25, 4)
```

Zero-energy cases pass through; nonfinite values raise. Statistics reduce over
all actual non-batch elements. The flattened `batch * text_length` layerwise
layout is restored to the original batch grouping for this statistic. No
synthetic padding is introduced. Each NAG text call has its own reference and
statistics. This is not normalization of the residual sum, final txtfusion
output, conditioning, LoRA weight matrices, or the positive/negative midpoint.
The guard has no dependence on NAG alpha.

The reference comes from selected tensors in a safetensors file, not from live
weights, ModelPatcher backups, temporary unpatching, or a deepcopy of a patched
module. Fresh native Comfy Attention/SwiGLU components receive independent file
tensors and use manual-cast operations. The live adapter path is not converted
between native patches and bypass execution. The checkpoint projector is not
guarded; affected attention/MLP components use their full pretrained reference.

## Scope and limitations

Default **Off**, and only DonutSampler's simple non-edit NAG first pass can enable
this prototype. Upscale code, Rebalance, NAG equations, sampler/scheduler, step
count and the previous experiments are unchanged. Disable midpoint compensation
and batching to isolate this experiment. No-adapter/zero-strength models bypass
the guard without reference I/O. SDA, editing, multi-model modes, model injections
other than recorded Donut bypass, custom txtfusion/block forwards/hooks and
quantized/dynamic-parameter txtfusion are explicitly rejected rather than
silently substituted. Arbitrary txtfusion LoRAs are not yet image-validated.

**Reference support is initially plain FP16/BF16/FP32 txtfusion tensors.** Packed
or FP8 txtfusion, including scaling/quantization metadata, is not supported in
this PR. Body-only quantization is separate. Do not switch execution mode or
use a different reference model merely to get past a check.

The reference dropdown must select the **same diffusion checkpoint** used to
load the base model, before additional adapters. Shape checks cannot prove this
identity. The file name and SHA-256 digest of selected reference tensors are
logged; the code does not assert it inferred the original loader's filename.

References stay on CPU and are cast per component call. This adds checkpoint
I/O, CPU memory, transfer overhead and extra component evaluations; no full-size
latency/VRAM claim is made. No persistent cache or model-manager memory budget
optimization has been implemented. Inferences are not training-supported.

## Integration and V5 panels

The existing DonutSampler node ID is overridden after the grounding sampler.
Two OPTIONAL widgets are appended, leaving all inherited widget positions
unchanged: `txtfusion_internal_guard` and `txtfusion_reference_checkpoint`.

On workflow load, the **Seed & guidance** panel gains an advanced group named
**Experimental · checkpoint txtfusion guard**. Its controls bind to the actual
inner DonutSampler, not a promoted outer widget. A single owner in the same
panel/seed family is required; ambiguous A/B graphs are not guessed. Existing
settings are neither copied nor reset, repeated migration is idempotent, and
only this extension's tagged group is repaired. No shipped workflow JSON changes
are included, so there is no Civitai workflow JSON replacement in this PR.

A proxy is supplied to the existing NAG wrapper on a model clone. Only the
proxy's txtfusion call changes. The NAG function/module, original model, live
weights, component forwards and upstream registry are not globally modified.
The wrapper registry order is preserved. A stale prepared runtime copy of the
same NAG key is removed only on the clone before Comfy prepares the new run.
The run releases references in `finally`, including interruption and logging
failure. No installed hook or in-place weight swap requires restoration.

## Validation performed

Commands from the repository root:

```
OMP_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_txtfusion*.py' -v
node --test tests/txtfusion_guard_controls.test.cjs
```

34 Python tests and 10 JavaScript tests passed in the development environment
(PyTorch 2.10.0 CPU, Python 3.13, Node 22). Python tests use real small tensors,
a real safetensors file, and explicitly identified Comfy interface/component
doubles. Nonempty attention/MLP blocks run; tests are not just registration
assertions or a zero-block forward. They exercise reference separation after
simulated live weight materialization, internal residual placement (effect
survives a following RMSNorm), original-forward parity, batch independence,
wrapper dispatch, unchanged live weights, native/bypass target detection,
unsupported paths, schema order, flag routing and cancellation/error cleanup.
JavaScript tests exercise nested bindings with real target widget names,
promoted controls, live/serialized subgraphs, rerender callbacks, repeated
reloads, existing values, multiple owners and cross-family isolation.

**Not performed:** complete ComfyUI runtime loading, full repository suite,
actual DynamicVRAM/bypass adapters or quantized kernels, GPU inference, checkpoint
image-quality comparison, browser Run-button execution, or PNG metadata
round-trip. Therefore the PR remains draft. Passing the focused tests does not
prove it fixes the user's images or that every third-party wrapper is compatible.

## Remaining acceptance check

In a separate audit workflow/session after restarting the backend and refreshing
the frontend, select the same reference checkpoint in the V5 panel. Keep NAG
alpha **0.45**, eight-step Turbo, existing Bleh preset/beta, Rebalance and
UncensorFix unchanged. Keep the older two experiments Off. Compare guard Off
versus On with the same seed/expanded prompt. Save the base-pass decode and PNG
workflow/execution metadata; cancellation after that preview is valid and does
not invalidate a completed base pass. Inspect the log for reference identity,
nonzero contribution calls and actual RMS/gains. Confirm panel values agree with
queued backend values. Repeat Off after On and after a cancelled run. Repeat
several known failing seeds and check whether UncensorFix's intended effect
survives, not merely whether artifacts are reduced.

Native model contracts inspected: ComfyUI `comfy/ldm/krea2/model.py` (blob
`8001812d77d2750dbadf25fbec16710ea1ba3e74`) and upstream
`iljung1106/ComfyUI-Krea2-NAG` wrapper API. These are runtime dependencies, not
vendored model weights. Changes to those contracts require retesting.

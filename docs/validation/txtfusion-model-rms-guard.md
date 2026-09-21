# Model-wide txtfusion RMS guard

Base revision: `57acc12b2dad50f80146c0964d9879eaf9c19787`.
This supersedes the NAG-bound activation design of PR #71. It does NOT claim
that checkpoint-referenced normalization has been shown to fix the user's
blotching. The earlier midpoint experiment is a different algorithm.

## User-facing behavior

The new **Models > Txtfusion RMS guard > Normalize txtfusion changes · all
stages** checkbox controls the MODEL produced by Fusion Control. Default Off.
The same guarded MODEL can be used by base generation, stock KSampler, both
upscales and the face detailer. There is no dependency on NAG, NAG alpha, phi,
sigma window, editing, SDA or sampler selection. NAG alpha 0 stays 0: it does
not disable this guard and is never quietly changed to a positive value.

A standalone **Donut Txtfusion RMS Guard (model-wide)** node is also provided.
Place it after model merging. V5's Fusion Control already follows the model
setup/merge. A reference checkpoint dropdown is not needed: the reference is
the effective loaded checkpoint/merge BEFORE adapters, resolved per parameter.
Primary/model2 mixtures in the same attention/MLP component are permitted.
Partial standard/Donut merges are reconstructed with Comfy's merge arithmetic.

The old sampler checkbox remains readable, at its original widget position,
and delegates to the same NAG-independent model guard for that sampler's input
model(s). It is not the new all-stages control. Its old filename field is
explicitly deprecated and unused; a non-None value produces a warning, never a
file read. No old flag is silently cleared, and a false legacy sampler flag
does not disable an upstream model-wide guard. Both controls On do not stack
normalization. The current saved V5 JSON does not need a Civitai replacement;
new model-panel aliases are added on load without copying widget values.

## What is normalized

Normal forward hooks observe native txtfusion attention/MLP component outputs
BEFORE residual addition, plus the projector output before refiner blocks.
For each original batch item, with the SAME incoming component activation:

```
r = RMS(checkpoint_component(h))
p = RMS(live_component(h))
gain = clamp(r / p, 0.25, 4)
returned_live_contribution = live_component(h) * gain
```

Zero-energy cases pass through, and nonfinite outputs raise rather than hiding
NaN/Inf. The gain bound means exact RMS matching is not promised outside that
range. Different examples/prompts do not share statistics. The layerwise
`batch * text_length` dimension is grouped back into original batch items.
There is no shared positive/negative midpoint and no compensation-alpha knob.

This normalizes component ACTIVATIONS, not LoRA files or weight-matrix RMS.
It can change or attenuate a LoRA's intended effect, not only its artifacts.
Rebalance's tap operation and conditioning inputs are not rewritten, and none
of the sampler, scheduler, VAE or NAG equations are changed. Guarding the
projector is intentional: a projector-only txtfusion LoRA must not be ignored.

## Reference and quantization

Fusion Control establishes the reference before the parent applies embedded
UncensorFix. Earlier native adapters are excluded from Comfy's checkpoint
recipes, including adapters nested in a model-merge source. Regular merge
payloads and their model1/model2 ratios are retained. Donut's hard-swap plans
resolve the actual owner of each individual Linear, its bias and buffers;
there is no arbitrary choice of one reference checkpoint for a mixed component.

Independent copies retain the operation class, packed weight format and its
scales/buffers. Comfy-kitchen's `clone()` alone shares layout Params, so the
implementation separately clones quantization metadata tensors as well as
qdata. The reference cannot inherit a late bypass forward, a DynamicVRAM
`_v`/`_prefetch` cache, or a materialized LoRA weight function. Core backup/hook
backup precedence is used through Donut's existing `_get_merge_key_patches`
helper. There is no shared live-weight swap or retained live bound forward.

Only txtfusion is duplicated, not the entire diffusion model. It is registered
as an additional ModelPatcher so Comfy's model loading/offloading accounts for
its weights. It adds reference memory, checkpoint reconstruction, transfers and
component evaluations; full-size VRAM/latency have NOT been benchmarked.

This handles native and recorded Donut bypass additions without changing the
selected execution mode. Quantized hard swaps keep the native operation class;
partial quantized merges use the format's core requantization method. No
blanket FP8/packed/dynamic parameter class rejection is used. A genuinely
unavailable/meta baseline or unrecognized executable source injection still
raises: pretending those weights are a clean checkpoint would be incorrect.
Place a standalone guard after merging and before unknown custom injections.
Arbitrary third-party merge encodings, compiled graphs and all quant formats
have not been validated merely by passing the CPU contracts.

## Lifecycle and scope

ModelPatcher ON_PRE_RUN installs ordinary module hooks; ON_CLEANUP and ON_DETACH
remove them. An OUTER_SAMPLE finally block is a backup when another cleanup
callback raises. There is no replacement of a NAG wrapper or diffusion forward,
and no extra model injection that forces Donut LoRAs out of bypass mode.

Callbacks resolve the FINAL sampling clone rather than capturing the original
loader model. Batch-call state is context-local; live weights, input tensors and
pre-existing hooks are not rewritten. The runtime checks that all native
components were observed, rather than reporting success on a skipped guard.
Garbage collection, interruption inside txtfusion and logging failures remove
this feature's hooks. Reference weights persist with the guarded model for
reuse, and disappear with it; per-run activations are not retained.

## Validation performed here

Environment: Python 3.13, PyTorch 2.10.0 CPU, Node 22. No ComfyUI installation,
checkpoint or GPU is available in this development environment.

```
OMP_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_txtfusion*.py' -v
node --test tests/txtfusion_model_guard_controls.test.cjs
```

The changed/new Python suites have **57 passing tests and 2 explicit skips**.
The two skipped tests require genuine Comfy/Kitchen and are NOT counted as
passes. JavaScript has **11 passing tests**. The unrelated pre-existing guard
suites/full repository suite were not executed in this partial local checkout.

Python tests use actual nonempty torch attention/MLP/residual forwards, actual
scaled FP8 and torch INT8 arithmetic, with clearly identified Comfy patcher,
merge and dispatch doubles. Covered: no-adapter exact parity; native
materialized-patch backup separation; later LoRA changes; bypass hooks; scalar
and directional adapters; projector-only changes; independent batch/unequal
text lengths; alpha 0/plain and active-NAG-style dispatch; edits/SDA/multi-model
argument preservation; repeated upscale/detailer clone runs; per-linear model2
swaps; partial regular merges; quantization metadata non-aliasing; stale vbar
cache removal; Off/On/Off; callback, cancellation, logging and GC cleanup.

JavaScript tests cover actual widget names, nested and promoted paths,
serialized/live graphs, remapped IDs, ambiguous A/B ownership, family isolation,
repeated reloads, existing values and calls to the existing panel renderer.
The baseline __init__.py was checked against the Git blob before modifying its
registration list; the patch only adds the new Fusion Control override/node.

`tests/test_txtfusion_model_guard_comfy.py` adds small native Krea2/ModelPatcher
and real Kitchen metadata checks for execution inside an actual ComfyUI install.
Those tests were skipped here. They do not require image generation but still
do NOT constitute a GPU-kernel or UI end-to-end test.

## Remaining end-to-end gate

Keep the normal V5 merge, quantization and adapter execution mode. Use the NEW
Models checkbox, not the old sampler-local option, and keep the two earlier
midpoint/batching experiments Off. Check the model-wide switch Off/On at the
same seed first with NAG 0.45, then with NAG disabled/alpha 0. Verify nonzero
`[Donut txtfusion RMS]` contribution counts in BOTH cases. Compare the saved
base-pass decode, then enable an upscale/detailer to verify that the same model
setting reaches those calls too. Cancellation after a completed base preview is
valid; also test Off after cancellation and save/reload the workflow.

Use ComfyUI's actual Run button and save workflow/prompt PNG metadata to verify
the panel value reaches the backend. Native GPU loading, real quantized kernels,
full-checkpoint output quality, all third-party hooks and the full repository
suite remain unverified. Do not report this as a proven grain/blotch fix or
publish a release solely on the focused CPU/JS tests.

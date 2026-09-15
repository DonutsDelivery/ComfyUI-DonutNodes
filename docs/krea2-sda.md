# Native Krea2 Turbo SDA — single-run sampling

Enable **SDA diversity** in V4's Generate panel or on the unified DonutSampler.
Keep **SDA strength = 1**, **Turbo mode on**, **8 steps**, **CFG 1** and
**denoise = 1**. V4's `bleh_preset_0` / `beta` combination is supported. The existing
widget names and order are unchanged; no V4 JSON migration is needed.

## What changed

The old implementation forced `multi_model` and invoked `common_ksampler` twice.
It carried the latent across the split, but not a stateful solver's local history
or stochastic noise-generator state. The replacement keeps the selected
**simple/advanced mode** and invokes the sampler exactly once.

The adapter is active at the first two entries of that run's actual sigma
schedule and off from the third entry onward. The boundary is **not** a fixed
25% diffusion-time threshold or a count of model calls. Both execution paths
use the same cutoff, derived from `sample_sigmas[2]`.

- **Comfy patches:** a native ComfyUI weight hook, attached to copies of the
  positive/negative conditioning. Upstream hooks and metadata are preserved.
- **Experimental bypass:** SDA-only forward adapters are scoped to each early
  model evaluation and removed in `finally`, including failed/interrupted
  evaluations. Existing LoRA forward adapters remain intact. Unsupported targets
  raise an error instead of silently becoming an always-on regular patch.

SDA loads no additional checkpoint beyond the models already used by the merge.
No initial noise is added at the boundary, and
no seed or solver history is reset. Bypass still needs adapter/activation memory;
this is not a zero-VRAM-cost feature. Native hooks can have repatching/offload
costs; neither speed nor quantized GPU parity has been benchmarked here.

## Supported configurations

SDA accepts **Euler, ER-SDE and DPM++ 2M**, directly or through a verified Bleh
preset. These stock solvers make one
prediction at each schedule entry. ER-SDE's history and random-generator state
therefore survive the cutoff. Other solvers, including adaptive solvers and
multi-evaluation methods such as Heun, are explicitly rejected: their internal
sub-evaluations must not be mistaken for two completed denoising steps.

The selected scheduler is retained, provided it produces a strictly descending
8-step schedule ending at zero. A different scheduler is not claimed to reproduce
the upstream quality measurements. `euler` / `simple` is an optional reference
comparison, not a required replacement for V4's configured sampler/scheduler.

### V4 default: Bleh preset 0, ER-SDE in ODE mode, beta

In `workflows/v4-beta/DonutWF_v4_beta.json`, outer node **1014** supplies
`bleh_preset_0` and `beta` to the linked inputs of sampler **993**. The nested
sampler's stored `er_sde` / `bong_tangent` values are not the effective selections.
**SamplerER_SDE 1045** registers **ODE**, `max_stage=3`, `eta=0`, `s_noise=1`
through **BlehSetSamplerPreset 1042**, slot **0**, with no sigma override.
Comfy's ODE builder resolves `s_noise` to **0** and supplies an identity
`noise_scaler`. Selecting stock `er_sde` would lose those options.

The SDA name check now defers Bleh preset verification to sampler entry. The
runtime guard resolves the live registry through the installed Bleh node,
verifies the underlying kernel identity and options, and then calls the
**original preset unchanged**. It does not replace the sampler, rewrite beta,
clear solver history, reseed, or mutate the Bleh registry. The console includes:

```text
[Donut SDA] Solver: bleh_preset_0 -> er_sde; s_noise=0.0; max_stage=3; noise_scaler=ode_noise_scaler
```

The noise-scaler name can differ by Comfy version. The registry is checked on
every run: slot 0 is not assumed always to contain ER-SDE. Missing registrations,
unsupported wrapped solvers, cyclic chains, Euler churn, and
`override_sigmas_opt` produce specific errors. Sigma overrides remain excluded
because Bleh substitutes them after Comfy publishes the gate's schedule.
The normal beta scheduler needs no sigma override.

The wrapper contract was checked against
[Bleh's workflow-pinned implementation](https://github.com/blepping/ComfyUI-bleh/blob/b889683c425f0870a6192606438fecb7a5bda8b9/py/nodes/samplers.py),
and the ODE options against
[Comfy's SamplerER_SDE builder](https://github.com/Comfy-Org/ComfyUI/blob/7a0b5eede3f9721c8faab290689893f36edc6d66/comfy_extras/nodes_custom_sampler.py).

Single Krea2 models, ordinary weight merges, and Donut's hard module-swap merges
in **Experimental bypass** are supported. The following remain excluded:
`torch.compile`, editing/inpainting, masked or partial-denoise generation, and
explicit `multi_model` mode are not supported by this fix. NAG is left on the
normal sampler path; its two text streams see the same scheduled model weights.
NAG/image-quality interaction still needs a real GPU A/B test.

### Hard module-swap merges

Keep V4's merge enabled, its existing ratios, **Experimental bypass**, and
**`bleh_preset_0` / `beta`**. The blanket module-swap rejection is removed.
SDA reads the same source/plan metadata as the real Donut merge injection:

- Exact model2 swaps route the corresponding SDA adapters to the **retained
  source layers**. Putting weight hooks on the unused model1 copies would have
  no effect; putting an `APPLY_MODEL` wrapper on model2 alone would also have no
  effect, because the merge calls its selected layers, not its full model.
- Kept model1 layers and materialized partial blends receive SDA on **model1**.
- Both sets are scoped around the same early main-model evaluation, then
  removed. Ordinary LoRAs, merge hooks, weights/biases, quantization metadata,
  source references, and merge ratios are not rewritten.
- At runtime the source is re-resolved from the active sampling patcher, not
  captured from the loader. This follows the merge injection across clones.
- Cleanup runs across both roots, including partial injection failures and
  denoiser exceptions. It does not leave SDA on for subsequent finishing passes.

The same two-of-eight gate and single solver run are retained. The console logs:

```text
[Donut SDA] Hard-swap routing: <N> primary / <M> retained model2 adapter(s)
```

Counts depend on the adapter and chosen swap plan. A zero source count is valid
when the SDA file has no targets among the swapped layers. Existing ON/OFF
transition logs still identify the actual sigma cutoff.

Donut's global execution policy already makes a hard-swap output inherit
Experimental bypass. A hand-built mixed-policy API call requesting native
weight hooks on swapped SDA targets is rejected rather than silently patching
unused model1 weights. Regular Comfy-patch merges and single-model native hooks
keep their original behavior. Missing/inconsistent source plans, incompatible
live targets, and compiled/nested sources fail explicitly; no always-on fallback
or full-model materialization is introduced.

For a GPU A/B test keep the merge, sampler, scheduler, seed and prompt fixed,
with Turbo on, eight steps and full denoise. Compare SDA off/on without changing
the merge controls. No V4 JSON migration or different adapter download is needed.

With SDA **off**, or strength **zero**, the existing sampler is called without
loading the file, adding hooks, changing modes or importing scheduling support.
Dormant advanced step controls are ignored in simple mode, just as before.

## Install the correct adapter

Install:

`ComfyUI/models/loras/krea2/krea2_turbo_sda_v1.0_comfy.safetensors`

[Download the pinned ComfyUI file](https://huggingface.co/F16/krea2-turbo-sda/resolve/cd6ffc8fbbbc4b07be023524c133905eed9c0fae/krea2_turbo_sda_v1.0_comfy.safetensors)

- Size: **469,315,664 bytes**.
- SHA-256: `0fafed045c53c4acd6165eb55da6ec04b24785b1eeed6f1be37b2cdcb66dba2b`.

The sampler checks the size and checksum before loading, then checks that the
adapter's weight tensors mapped to model targets. Verified CPU data is cached on
the sampler instance; a changed path/file fingerprint invalidates that cache.
Do not rename the Diffusers-format file to the ComfyUI filename. Do not also
place SDA in your ordinary always-on LoRA stack: that would apply it twice and
leave an ungated copy active.

Enabling SDA does **not** download anything. The full GitHub build's **Download
missing** button can fetch the catalog entry after an explicit click. The
Registry/Manager build requires manual installation.

## References and test workflow

The [F16 model card](https://huggingface.co/F16/krea2-turbo-sda) specifies disabling
SDA after denoising step index 1. Its Diffusers example does that inside one
pipeline call. The [supplied ComfyUI workflow](https://huggingface.co/F16/krea2-turbo-sda/blob/a7d6122/Krea2_turbo_sda_workflow.json)
uses RES4LYF `linear/euler`, `simple`, eight steps, and a two-step first stage
followed by `resample`. It also selects an extra LoRA on the second model branch;
that extra adapter is not a requirement of this native implementation.

**Correction to the earlier diagnosis:** that reference uses Euler. Its use of
`resample` alone does not establish that losing multistep history caused a given
bad Euler image. The old split *does* restart stateful solvers, but the user's
exact GPU failure has not been reproduced here.

`workflows/examples/krea2_sda_reference_api.json` is an **API-format** A/B prompt:
SDA off, native scheduled SDA, and a stock Euler 2+6 reference. Submit it as the
`prompt` field to ComfyUI's `/prompt` endpoint, or import it using a frontend
that supports API-format workflows. It is not a serialized V4 canvas workflow.
All branches share the same model, prompt, seed and initial empty latent. Replace
the example loader filenames with your installed Krea2 Turbo/encoder/VAE choices;
do not download a new large checkpoint just for this test. The stock split is
an Euler-only reference, not a recommended implementation for ER-SDE.

`workflows/examples/krea2_sda_v4_bleh_api.json` adds an **API-format** SDA off/on
comparison using V4's exact ODE preset settings and beta. Both sampler model
inputs depend on the preset setter, ensuring registration precedes sampling.
This isolates the sampler; it does not include the model-swap merge or editing,
and is not a replacement V4 canvas. Use the existing V4 merge for that comparison.

For the first V4 comparison disable Seed Variance, extra fusion experiments,
NAG, caches, finishing stages and other optional adapters. Compare identical
seeds with SDA off/on, then restore features one at a time. Test Experimental
bypass using the upstream execution-mode control; native SDA inherits it.

The info output includes the verified adapter and `single run, ON 1-2 / OFF 3-8`.
The console reports the actual cutoff sigma at sampler entry and logs the
weight-hook/runtime-adapter ON and OFF transitions at model evaluation.

## Validation

Run `python -m unittest -v test_krea2_sda_native test_sda_sampler_presets test_sda_merge`
from the DonutNodes checkout.
The tests use real CPU tensors/files and ComfyUI interface doubles. They cover
single-call dispatch, the gate, upstream hook preservation, scoped bypass
arithmetic/cleanup, checksum failures, cache reuse and unsupported settings.
Bleh tests cover live registry resolution, preserved ODE options/beta selection,
single-call dispatch through both SDA paths and unsupported wrappers/overrides.
Merge tests execute the real Donut merge builder, dynamic swap injection and
source resolver with small CPU Linear layers. They check forward deltas on the
correct roots, first-two ON/remaining-six OFF, sampling-clone rebinding, ordinary
LoRA preservation, failures/cleanup, unchanged weights, and V4 node dispatch.
The patcher and adapter-manager lifecycle interfaces are test doubles.
They do **not** validate the full installed ComfyUI lifecycle or real Krea2,
quantized GPU output, speed, or the visual effect of NAG. Keep this change in
review until the reference A/B has been run on the target setup.

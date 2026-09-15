# Native Krea2 Turbo SDA — single-run sampling

Enable **SDA diversity** in V4's Generate panel or on the unified DonutSampler.
Keep **SDA strength = 1**, **Turbo mode on**, **8 steps**, **CFG 1** and
**denoise = 1**. Use `euler` / `simple` for the reference test. The existing
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

No second checkpoint is loaded, no initial noise is added at the boundary, and
no seed or solver history is reset. Bypass still needs adapter/activation memory;
this is not a zero-VRAM-cost feature. Native hooks can have repatching/offload
costs; neither speed nor quantized GPU parity has been benchmarked here.

## Supported configurations

SDA currently accepts **Euler, ER-SDE and DPM++ 2M**. These stock solvers make one
prediction at each schedule entry. ER-SDE's history and random-generator state
therefore survive the cutoff. Other solvers, including adaptive solvers and
multi-evaluation methods such as Heun, are explicitly rejected: their internal
sub-evaluations must not be mistaken for two completed denoising steps.

The selected scheduler is retained, provided it produces a strictly descending
8-step schedule ending at zero. A different scheduler is not claimed to reproduce
the upstream quality measurements. Start with `euler` / `simple`, then test your
normal `er_sde` scheduler separately.

Use a single Krea2 model or an ordinary weight merge. Hard module-swap merging,
`torch.compile`, editing/inpainting, masked or partial-denoise generation, and
explicit `multi_model` mode are not supported by this fix. NAG is left on the
normal sampler path; its two text streams see the same scheduled model weights.
NAG/image-quality interaction still needs a real GPU A/B test.

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

For the first V4 comparison disable Seed Variance, extra fusion experiments,
NAG, caches, finishing stages and other optional adapters. Compare identical
seeds with SDA off/on, then restore features one at a time. Test Experimental
bypass using the upstream execution-mode control; native SDA inherits it.

The info output includes the verified adapter and `single run, ON 1-2 / OFF 3-8`.
The console reports the actual cutoff sigma at sampler entry and logs the
weight-hook/runtime-adapter ON and OFF transitions at model evaluation.

## Validation

Run `python -m unittest -v test_krea2_sda_native` from the DonutNodes checkout.
The tests use real CPU tensors/files and ComfyUI interface doubles. They cover
single-call dispatch, the gate, upstream hook preservation, scoped bypass
arithmetic/cleanup, checksum failures, cache reuse and unsupported settings.
They do **not** validate the full installed ComfyUI lifecycle or real Krea2,
quantized GPU output, speed, or the visual effect of NAG. Keep this change in
review until the reference A/B has been run on the target setup.

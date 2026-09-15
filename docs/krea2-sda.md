# Native Krea2 Turbo SDA diversity

DonutSampler can apply F16's Krea2 Turbo SDA diversity LoRA natively without a
manual two-model graph or a LoRA scheduling node.

The adapter is trained for one specific inference recipe: the full eight-step
Krea2 Turbo schedule, with the adapter active for exactly the first two denoise
steps. Keeping it active for later detail-refinement steps degrades quality. The
native Donut integration therefore owns the gate rather than exposing SDA as an
ordinary always-on LoRA row.

## V4 workflow

Existing V4 workflows gain two controls beside **Turbo mode** after DonutNodes is
updated and ComfyUI/browser are restarted:

- **SDA diversity**: enable the native two-step diversity phase.
- **SDA strength**: adapter strength; `1.0` is the upstream recommended value.

No workflow JSON migration is required. The frontend discovers the existing
DonutSampler that already supplies the Turbo controls and exposes the newly
registered widgets on the same V4 panel.

When SDA is enabled, **Download missing** includes
`models/loras/krea2/krea2_turbo_sda_v1.0_comfy.safetensors`. The reviewed Donut
catalog pins the upstream file's exact SHA-256 and byte size before downloading.

## Sampling behavior

SDA reuses DonutSampler's existing latent-continuation multi-model engine:

1. The normal incoming model path is cloned with SDA applied.
2. Steps `0..1` run with the SDA model.
3. Steps `2..7` continue the same latent with the clean incoming model.
4. No new noise is added at the phase boundary and the seed is not changed.

The feature requires:

- **Turbo mode** enabled.
- **Steps = 8**.
- **Denoise = 1.0**.
- Sampling from step 0 through the complete schedule.
- No manually connected `model_2` / `model_3` multi-model phase.

SDA intentionally refuses partial-denoise, editing/inpainting, or a shortened
schedule because those runs do not correspond to the adapter's trained two
high-noise steps.

## Experimental bypass

SDA inherits the Donut LoRA execution mode already published on the connected
model path. With **Experimental bypass**, the temporary SDA phase uses the same
runtime forward-adapter implementation as Donut's other bypass LoRAs. It does
not rebuild the quantized model weights just for SDA. After step 2 the sampler
continues with the original clean model path, so the SDA runtime adapter is no
longer present.

With **Comfy patches**, SDA uses Donut's existing block-weight-aware model-only
LoRA loader instead. Both execution modes share the same fixed 2/8 gate.

## SDA vs Seed Variance

Both features aim to reduce repetitive outputs across seeds, but they act at
different points:

- **SDA diversity** changes the model only during the first two high-noise Turbo
  steps where composition is established.
- **Seed Variance** perturbs prompt conditioning through the optional
  `krea-seed-variance-enhancer` pack.

For a clean baseline, use one diversity mechanism at a time. SDA does not remove
Seed Variance from existing workflows because that would change saved behavior
and because users may still want conditioning-level variation. For Krea2 Turbo
text-to-image generation, SDA is intended to be the simpler native option.

## Source

Upstream adapter: `F16/krea2-turbo-sda`, file
`krea2_turbo_sda_v1.0_comfy.safetensors`.

The Donut catalog currently pins the upstream file uploaded at commit
`cd6ffc8fbbbc4b07be023524c133905eed9c0fae`:

- SHA-256: `0fafed045c53c4acd6165eb55da6ec04b24785b1eeed6f1be37b2cdcb66dba2b`
- Size: `469315664` bytes

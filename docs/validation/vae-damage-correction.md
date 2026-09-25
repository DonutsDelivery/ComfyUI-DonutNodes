# VAE damage correction — implementation review

Implemented in `DonutVAEDecode`, the existing registered `DonutTiledUpscale`
stage, and `DonutFaceDetailer`. The optional `vae_damage_correction` and
`vae_damage_strength` inputs are appended after previous inputs/widgets.
Defaults are Off and 1.0; each independent strength slider spans 0–4.

Source trace: the V5 generation panel's existing control paths identify its
Donut sampler. The migration follows that sampler's latent output to the directly
connected stock decoder within the same graph. On tagged V5 workflows, only that
decoder becomes `DonutVAEDecode`; its ID, sockets and connections are retained.
The migration initializes the two new controls Off/1 only on the stock node, so
subsequent loads retain saved choices. It skips nonstandard decoder contracts
and custom chains, and runs only when the backend advertises the new node.
No global ComfyUI decoder patch is installed. The distributed JSON is unchanged.

The generation, first/second upscale and face panels bind to the actual nested
stage widgets using the existing panel commit/serialization path. Path/widget
deduplication prevents repeat organization from adding the same controls. No
setting is copied into a second store or promoted to an outer input.

All stages share the same float32 `y + strength * (y - decode(encode(y)))`
correction, clamped to RGB range. Each still image receives its own VAE round
trip; batches are not treated as video frames. Non-grid dimensions are padded
on the bottom and right, then cropped back without resizing or pixel shifts.

- `DonutVAEDecode.decode` calls the stock decoder first, then corrects the RGB
  result before the existing downstream inpaint composite, previews and hires.
- `DonutTiledUpscaleStage.run_stage` runs its existing upscale and colour
  preservation first, then corrects its finished image, retaining the debug
  image. The explicit arguments do not reach NAG/colour/SeedVR2 option maps.
- `DonutFaceDetailer.doit` passes the explicit controls through `enhance_face`
  into `enhance_detail_megapixel`. Correction runs after the last decode and
  any post-decode hook, before edit padding is cropped and before the face crop
  is resized/composited. It runs once per refined crop, outside the sampling
  cycle loop. Cropped preview outputs use the corrected crop too. No faces,
  skipped segments, skip-sampling hooks and bypassed detailers add no correction
  pass. These arguments are consumed before NAG receives its options.

Off/zero-strength correction adds no VAE work. Disabled upscale stages skip it;
SeedVR2 replacement also skips it and hides the corresponding panel group.

Validation scope: source inspection only. No unit tests, browser interaction,
queued generation, PNG metadata comparison, save/reload exercise or GPU quality
comparison was run. The running ComfyUI process was not restarted. Runtime and
panel-to-generation behavior remain unverified. No registry publication or
distributed workflow JSON update was made.

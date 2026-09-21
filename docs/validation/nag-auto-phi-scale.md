# NAG auto-phi scaling validation

This change adds an optional alpha-normalized phi mode.

Formula:

```
effective_phi = nag_phi_scale / nag_alpha
```

when `nag_auto_phi=true`. The reference scale is chosen so the unclipped
linear coefficient `alpha * phi` equals the upstream default
`0.25 * 4.0 = 1.0` when `nag_phi_scale=1.0`.

Manual mode remains unchanged. If alpha or scale is zero, effective phi resolves
to zero without division.

## V5 workflow

The distributed V5 workflow enables auto phi with guidance scale 1.0 on all four
NAG-capable stages stored in the generation subgraph: base sampler, both tiled
upscale paths, and face detailer.

The NAG panel groups expose:

- Manual phi (auto off)
- Auto phi from alpha
- NAG guidance scale

The new backend widgets are appended to the existing optional NAG inputs to
preserve serialized positions of prior widgets.

## Focused regression coverage

Added tests verify:

- manual phi remains unchanged
- alpha 0.45 resolves phi to 2.222...
- alpha 0.25 still resolves phi to 4.0
- guidance scale multiplies the effective linear coefficient
- zero alpha/scale does not divide by zero
- resolved phi reaches the upstream NAG patch
- all four V5 NAG-capable stages save auto mode On with scale 1.0
- all four V5 panel groups bind the new controls to the same NAG node path

## End-to-end status

Per AGENTS.md, this workflow/panel change still requires live ComfyUI
verification before merge/release:

1. load the changed V5 workflow after backend restart/browser refresh
2. set a distinctive alpha and guidance scale through the user-facing panel
3. queue with the Run button and execute the affected base/upscale stages
4. save PNG metadata and confirm panel values match embedded workflow and
   execution prompt
5. reload and confirm values survive

That live GPU/UI/PNG round-trip has not been executed from this environment.

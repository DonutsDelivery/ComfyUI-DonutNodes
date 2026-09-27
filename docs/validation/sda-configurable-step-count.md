# SDA configurable Turbo step count

## Change

SDA now accepts the sampler's configured positive Turbo step count and validates
the complete sigma schedule received at runtime. Its gate scales the reference
2-of-8 active fraction to the nearest whole-step count, rounding half-step ties
up. At 12 steps, SDA is active for steps 1-3 and inactive for steps 4-12. The
8-step reference remains active for steps 1-2 and inactive for steps 3-8.

The single solver invocation, full-denoise requirement, supported sampler list,
and scheduler-specific Turbo/upscale denoise resolver are unchanged.

## Verification

- Configured schedule/gate cases: 6 -> 2 active; 8 -> 2; 10 -> 3; 12 -> 3.
- SDA validation and node forwarding: 12 steps accepted in simple and advanced
  modes, both execution modes; advanced mode rejects a range ending at step 11.
- Runtime sampling guard and native hook: 12-step sigma schedule accepted; SDA
  switches off at the fourth evaluation while retaining one solver call.
- SDA native suite: 22 passed; Bleh sampler suite: 17 passed; merge suite: 21
  passed; Turbo schedule suite: 13 passed.
- Upscale lifecycle suite: 7 passed, including Turbo at 8 partial-denoise steps
  and 12 full-denoise steps. The isolated test needed an in-memory
  `donut_vae_upscale` stub because its fixture's fake `nodes` module does not
  define `VAELoader`; no repository code was changed for that harness issue.
- Python compilation: passed for the modified backend and test files.

## Verification limits

No ComfyUI application/session was available, so this was not run through the
user-facing Run button and no output PNG or queued-prompt metadata was produced.
This backend change does not alter panel bindings, widget serialization or
workflow JSON. Restart ComfyUI before using the updated Python node. GPU image
quality for the 12-step SDA gate has not been evaluated; its gate preserves the
adapter's reference 2/8 fraction, while the published SDA reference itself uses
8 steps.

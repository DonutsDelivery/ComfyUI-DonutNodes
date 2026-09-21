# NAG tau-clipping bypass experiment

This draft adds one opt-in boolean: `nag_disable_tau_clipping` (default **Off**).

Purpose: test whether the reported Rebalance + UncensorFix + NAG blotch/grain
correlates with NAG's tau limiter itself. It does **not** change Rebalance,
UncensorFix, phi, alpha, sigma windows, sampler/scheduler, seeds, or conditioning.

## Math

Upstream Krea2 NAG computes:

```
guided = z_pos + phi * (z_pos - z_neg)
ratio = L1(guided) / L1(z_pos)
normalized = guided * min(ratio, tau) / ratio
refined = alpha * normalized + (1 - alpha) * z_pos
```

With **Disable tau clipping** enabled, only the middle normalization step is
removed:

```
guided = z_pos + phi * (z_pos - z_neg)
refined = alpha * guided + (1 - alpha) * z_pos
```

The implementation is model-local. It clones the installed upstream Python
block function with a replacement guidance global instead of mutating the
upstream NAG package process-wide. The combined Krea2Edit NAG wrapper is handled
the same way by cloning its local forward globals.

If the unclipped calculation produces NaN/Inf, the experiment raises rather
than silently allowing a corrupted image to continue.

## UI / serialization

The backend input is appended at the **end** of the existing NAG optional inputs
so saved workflow widget positions do not shift.

A presentation-only frontend migration displays the same real backend widget
immediately after **Nag tau** in every Donut panel group that already exposes
`nag_tau`:

**Disable tau clipping · experiment**

No distributed V5 workflow JSON is changed.

## Suggested A/B

Hold all other values fixed, especially the failing case:

- Rebalance
- UncensorFix
- NAG alpha 0.45
- phi 4.0
- tau 2.5
- same seed/prompt/stage settings

Compare toggle **Off** vs **On** separately for base pass and upscale. Record
whether the stage is clean/blotched and whether the no-tau path raises a
nonfinite error.

This is diagnostic only. Removing the limiter can expose larger attention
vectors than upstream NAG permits, so it should stay default-Off until actual
image behavior is understood.

## Validation status

Focused Python and panel regression tests are included for the unclipped
equation, overflow reporting, model-local function cloning, installer routing,
append-only schema placement, and panel idempotency.

Full ComfyUI/GPU execution, Run-button validation, PNG metadata round-trip and
image-quality A/B remain required before merging/releasing.

# Krea2 fusion rendering/coherence experiments

This branch adds four opt-in diagnostic presets to compare against **Rebalance**
and **Balanced** with identical prompts, seeds and sampling settings. Existing
presets are unchanged.

## Presets

| Preset | Tap profile | Strength/formula | Normalization | Purpose |
| --- | --- | --- | --- | --- |
| `Experiment · NAG-friendly mean` | classic | 1.0 / scale-around-1 | `mean_gain` | Keep the classic semantic emphasis but use one prompt-independent global gain. Less suppression of ordinary taps than static RMS. |
| `Experiment · NAG-friendly static RMS` | classic | 1.0 / scale-around-1 | `rms_gain` | Normalize from the gain vector rather than each prompt tensor. Positive and NAG-negative prompts therefore receive the same transform. |
| `Experiment · NAG-friendly power 0.60` | classic | 0.60 / geometric power | none | Keep all 1.0 taps at exactly 1.0 while compressing the boosted late taps. Designed to preserve more baseline color/light information. |
| `Experiment · soft tensor RMS 0.75` | classic | 0.75 / scale-around-1 | `tensor_rms` | Keep the current prompt-dependent RMS matching, but reduce the classic-profile contrast before normalization. This is the adherence-first comparison and is **not** expected to fix the NAG scale mismatch by itself. |

`NAG-friendly` here has a narrow meaning: the tap transform itself is fixed and
prompt-independent, so positive and NAG-negative conditioning use the same gain
vector. It does not claim that every NAG phi/tau/alpha setting will be artifact-free.

## Suggested A/B sequence

Use one prompt, one seed and the same Krea2 Turbo settings. Disable Seed Variance
and other changing inputs while comparing.

1. **Rebalance** — rendering/color/lighting reference.
2. **Balanced** — current prompt-adherence reference.
3. **Experiment · NAG-friendly mean** — first middle-ground candidate.
4. **Experiment · NAG-friendly static RMS** — stronger static normalization.
5. **Experiment · NAG-friendly power 0.60** — preserves baseline 1.0 taps.
6. **Experiment · soft tensor RMS 0.75** — tests whether less tap emphasis fixes
   rendering while retaining tensor-RMS coherence.

Run that sequence once with NAG off, then repeat with the same NAG settings.
Useful observations are prompt-object count/placement, colors, black level,
highlight rolloff, local contrast, skin/background separation, and whether NAG
introduces hue/brightness shifts or loses prompt elements.

## Why these are presets rather than new permanent controls

The goal is to identify which mechanism is responsible before expanding the UI.
If one family consistently wins, a later change can replace the fixed experiment
with a proper continuous control (for example a normalization amount) without
making the stable panel more complicated prematurely.

# Synthetic gamma restoration: a separate Tone Lab profile

Status: proposed training experiment, not implemented by the inference PR or
the community HTML trainer. The historical filename is retained for links;
this is **not pretraining for, fine-tuning of, or a replacement for Millo's
personal taste profile**. No photographs are downloaded by this design note.

## Two different objectives

Millo's current checkpoint learns personal approvals on their own image
collection, monitor and viewing setup. Its goal is the look they prefer, not
reproduction of a professional photographic reference. Preserve that checkpoint
unchanged; do not mix its ratings or normalization with this experiment.

The proposed photographic profile instead learns to restore the appearance of
chosen reference photographs after known synthetic gamma offsets. Train it
independently from scratch, with separate examples, labels, normalization and
weights. Export it under a distinct model filename and let the user select it
explicitly in the same inference node. The objectives are different; neither
profile is inherently an upgrade over the other, and transfer is not assumed.
The broader-use intent is a hypothesis to evaluate, not a claim of universal tone.

The public [HTML trainer](../tools/tone_lab/README.md) supports personal profile
training. It currently has one active local session: loading a different folder
or resetting only the model does not isolate a new profile. Export a session
backup and explicitly start a blank session, or use separate browser profiles.
The synthetic-data pipeline described below remains future work.

## Known-input / known-correction supervision

Use a rights-cleared, aesthetically approved photograph R as the reference.
Generate perturbed inputs X, analyze X with the same versioned preprocessing as
the deployed predictor, and fit ONE global set of weights to predict the inverse
correction. References/IDs/ratings are never inputs to runtime prediction.

For pure display-RGB gamma, without clipping/quantization:

```
X = R ** a
corrective_gamma = 1 / a
reconstructed = X ** corrective_gamma
```

In the existing gamma slider convention `a = 3 ** (-slider / 100)`, undoing
this perturbation corresponds to the negative of its slider offset. Exponents
are reciprocals, not negatives. This is recovery of a deliberately introduced
change; it is not measurement of an image's unknowable historical/display gamma.

With multiplicative brightness as well:

```
X = b * R ** a
corrective_gamma = 1 / a
corrective_gain = b ** (-1 / a)
reconstructed = X ** corrective_gamma * corrective_gain
```

The inverse post-gamma gain is generally NOT `1/b`. Derive both labels jointly.
The existing export expects correction gamma/gain; a network predicting the
corruption instead must be converted or given a different decoder/schema. Keep
inverse labels within the existing output domains (gamma 1/3..3, gain 0.8..1.25).
A compatible v4 model-only JSON can be selected without changing the node, while
keeping the personal profile file and its weights intact.

## Proposed experiment

1. Start an independent model/dataset. Use gamma-only perturbations, distributed
   in log-exponent/slider space in both directions, plus unchanged reference
   examples with gamma=1 and gain=1. Learn to undo the offset AND leave the
   chosen originals alone; do not initialize from the personal checkpoint.
2. Split by ORIGINAL photo/scene before generating variants. All retouches,
   crops, duplicates and perturbations of a source belong to the same split.
   Normalize from training sources only. Many variants are not many independent
   photos; balance sampling by source rather than over-weighting prolific ones.
3. Apply perturbations before proxy generation. Power curves and downsampling
   do not commute. Use the deployed analysis path, retain float references when
   possible and record the precise resize/encoding implementation.
4. Add modest independent brightness perturbations after the gamma-only baseline
   works. Include intentionally dark and high-key references, varied lighting,
   contrast, colour, subjects and compositions. Do not teach a single target mean
   or histogram for every scene.
5. Measure inverse log-gamma/gain errors and reconstruction error on entirely
   held-out ORIGINALS. Separately test unchanged-image drift, agreement of outputs
   across variants of a source and repeat-pass stability. Reserve a final test
   set that is not used for architecture/checkpoint selection.
6. Evaluate broader-use behavior separately from reference reconstruction. Keep
   any new human evaluation associated with this photographic profile, not
   imported as supervision from Millo's personal taste dataset. Publish a
   separately named checkpoint only when its own results justify doing so.

Clipping, quantization and lossy compression discard information. They can be
robustness tests, but do not claim an exact inverse of discarded pixels. Restrict
perturbations or use suitable reconstruction masks/losses and report clipping.
Contrast curves, white-balance shifts and exposure operators are not interchangeable
with gamma; introducing them requires nuisance handling or a richer correction
model, not incorrect gamma labels.

## References and rights

MIT-Adobe FiveK is a relevant RESEARCH candidate: 5,000 photographs, each retouched
by five trained photographers. Use approved finished renditions, not raw inputs
as arbitrary no-change targets. Professional editing is still a style preference,
not proof that one rendition is universally correct.

The official image licenses restrict their granted uses to research not directed
toward commercial advantage or monetary compensation. A mirror's code license
is not evidence that the photographs are cleared for production training or
checkpoint distribution. Use owned/commissioned/appropriately licensed references
or establish the necessary permissions before a distributed-model training run.
No conclusion about deployment rights is implied by public download availability.

Official sources checked 2026-09-22:
- Project and dataset description: https://people.csail.mit.edu/vladb/photoadjust/
- Adobe image license: https://data.csail.mit.edu/graphics/fivek/legal/LicenseAdobe.txt
- Adobe/MIT image license: https://data.csail.mit.edu/graphics/fivek/legal/LicenseAdobeMIT.txt

## Expected benefit and limitation

This supplies known synthetic inverse targets for a separately trained profile.
It does not remove the underlying ambiguity: an intentionally dark photograph
and a gamma-darkened photograph can have similar statistics. The model learns a
reference-photo prior, so success on synthetic corruptions must not be presented
as proof of Millo's preferred look on AI images or as monitor/ICC calibration.
Keep each profile's evaluation, training history and checkpoint lineage separate.

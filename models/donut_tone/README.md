# Included Tone Lab checkpoint

`donut-tone-v4-r12.json` is the user-supplied `donut-tone-model-v4_2.json`,
minified without changing any value. No retraining, rounding or synthetic
augmentation was performed for this commit. This is one shared global model,
not a table of image-specific corrections.

- Model format: `donut_feature_mlp_v4.0`
- Analysis schema: `donut_srgb256_features_v4.0`
- Architecture: 169 → 16 → 8 → 3; 2,883 weights and biases
- Stored model revision: **12** (not the HTML review-round number)
- Stored fitting metadata: 395 training images/groups, 84 validation images,
  400 epochs, regularization 0.01, tone step 0.025, seed 50301

Later screenshot review-pool sizes are not the fitted checkpoint's training
metadata. Human approval rates in particular review batches are not a measured
universal accuracy or independent benchmark for this file.

## Integrity

Source upload SHA-256 (109,153 bytes):

```
638048dde1ed4ca9c26355f18c75b5f4dc210d4d1bac3062b773db948f42a78b
```

Bundled minified JSON SHA-256 (73,485 bytes):

```
5542880ab9b627d828e2849a86818f2ccc100afd72da19cf98636abef38f4f28
```

Bundled Git blob: `d83c23f409319fe9494e5dd8145d4141b93b21c0`.
The byte hashes differ solely because whitespace outside strings was removed;
all number literals, strings, array order and object values are unchanged.

## Second bundled checkpoint: donut-tone-v4-general-synth.json

A second, independently trained model ships alongside r12 as an alternative.
Source upload SHA-256 (105,724 bytes):

```
ff1c1f5c3ca934795352d51d0848179003956bbf4761bdca7404df07131699d5
```

Bundled minified JSON SHA-256 (69,874 bytes; whitespace-only difference, every
value unchanged):

```
64a5288e9f4050291fa3ad136a34123522c1f31ec771b9ae74b441932571d0ef
```

- Architecture: identical 169 → 16 → 8 → 3; 2,883 weights and biases
- Stored model revision: **1**
- Stored fitting metadata: 20,057 training groups and 2,421 validation
  groups — real high-quality professional photographs (Unsplash) with a
  randomly drawn gamma offset applied; 50 epochs (best 28), regularization
  0.0001, tone step 0.0025, seed 50301, generator
  `train_donut_tone_synth.py`, identity probability 0.18
- Best validation: slider MAE **12.47957** (±55 slider scale), gain MAE
  0.106 pp, no-op accuracy 0.813

Training characteristics and honest limits: the photographs are real; the
tone errors are simulated — random gamma offsets (slider standard deviation
18, maximum 55) with 18% identity samples. On that distribution an
always-predict-identity baseline scores roughly 14.4 MAE, so 12.5 is a real
but modest edge, and the model was never fitted to match human taste. It sees
far more content variety than r12's 395-photo set, but is calibrated less
tightly and its identity gate is less conservative. `gainPercentMax: 0.0`
means the gain head was not trained; the model corrects gamma only. Its
predictions on flat/bright frames differ substantially from r12's (e.g.
gamma 0.89–1.21 vs r12's 1.33–1.45 on flat grays), so A/B the two on real
images before standardizing on either.

No training photos, image features or rating records are committed for either
checkpoint. Selection stays explicit per workflow: pick the filename in the
Save images panel; defaults and saved selections are never rewritten.

## Use and limitations

The node discovers this directory after user-configured model roots. Select the
filename and enable correction explicitly; defaults/saved selections are not
changed. A same-named file in a user root takes priority. Custom checkpoints
belong in `ComfyUI/models/donut_tone/`, not this bundled directory.

The model-only export contains no source images, filenames or rating history.
The node is deterministic, inference-only and still-image oriented. It predicts
a global RGB gamma/gain curve, not recovery of clipped detail, white balance,
semantic understanding or local exposure. Real-checkpoint numerical compatibility
is tested on synthetic proxies; CUDA, live ComfyUI workflow execution and image
quality still need validation in the intended environment.

See [node setup](../../docs/tone-lab.md) and
[training your own model](../../docs/tone-lab-training.md).

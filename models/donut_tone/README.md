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

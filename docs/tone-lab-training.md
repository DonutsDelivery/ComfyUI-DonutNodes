# Train a personal Tone Lab model

The repository provides both entry points:

- **Portable v4 snapshot:** download [donut_tone_lab_feature_learner_v4.html](../tools/donut_tone_lab_feature_learner_v4.html) as a raw file and open it locally. It is the original standalone v4 trainer, with one final newline added; no algorithm changes or personal session data are embedded.
- **Maintainable source:** open [tools/tone_lab/index.html](../tools/tone_lab/index.html) with its sibling CSS/JavaScript files present. The [trainer guide](../tools/tone_lab/README.md) explains the source layout and optional standalone builder.

Both use the v4 export contract supported by the node. The trainer starts with
unchanged previews unless a local session is restored or you import a model.
You can explicitly import the included
[donut-tone-v4-r12.json](../models/donut_tone/donut-tone-v4-r12.json) as starting
weights for a personal profile; it is not automatically embedded or loaded.

## Workflow

Choose your images, inspect them in the lightbox, rate the actual model output,
and use Fine-tune target or Original is already right for exact teaching targets.
Train + next round learns one shared feature model. Save target returns to the
model preview rather than installing an image-specific override.

In current v4, the first Good creates a stable target. To revise that approval,
use Fine-tune and Save target or Remove target. A later directional rating alone
does not replace a saved target. A rejected training candidate is shown in the
report's after-columns but is not installed or exported as the active model.

Use **Export session** for a local backup of features, filenames, judgments and
weights. Keep this private unless you intend to share those records. Use
**Export model only** for the shared weights/normalization without image records.
Place custom model-only JSON in `ComfyUI/models/donut_tone/`, refresh the model
list, select it explicitly and enable the node. No code change or public PR is
required. See [node setup](tone-lab.md).

## Keep profiles and evaluation separate

Changing folders retains both weights and accumulated training examples. Reset
model alone also retains examples. To start a different objective, export your
session/model first, then clear all saved examples and the model, or use a
separate browser profile. Follow the [full guide](../tools/tone_lab/README.md)
for group splits, storage limitations and resuming independent sessions.

The proposed [photographic gamma-restoration experiment](tone-lab-synthetic-pretraining.md)
is separate from the included personal-taste checkpoint. It has not been
implemented or used to retrain these weights. Recovery of synthetic offsets
and a person's preferred tone are different objectives; compare separate
checkpoints rather than silently replacing this one.

Before publishing a future trainer update, keep the portable snapshot and
maintainable sources synchronized and rerun the HTML/Python conformance tests.

# Tone Lab v4: train your own tone profile

This is the local HTML trainer used to develop the Tone Lab feature model. It
learns **one shared mapping** from 169 image measurements to gamma, brightness
gain and an uncalibrated no-change score. Normal previews use the same global
model for every image; saved targets never silently replace its predictions.

The trainer is independent of ComfyUI and starts without pretrained weights in
an empty browser session. An existing local session is restored automatically.
Its analysis, prediction, training and rating behavior are unchanged from the
original standalone v4 trainer; the source has been separated into local files
for maintenance.

## Open it

Download/clone the repository and open **`tools/tone_lab/index.html`** in a modern
browser. Keep `style.css`, `engine.js`, `state.js`, `app.js`, and `lightbox.js`
beside it. Downloading `index.html` alone is not sufficient. No npm install,
ComfyUI server, remote service or external JavaScript library is required.

For a portable **single-file HTML**, run from the repository root:

```sh
python tools/tone_lab/build_standalone.py donut-tone-trainer.html
```

Open the resulting file locally. The builder uses Python's standard library,
inlines the same assets, and refuses to overwrite an existing output. Building
is optional when opening the complete source directory. Browser restrictions
on local files, workers or storage can vary; no direct-file compatibility across
all browsers is claimed.

Use **Choose folder**, **Choose images**, or drop image files on the page. The
page cannot silently enumerate the directory beside itself. Images and training
run in the browser; this tool has no image-upload endpoint or telemetry. It does
not modify or overwrite source photographs.

## Teach and export

1. Load representative images. Click either preview or **Inspect** to open the
   large lightbox; split/side-by-side comparison, zoom, pan, fullscreen and
   hold-Space-for-original are available.
2. Rate the **model output** with Good, Too bright or Too dark. Use **Fine-tune
   target** to set both gamma and gain explicitly, or **Original is already
   right** to teach identity. The manual preview is clearly labeled. Saving a
   target returns to the model output; it does not install a per-image override.
3. Use **Train + next round**, or enable **Train at batch end**. All saved labeled
   training examples participate, including examples from earlier folders.
   Validation scene groups are excluded from fitting and normalization.
4. Review the updated predictions. **Undo model update** restores the preceding
   model without discarding your teaching examples. Export checkpoints rather
   than assuming every successful training run looks better.
5. Choose **Export model only**, give the JSON a distinctive profile/checkpoint
   filename, and place it in `ComfyUI/models/donut_tone/`. Restart ComfyUI after
   installing node code, refresh the model list after adding a JSON, select it
   in **Save images → Tone Lab · learned auto tone**, and explicitly enable the
   stage. See [node setup](../../docs/tone-lab.md).

Model-only exports carry the shared weights, feature order and normalization,
not image records or rating history. **Export session** is the portable backup
for continuing training. Session exports contain filenames, measurements,
targets and history, although not image pixels; do not publish them as if they
were model-only files.

## Separate profiles, not one universal ideal

A profile represents its training objective. Two distinct uses are:

| Profile | What defines the target? |
| --- | --- |
| Personal taste | A person's approvals and edits, judged in their own viewing setup. The existing Millo checkpoint belongs to this category. |
| Photographic reference restoration (future experiment) | Original photographs chosen as references, paired with synthetically gamma-altered versions, with recovery of the reference as the objective. |

The proposed photography experiment is **not implemented or trained in this
addition**. It must have its own dataset, labels, normalization, weights and
checkpoint. It is not an upgrade, pretraining step or replacement for the
personal profile. There is no claim that the two objectives transfer to each
other or that a curated collection defines universally correct tone. Compatible
profiles can use the same inference node and be chosen explicitly by model file.

Personal judgments can reflect a monitor and viewing environment, but the model
does not measure that monitor. A tone profile is not an ICC profile, display
calibration or measured compensation for another monitor.

### Keep training sessions separate

V4 has **one active local session per browser storage context**, not a multi-profile
workspace. Opening another folder retains the existing model **and** accumulated
teaching examples. **Reset model to untrained** also retains those examples.
Neither action starts an independent profile.

Before starting a different objective, export both the full session and the
model. Then explicitly use **Clear all saved examples & model** to start a blank
session, or use a separate browser profile for parallel independent work. Import
the corresponding full session to resume a profile. Importing model-only weights
into a populated session keeps its examples; that can mix objectives on the next
training run. Different filenames or another tab alone do not guarantee storage
isolation. Keep separate exported sessions as the authoritative backups.

## What the current feedback actually means

- The **first Good** approval stores a target curve. Later Good votes do not
  move it. An original-is-right or manually saved target can explicitly replace it.
- Directional votes guide examples without a target. For an image with a saved
  target, subsequent directional votes are recorded, but training continues to
  fit that target until you explicitly revise or remove it. **Remove target**
  also clears that example's directional constraints. This v4 behavior is not a
  probabilistic model of changing personal preferences.
- When a report says **model kept**, the attempted candidate was not installed.
  The report's after-columns, arrows and target-disagreement counts still describe
  that rejected candidate. **Export model only** exports the active retained model.
- Feedback loss and historical-target errors are diagnostics, not aesthetic
  accuracy. A displayed Good fraction describes those particular judgments.

Keep related variants of the same photograph in one **scene group** before
training. Common corrected/copy suffixes are grouped automatically; check the
result manually. Reassigning a group/role after training prompts to reset weights
to avoid leakage. Validation labels are not used for epoch selection; repeatedly
making development decisions from validation still makes it a development set.
Reserve another untouched set for final evaluation.

## Limits and maintenance

Analysis uses a browser-decoded, 256-pixel-long-edge sRGB proxy. The learned output
is a global RGB power curve and gain, not local contrast recovery, white-balance
correction or scene understanding. Full-size lightbox previews have a labeled
16-megapixel / 8192-pixel limit. Browser decoding, resizing and colour management
can affect agreement with Python inference; test typical images and save formats.

Browser storage is convenient but can fail or be cleared. Keep exported sessions
and model checkpoints, especially before clearing data or changing browser/profile.

The classic scripts load in a fixed order. `engine.js` is also serialized into a
local training worker; it must remain self-contained. The builder produces the
single-file equivalent without changing model math. Run the bundled checks with:

```sh
node --test tools/tone_lab/test_trainer.cjs
```

The tests require Node.js and Python 3 (`PYTHON` can override the Python command).
See [validation record](VALIDATION.md) for checks performed and their limits.

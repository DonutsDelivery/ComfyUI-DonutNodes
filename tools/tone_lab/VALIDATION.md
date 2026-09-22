# Trainer repository packaging validation

Date: 2026-09-22. Scope: publishing the existing Tone Lab v4 trainer as local
HTML/CSS/classic-script source plus an optional standalone builder. No training
formula, inference schema, checkpoint weights or ComfyUI workflow was changed.
No professional photography dataset was acquired or used.

## Repeatable source checks

Command: `node --test tools/tone_lab/test_trainer.cjs`.

**6 passed, 0 failed, 0 skipped**:

1. Every script parses; the entry point loads only the expected local assets.
2. Schema remains 169 features, 169→16→8→3 and 2,883 weights/biases; untrained
   inference is identity.
3. Actual training improves loss on synthetic exact targets without mutating
   the supplied records.
4. Changing validation measurements and targets cannot alter the fitted weights
   or normalization for a fixed training set/seed.
5. Filenames, target records and history do not enter the predictor.
6. The standalone builder inlines assets into valid JavaScript and refuses to
   overwrite an existing output.

## Browser smoke checks

Chromium with Playwright, synthetic 128×96 PNG fixtures only. **14 checks passed**:
empty-session startup, loading images, untrained identity, explicit manual/model
preview separation, saving a target returns to model output, lightbox zoom and
comparison, training eligibility, actual worker training, model-only export
without image records, persistence of in-memory examples/model across visible
folder replacement, mobile lightbox, no runtime errors, no external requests,
and standalone startup without sibling assets.

Desktop 1360×900 and mobile 390×844 screenshots were visually inspected. No user
photograph collection, trained personal weights or private session was used for
these checks.

## Verification limits

Browser policy blocked direct local navigation. The browser checks loaded the
HTML/CSS and injected the classic scripts in their actual order into a test page;
they do **not** establish end-to-end `file://` or local-server launch compatibility.
The generated standalone was likewise tested as page content. Successful durable
browser-storage reload was not verified in this environment; use session exports
as backups.

These tests exercise packaging, inference/training separation and interaction,
not aesthetic quality. They do not constitute live ComfyUI panel → Run → PNG
metadata → reload verification, CUDA testing or browser/codec parity on real
photographs. Existing node integration checks and limits remain in
[`docs/validation/tone-lab-v4-port.md`](../../docs/validation/tone-lab-v4-port.md).

## Publication scope

The repository trainer contains no bundled personal model, image pixels, sessions
or external photography-training pipeline. It starts untrained in a fresh local
session; users may explicitly import their own models/sessions. The personal-taste
profile and a future photographic-reference profile are distinct objectives and
must not silently share an active training session.

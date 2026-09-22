# Tone Lab checkpoint/trainer follow-up — 2026-09-22

This follow-up adds the supplied retained model revision 12 and the original v4
standalone trainer to PR #75. The trainer differs from the supplied HTML only
by one final newline. Checkpoint JSON is minified, with every value unchanged.
Their Git blob identities are checked before committing. No private source
photos, image features, session backups or rating records are committed.

The sole inference-code change adds the package's `models/donut_tone/` as a
last-priority search root plus explanatory messages. It does not change the
analyzer, predictor, input order, workflow migration, panel paths, edit protection,
or enabled/None defaults. Existing user roots and extensions remain registered;
no checkpoints are copied into user directories or downloaded at runtime.

## Executed locally

`python tests/test_tone_lab_bundle.py -v`: **9 tests passed**.

- Source digests, trained model contract, revision and parameter count.
- Shipped HTML integrity (original + final newline).
- Bundled discovery with an empty user models folder; defaults stay unchanged.
- Repeated registration preserves extra roots and extensions, without duplicates.
- Explicit user same-name checkpoint takes precedence; content replacement
  invalidates cache without modifying the bundle.
- Off, Strength 0 and protected editing never read model weights.
- Real-checkpoint CPU RGB/RGBA batch curves, float32/float16, alpha, dtype,
  device and input immutability.
- Traversal rejection remains effective.
- All 169 features and gamma/gain/raw heads/no-op/coverage compared between
  Python and the engine extracted from the shipped HTML on **18 synthetic RGBA
  proxies**, including dark/bright/flat, coloured, narrow and alpha-threshold
  cases. Features agree to 1e-11 tolerance; numeric predictions agree to ten
  decimal places and identity gates agree exactly.

`node --check` on the shipped HTML's inline script: **passed**.

These are isolated CPU tests with a `folder_paths` test double, not live ComfyUI
execution. The original PR's broader tests are recorded separately; they were
not all rerun in this partial checkout for this follow-up. Tests do not establish
aesthetic quality or preprocessing parity for every codec/browser.

## Remaining release checks

No CUDA execution, real-photo quality comparison, or live ComfyUI panel → Run →
saved PNG metadata → reload test was performed. No generation PNG was produced.
Verify the selected bundled and custom model names, enable/disable transitions,
Strength 0/1, edited-image protection, actual output digest and saved/reloaded
widget values in the real installation. Compare model predictions against Tone
Lab using the same source images and browser before removing experimental status.

The distributed workflow JSON is unchanged by this follow-up. No registry
publication, version bump or merge is performed. The PR remains draft pending
end-to-end checks. Synthetic-photo augmentation is documented as a proposed
experiment only; this checkpoint has not been replaced by an augmented model.

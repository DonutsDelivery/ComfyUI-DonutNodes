# Tone Lab v4 port — verification record

Date: 2026-09-22. Base repository commit:
`b537c4ec233c47c3f3fd9783821491765b68decb`.
HTML source SHA-256:
`90a58668b7b2bcd42aae9cb32cb56053d43d4f2710949ca677dbdd7361172dcb`.
Only the inference reference is included in tests; no private images, session
histories or trained aesthetic weights are committed.

## Executed locally

- `python tests/test_tone_lab.py -v`: 18 passed. Includes JS/Python conformance on
  16 synthetic RGBA proxies, all 169 features, numeric model validation, feature-
  only inference, identity gating, RGB/RGBA batches, float16, no mutation, alpha,
  half strength, edit protection, disabled/zero passthrough, missing checkpoint,
  content-hash invalidation, traversal/escaping-symlink and model-size rejection.
- `node --test tests/tone_lab_workflow.test.cjs`: 6 passed, 1 skipped. Fixtures
  exercise main and secondary final branches, preserved internal graphs and
  preview IDs, panel-to-widget paths, edit wiring, idempotence, explicit manual
  settings after JSON save/reload, deleted-stage preservation and refusal to
  rewrite ambiguous/custom/untagged graphs. The additional packaged-V5 test runs
  in a complete checkout; the large bundled JSON was not available locally in
  this partial worktree, so that test was not claimed as passed.
- `python tests/test_tone_lab_browser.py /usr/bin/chromium`: six synthetic RGB PNG
  cases tested against actual Chromium canvas. Proxy pixel error 0 on all six;
  maximum feature error below 1.1e-13; maximum gamma error below 2.3e-16. Sizes:
  37×31, 800×620, 1024×768, 237×403, 1152×896, 2048×1536. These are parity tests,
  not human image-quality tests and not a ComfyUI UI run.

## Not verified / merge checklist

A live ComfyUI installation with the generation models and the user's finalized
Tone Lab model was not available. Do not call this end-to-end verified based on
static wiring or mocked folder_paths tests. No output PNGs from ComfyUI exist
for this run, and no registry publication was performed.

Use a separate audit workflow/session, not the user's working graph:

1. Install this branch and restart backend + frontend. Confirm the changed
   extension is loaded. Import the standard V5 graph and verify a single disabled
   Tone Lab stage and the Save images control group. Test the packaged workflow
   regression in a complete checkout before publishing.
2. Add an actual v4 model-only export to models/donut_tone. Through the actual
   Graph/App Mode panels, set distinctive values (e.g. strength 0.65), turn On,
   select the JSON and use **Run**. Check both final-save branches and final
   preview; earlier stage previews must remain unchanged by Tone Lab.
3. With workflow metadata enabled, save PNG. Compare panel values at queue time
   with embedded workflow and execution prompt, resolving the live Editing wire.
   Reload the PNG and confirm choices persist. Confirm runtime report digest and
   numerical predictions separately; metadata alone is not behavior proof.
4. Exercise On→Off, strength 1→0, model A→B, replace model contents at the same
   filename, and save/reload. Off must not require any checkpoint; enabled with
   None/missing/malformed JSON must report a model error.
5. Test Editing with grading disabled, then explicitly enabled, including
   inpaint/outpaint surroundings. Disabled editing grading must be passthrough.
6. Compare actual-image features and gamma/gain against Tone Lab, plus final
   pixel transforms. Include dark/high-key images, approved originals,
   corrected/uncorrected pairs, coloured images and common resolutions/codecs.
   Test CUDA and high-resolution memory use; browser resizing/codec/profile
   differences remain possible and must not be labelled bit-exact by assumption.

Record actual transitions, observed widget values, checkpoint digest, browser /
ComfyUI versions and PNG output paths here after running. If distributing a
statically baked JSON to Civitai, manually upload that verified JSON; this PR
itself uses the frontend import migration and leaves the existing JSON intact.

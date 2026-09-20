"""CPU regressions for the real Donut bridge + schedule, with Comfy/NAG doubles.

No model weights or GPU are needed. The NAG double follows the installed public
patch contract, including captured negative context and reference VAE priming.
These are integration-contract tests, not image-quality acceptance tests.
"""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch

import donut_grounding_nag as support
from test_donut_grounding_schedule import grounding, request


DIFFUSION = "diffusion_model"
NAG_KEY = "krea2_edit_normalized_attention_guidance"


def conditioning(px, polarity=1):
    return [[torch.full((1, max(1, px // 64), 4), float(polarity * px)),
             {"pooled_output": torch.ones(1, 4), "keep": object()}]]


class Model:
    def __init__(self):
        self.wrappers = {DIFFUSION: {"other": [object()], "donut_krea2_edit": [object()]}}
        self.model_options = {"preserve": object()}

    def clone(self):
        result = Model()
        result.wrappers = {kind: {key: list(value) for key, value in groups.items()}
                           for kind, groups in self.wrappers.items()}
        result.model_options = self.model_options.copy()
        return result

    def remove_wrappers_with_key(self, kind, key):
        self.wrappers.setdefault(kind, {}).pop(key, None)

    def add_wrapper_with_key(self, kind, key, wrapper):
        self.wrappers.setdefault(kind, {}).setdefault(key, []).append(wrapper)


class VAE:
    downscale_ratio = 8

    def __init__(self):
        self.calls = []
        self.fail = False

    def encode(self, pixels, *args, **kwargs):
        if self.fail:
            raise RuntimeError("VAE interrupted")
        self.calls.append((pixels.clone(), args, kwargs))
        return pixels.mean(dim=-1)


class NAGTests(unittest.TestCase):
    def setUp(self):
        self.calls, self.encodes, self.fusion_inputs = [], [], []
        calls, encodes = self.calls, self.encodes
        class NAGNode:
            def patch(self, **arguments):
                calls.append(arguments)
                if arguments.get("vae") is not None:
                    # Upstream prepares the same pixels again for each variant.
                    for name in ("source_image", "source_image_b"):
                        image = arguments.get(name)
                        if image is not None:
                            arguments["vae"].encode(image.clone())
                model = arguments["model"].clone()
                negative = arguments["nag_negative"][0][0]
                def forward():
                    return negative
                key = NAG_KEY if "source_latent" in arguments else support._NAG_KEY
                model.add_wrapper_with_key(DIFFUSION, key, forward)
                return (model,)
        class Encoder:
            def encode(self, clip, prompt, **options):
                encodes.append((clip, prompt, options))
                return (conditioning(options["grounding_px"], -1 if prompt == "negative" else 1),)
        class ZeroOut:
            def zero_out(self, cond):
                return ([[torch.zeros_like(t), dict(m, pooled_output=torch.zeros_like(m["pooled_output"]))]
                         for t, m in cond],)
        nodes = types.ModuleType("nodes")
        nodes.NODE_CLASS_MAPPINGS = {"Krea2EditNormalizedAttentionGuidance": NAGNode,
                                   "Krea2NormalizedAttentionGuidance": NAGNode,
                                   "Krea2EditGroundedEncode": Encoder}
        nodes.ConditioningZeroOut = ZeroOut
        comfy = types.ModuleType("comfy")
        extension = types.ModuleType("comfy.patcher_extension")
        extension.WrappersMP = types.SimpleNamespace(
            DIFFUSION_MODEL=DIFFUSION, PREDICT_NOISE="predict", SAMPLER_SAMPLE="sample")
        comfy.patcher_extension = extension
        fusion = types.ModuleType("DonutKrea2FusionControl")
        def prepare(model, cond):
            self.fusion_inputs.append(cond)
            return [[t * 2, m.copy()] for t, m in cond]
        fusion.prepare_nag_conditioning = prepare
        edit = types.ModuleType("krea2_edit_integration")
        edit.scale_image_to_megapixels = lambda image: image
        variance = types.ModuleType("krea2_variance_integration")
        variance.reapply_edit_variance = lambda cond, donor: [[t, dict(m, donor=donor)] for t, m in cond]
        spec = importlib.util.spec_from_file_location(
            "krea2_nag_integration", Path(__file__).with_name("krea2_nag_integration.py"))
        self.bridge = importlib.util.module_from_spec(spec)
        modules = {"nodes": nodes, "comfy": comfy, "comfy.patcher_extension": extension,
                   "DonutKrea2FusionControl": fusion, "krea2_edit_integration": edit,
                   "krea2_variance_integration": variance, spec.name: self.bridge}
        self.modules = patch.dict(sys.modules, modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        spec.loader.exec_module(self.bridge)
        self.nodes = nodes
        self.model, self.vae = Model(), VAE()
        self.image_a, self.image_b = torch.ones(1, 8, 12, 3), torch.zeros(1, 8, 12, 3)
        self.mask, self.target = object(), {"samples": torch.zeros(1, 16, 1, 2)}
        self.sources = [{"samples": torch.ones(1, 16, 1, 2)}, {"samples": torch.zeros(1, 16, 1, 2)}]

    def prepare(self, values=(512, 768, 1088), turbo=True, **overrides):
        arguments = dict(nag_enabled=True, nag_phi=4., nag_alpha=.25, nag_tau=3.,
                         nag_sigma_start=9., nag_sigma_end=.1,
                         nag_ref_boost=1.6, nag_ref_boost_a=.8,
                         nag_ref_boost_mask=self.mask, nag_fit_mode="fit",
                         source_latent=self.sources, vae=self.vae,
                         source_image=self.image_a, source_image_b=self.image_b,
                         target_latent=self.target)
        arguments.update(overrides)
        raw_negative = conditioning(512, -1)
        model = self.bridge.apply_krea2_nag(self.model, raw_negative, **arguments)
        self.prepared = model
        return grounding._prepare_conditions(
            request(image=self.image_a, image_b=self.image_b, turbo=turbo), model,
            conditioning(512), self.bridge.sampler_negative(raw_negative, turbo), values)

    def exercise(self, model, positive, negative, values, fail=False):
        original = {"positive": [dict(m, tensor=t) for t, m in positive],
                    "negative": [dict(m, tensor=t) for t, m in negative]}
        guider = types.SimpleNamespace(conds=original, _step_index=0, cfg_values=[1.] * len(values))
        options = {"transformer_options": {"wrappers": model.wrappers, "sigmas": object(),
                                            "patches": {"upstream": object()}}, "seed": 42}
        original_nag = options["transformer_options"]["wrappers"][DIFFUSION][NAG_KEY]
        seen = []
        class Executor:
            class_obj = guider
            def __call__(self, x, timestep, selected_options, seed):
                selected_wrappers = selected_options["transformer_options"]["wrappers"][DIFFUSION]
                seen.append((guider.conds["positive"][0]["tensor"], selected_wrappers[NAG_KEY][0]()))
                assert selected_wrappers["other"] is model.wrappers[DIFFUSION]["other"]
                assert selected_options["transformer_options"]["patches"] is options["transformer_options"]["patches"]
                if fail:
                    raise RuntimeError("denoiser interrupted")
                return x
        selector = model.wrappers["predict"][grounding._WRAPPER_KEY][0]
        for index in range(len(values)):
            guider._step_index = index
            try:
                selector(Executor(), "x", "sigma", options, 42)
            finally:
                self.assertIs(guider.conds, original)
                self.assertIs(options["transformer_options"]["wrappers"][DIFFUSION][NAG_KEY], original_nag)
        return seen

    def test_turbo_switches_unzeroed_nag_negative_with_positive(self):
        values = (512, 512, 768, 1088)
        with support.capture_nag_preparations():
            model, positive, negative = self.prepare(values)
        seen = self.exercise(model, positive, negative, values)
        self.assertEqual([float(p.flatten()[0]) for p, _ in seen], list(values))
        self.assertEqual([float(n.flatten()[0]) for _, n in seen], [-2. * v for v in values])
        self.assertTrue(all(torch.count_nonzero(t) == 0 for t, _ in negative))
        self.assertTrue(all(torch.count_nonzero(m["pooled_output"]) == 0 for _, m in negative))
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(len(self.encodes), 4)
        self.assertEqual(len(self.fusion_inputs), 3)
        self.assertEqual(len(self.vae.calls), 2)  # A/B once, not once per resolution
        self.assertEqual([p.shape[1] for p, _ in seen], [v // 64 for v in values])

    def test_raw_negative_remains_nonzero(self):
        with support.capture_nag_preparations():
            model, pos, neg = self.prepare(turbo=False)
        self.assertEqual([float(t.flatten()[0]) for t, _ in neg], [-512., -768., -1088.])
        self.exercise(model, pos, neg, (512, 768, 1088))

    def test_zero_phi_or_alpha_keeps_reference_guidance_forward(self):
        for option in ({"nag_phi": 0.}, {"nag_alpha": 0.}):
            with self.subTest(option=option), support.capture_nag_preparations():
                before = len(self.calls)
                model, pos, neg = self.prepare(**option)
                self.assertEqual(len(self.calls), before + 1)
                self.assertIn(NAG_KEY, model.wrappers[DIFFUSION])
                self.assertIsNone(model.wrappers["predict"][grounding._WRAPPER_KEY][0].nag_wrappers)
                self.exercise(model, pos, neg, (512, 768, 1088))

    def test_explicit_negative_override_is_preserved_at_every_step(self):
        explicit = conditioning(320, -1)
        with support.capture_nag_preparations():
            model, pos, neg = self.prepare(nag_negative=explicit)
        seen = self.exercise(model, pos, neg, (512, 768, 1088))
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(all(torch.equal(n, explicit[0][0] * 2) for _, n in seen))
        self.assertIs(self.fusion_inputs[0], explicit)

    def test_reference_geometry_masks_and_nag_parameters_forwarded(self):
        for fit_mode in ("fit", "crop (legacy)"):
            with self.subTest(fit_mode=fit_mode), support.capture_nag_preparations():
                self.prepare(nag_fit_mode=fit_mode)
                for call in self.calls[-3:]:
                    for key, value in (("source_latent", self.sources[0]), ("source_latent_b", self.sources[1]),
                                       ("source_image", self.image_a), ("source_image_b", self.image_b),
                                       ("ref_boost_mask", self.mask), ("target_latent", self.target)):
                        self.assertIs(call[key], value)
                    self.assertEqual((call["phi"], call["tau"], call["alpha"]), (4., 3., .25))
                    self.assertEqual((call["sigma_start"], call["sigma_end"]), (9., .1))
                    self.assertEqual((call["ref_boost"], call["ref_boost_a"]), (1.6, .8))
                    self.assertEqual(call["fit_mode"], fit_mode)
                    self.assertIs(call["vae"].vae, self.vae)
        self.assertEqual(set(self.model.wrappers[DIFFUSION]), {"other", "donut_krea2_edit"})

    def test_single_reference_does_not_create_second_reference(self):
        with support.capture_nag_preparations():
            self.prepare(source_latent=self.sources[0], source_image_b=None)
        self.assertTrue(all(c["source_latent_b"] is None and c["source_image_b"] is None for c in self.calls))
        self.assertEqual(len(self.vae.calls), 1)

    def test_one_step_uses_end_for_both_contexts(self):
        with support.capture_nag_preparations():
            model, pos, neg = self.prepare(values=(1088,))
        seen = self.exercise(model, pos, neg, (1088,))
        self.assertEqual(float(seen[0][1].flatten()[0]), -2176.)
        self.assertEqual(len(self.calls), 2)

    def test_denoiser_exception_restores_conditions_and_options(self):
        with support.capture_nag_preparations():
            model, pos, neg = self.prepare()
        with self.assertRaisesRegex(RuntimeError, "denoiser interrupted"):
            self.exercise(model, pos, neg, (512, 768, 1088), fail=True)
        self.assertIsNone(support._PREPARATIONS.get())
        self.exercise(model, pos, neg, (512, 768, 1088))

    def test_nested_capture_and_failure_do_not_leak(self):
        with support.capture_nag_preparations():
            self.prepare()
            outer = support.get_nag_preparation(self.prepared)
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                with support.capture_nag_preparations():
                    self.assertIsNone(support.get_nag_preparation(outer.patched))
                    raise RuntimeError("interrupted")
            self.assertIs(support.get_nag_preparation(outer.patched), outer)
            with support.capture_nag_preparations(enabled=False):
                args = {"vae": self.vae}
                self.assertIs(support.prepare_nag_arguments(args), args)
                self.assertIsNone(support.get_nag_preparation(outer.patched))
        self.assertIsNone(support.get_nag_preparation(outer.patched))

    def test_constant_after_dynamic_retains_no_recipe_and_original_vae(self):
        with support.capture_nag_preparations():
            self.prepare()
        model = self.bridge.apply_krea2_nag(self.model, conditioning(512, -1), nag_enabled=True,
                                           source_latent=self.sources[0], vae=self.vae,
                                           source_image=self.image_a, target_latent=self.target)
        self.assertIsNone(support.get_nag_preparation(model))
        self.assertIs(self.calls[-1]["vae"], self.vae)

    def test_missing_runtime_wrapper_is_not_silently_ignored(self):
        with self.assertRaisesRegex(RuntimeError, "missing from sampling options"):
            support.select_nag_options({}, {NAG_KEY: [object()]})

    def test_missing_registered_wrapper_reports_incompatible_install(self):
        with support.capture_nag_preparations():
            self.prepare()
            recipe = support.get_nag_preparation(self.prepared)
            self.prepared.wrappers[DIFFUSION].pop(NAG_KEY)
            with self.assertRaisesRegex(RuntimeError, "did not register"):
                recipe.wrappers_for()

    def test_disabled_nag_requires_no_optional_nag_node(self):
        with patch.dict(self.nodes.NODE_CLASS_MAPPINGS, {}, clear=True):
            self.assertIs(self.bridge.apply_krea2_nag(self.model, None), self.model)


class VAECacheTests(unittest.TestCase):
    def setUp(self):
        self.vae = VAE()
        self.cache = support._ReferenceVAECache(self.vae)

    def test_content_not_shape_or_pointer_determines_hit(self):
        first = torch.ones(1, 4, 4, 3)
        a = self.cache.encode(first)
        self.assertIs(self.cache.encode(first.clone()), a)
        self.cache.encode(torch.zeros_like(first))
        self.assertEqual(len(self.vae.calls), 2)
        first.zero_()
        self.cache.encode(first)
        self.assertEqual(len(self.vae.calls), 2)

    def test_failed_encoding_is_not_cached(self):
        self.vae.fail = True
        with self.assertRaisesRegex(RuntimeError, "VAE interrupted"):
            self.cache.encode(torch.ones(1, 4, 4, 3))
        self.assertEqual(self.cache.entries, [])
        self.vae.fail = False
        self.cache.encode(torch.ones(1, 4, 4, 3))
        self.assertEqual(len(self.vae.calls), 1)

    def test_bound_dtype_and_nonstandard_calls(self):
        for index in range(6):
            self.cache.encode(torch.full((1, 4, 4, 3), float(index)))
        self.assertEqual(len(self.cache.entries), 4)
        image = torch.full((1, 4, 4, 3), 5.)
        self.cache.encode(image.double())
        self.cache.encode(image, tiled=True)
        self.assertEqual(len(self.vae.calls), 8)
        self.assertEqual(self.cache.downscale_ratio, 8)


if __name__ == "__main__":
    unittest.main()

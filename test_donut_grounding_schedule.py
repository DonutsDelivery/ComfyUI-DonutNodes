"""Focused CPU/stub tests; no ComfyUI installation or model download required."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

# Retain one ContextVar owner when the temporary base-module stub is removed.
import donut_grounding_nag


class BaseSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)},
                "optional": {"grounding_px": ("INT", {"default": 768}),
                             "sda_enabled": ("BOOLEAN", {"default": False})}}

    def sample(self, model, sampler_name="euler", steps=5, positive=None, negative=None,
               edit_mode=False, grounding_px=768, mode="simple", start_at_step=0,
               end_at_step=10000, clip=None, source_image=None, source_image_b=None,
               edit_prompt="", edit_negative_prompt="", edit_inpaint=None,
               turbo_mode=False, sda_enabled=False, **nag_options):
        self.received = locals().copy()
        if getattr(self, "fail", False):
            raise RuntimeError("interrupted")
        if getattr(self, "dispatch", False):
            if mode == "advanced":
                return self.run_advanced(model, steps, positive, negative,
                                         start_at_step, end_at_step)
            return self.run_simple(model, steps, positive, negative)
        return model, "base result"

    def run_simple(self, model, steps, positive, negative, denoise=1.0):
        self.run_received = locals().copy()
        return model, "simple result"

    def run_advanced(self, model, steps, positive, negative,
                     start_at_step=0, end_at_step=10000, denoise=1.0):
        self.run_received = locals().copy()
        return model, "advanced result"


base_module = types.ModuleType("donut_krea2_sda")
base_module.DonutSampler = BaseSampler
spec = importlib.util.spec_from_file_location(
    "grounding_under_test", Path(__file__).with_name("donut_grounding_schedule.py"),
)
grounding = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"donut_krea2_sda": base_module, spec.name: grounding}):
    spec.loader.exec_module(grounding)


def request(**changes):
    values = dict(start=512, end=1088, curve="linear", clip="clip", image="image-a",
                  image_b=None, prompt="positive", negative_prompt="negative",
                  original_positive="variance-donor", turbo=False)
    values.update(changes)
    return grounding._Request(**values)


class ScheduleTests(unittest.TestCase):
    def test_linear(self):
        self.assertEqual(grounding.grounding_values(0, 1024, 5, "linear"), (0, 256, 512, 768, 1024))

    def test_ease_in(self):
        self.assertEqual(grounding.grounding_values(0, 1024, 5, "ease_in"), (0, 64, 256, 576, 1024))

    def test_ease_out(self):
        self.assertEqual(grounding.grounding_values(0, 1024, 5, "ease_out"), (0, 448, 768, 960, 1024))

    def test_ease_in_out(self):
        self.assertEqual(grounding.grounding_values(0, 1024, 5, "ease_in_out"), (0, 128, 512, 896, 1024))

    def test_documented_example(self):
        self.assertEqual(grounding.grounding_values(512, 1088, 8, "ease_in"),
                         (512, 512, 576, 640, 704, 832, 960, 1088))

    def test_zero_and_single_step(self):
        self.assertEqual(grounding.grounding_values(512, 1088, 0, "linear"), ())
        self.assertEqual(grounding.grounding_values(512, 1088, 1, "linear"), (1088,))

    def test_equal_endpoints(self):
        self.assertEqual(grounding.grounding_values(577, 577, 4, "ease_out"), (577,) * 4)

    def test_monotonic_bounded_and_exact_unaligned_endpoints(self):
        for curve in grounding.CURVES[1:]:
            for start, end in ((0, 4096), (4096, 0), (513, 1087), (1087, 513)):
                with self.subTest(curve=curve, start=start, end=end):
                    values = grounding.grounding_values(start, end, 137, curve)
                    self.assertEqual((values[0], values[-1]), (start, end))
                    self.assertTrue(all(min(start, end) <= value <= max(start, end) for value in values))
                    self.assertEqual(list(values), sorted(values, reverse=start > end))

    def test_bad_px(self):
        for value in (-1, 4097, 1.5, "512", True, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                grounding.grounding_values(value, 1024, 5, "linear")

    def test_bad_curve_or_count(self):
        with self.assertRaises(ValueError):
            grounding.grounding_values(0, 1024, 5, "unknown")
        with self.assertRaises(ValueError):
            grounding.grounding_values(0, 1024, -1, "linear")


class NodeTests(unittest.TestCase):
    def test_widgets_append_without_mutating_parent(self):
        old = BaseSampler.INPUT_TYPES()
        new = grounding.DonutSampler.INPUT_TYPES()
        self.assertEqual(list(new["optional"])[:2], list(old["optional"]))
        self.assertEqual(list(new["optional"])[2:],
                         ["grounding_schedule", "grounding_start_px", "grounding_end_px"])
        self.assertEqual(new["optional"]["grounding_schedule"][1]["default"], "constant")
        self.assertEqual(BaseSampler.INPUT_TYPES(), old)

    def test_constant_preserves_original_arguments(self):
        node, model = grounding.DonutSampler(), object()
        self.assertEqual(node.sample(model, grounding_px=777, grounding_start_px="ignored"),
                         (model, "base result"))
        self.assertEqual(node.received["grounding_px"], 777)
        self.assertEqual(node.received["nag_options"], {})

    def test_non_edit_ignores_dynamic_controls(self):
        node = grounding.DonutSampler()
        node.sample("model", grounding_schedule="invalid", grounding_start_px="ignored")
        self.assertEqual(node.received["grounding_px"], 768)

    def test_equal_endpoints_use_static_path_even_with_nag(self):
        node = grounding.DonutSampler()
        node.sample("model", edit_mode=True, grounding_schedule="linear",
                    grounding_start_px=640, grounding_end_px=640, nag_enabled=True)
        self.assertEqual(node.received["grounding_px"], 640)
        self.assertIsNone(grounding._REQUEST.get())

    def test_native_zero_is_not_treated_as_disabled_grounding(self):
        for start, end in ((0, 1024), (512, 0)):
            with self.subTest(start=start, end=end), self.assertRaisesRegex(ValueError, "native/unlimited"):
                grounding.DonutSampler().sample("model", edit_mode=True, grounding_schedule="linear",
                                                grounding_start_px=start, grounding_end_px=end)
        node = grounding.DonutSampler()
        node.sample("model", edit_mode=True, grounding_schedule="linear",
                    grounding_start_px=0, grounding_end_px=0)
        self.assertEqual(node.received["grounding_px"], 0)

    def test_multimodel_rejected(self):
        with self.assertRaisesRegex(ValueError, "multi_model"):
            grounding.DonutSampler().sample("model", edit_mode=True,
                                            grounding_schedule="linear", mode="multi_model")

    def test_nag_enabled_and_zero_strength_are_forwarded(self):
        for phi, alpha in ((0., .25), (4., 0.), (4., .25)):
            node = grounding.DonutSampler()
            node.sample("model", edit_mode=True, grounding_schedule="linear",
                        nag_enabled=True, nag_phi=phi, nag_alpha=alpha)
            self.assertEqual(node.received["nag_options"],
                             dict(nag_enabled=True, nag_phi=phi, nag_alpha=alpha))
            self.assertIsNone(donut_grounding_nag._PREPARATIONS.get())

    def test_multievaluation_sampler_rejected(self):
        with self.assertRaisesRegex(ValueError, "Euler"):
            grounding.DonutSampler().sample("model", edit_mode=True,
                                            grounding_schedule="linear", sampler_name="heun")

    def test_request_cleanup_on_exception(self):
        node = grounding.DonutSampler()
        node.fail = True
        sentinel = object()
        token = grounding._REQUEST.set(sentinel)
        try:
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                node.sample("model", edit_mode=True, grounding_schedule="linear")
            self.assertIs(grounding._REQUEST.get(), sentinel)
        finally:
            grounding._REQUEST.reset(token)

    def test_inpaint_and_two_references_reach_preparation(self):
        node = grounding.DonutSampler()
        node.dispatch = True
        def prepare(req, model, positive, negative, values):
            self.assertEqual(req.image, "masked-base")
            self.assertEqual(req.image_b, "subject")
            self.assertEqual(req.original_positive, "original-positive")
            return model, positive, negative
        with patch.object(grounding, "_prepare_conditions", side_effect=prepare) as mocked:
            node.sample("model", positive="original-positive", edit_mode=True,
                        grounding_schedule="linear", source_image="unused",
                        source_image_b="subject", edit_inpaint={"image": "masked-base"})
        self.assertEqual(mocked.call_count, 1)
        self.assertIsNone(grounding._REQUEST.get())

    def test_advanced_spans_executed_steps(self):
        node = grounding.DonutSampler()
        node.dispatch = True
        def prepare(req, model, positive, negative, values):
            self.assertEqual(values, (512, 832, 1088))
            return model, positive, negative
        with patch.object(grounding, "_prepare_conditions", side_effect=prepare):
            _, info = node.sample("model", steps=20, mode="advanced", end_at_step=3,
                                  edit_mode=True, grounding_schedule="linear")
        self.assertIn("(3 steps)", info)
        self.assertEqual(node.run_received["steps"], 20)

    def test_zero_step_range_does_not_encode(self):
        node = grounding.DonutSampler()
        node.dispatch = True
        with patch.object(grounding, "_prepare_conditions") as prepare:
            node.sample("model", mode="advanced", end_at_step=0,
                        edit_mode=True, grounding_schedule="linear")
        prepare.assert_not_called()


class WrapperTests(unittest.TestCase):
    def setUp(self):
        self.conditions = {
            "positive": [{grounding._TAG: 512, "tensor": "short"},
                         {grounding._TAG: 1088, "tensor": "long"}],
            "negative": None,
        }
        self.guider = types.SimpleNamespace(_step_index=0, cfg_values=(1, 1), conds=self.conditions)
        self.seen = []
        self.executor = types.SimpleNamespace()

    def executor_for(self, function):
        class Executor:
            class_obj = self.guider
            def __call__(self, *args):
                return function(*args)
        return Executor()

    def test_selects_one_resolution_and_restores(self):
        wrapper = grounding._SelectGrounding((512, 1088))
        def execute(*args):
            self.seen.append(self.guider.conds["positive"][0]["tensor"])
            self.assertEqual(len(self.guider.conds["positive"]), 1)
            self.assertIsNone(self.guider.conds["negative"])
            return "prediction"
        executor = self.executor_for(execute)
        for index in (0, 0, 1):
            self.guider._step_index = index
            self.assertEqual(wrapper(executor, "x", "sigma", {}, 42), "prediction")
            self.assertIs(self.guider.conds, self.conditions)
        self.assertEqual(self.seen, ["short", "short", "long"])

    def test_restores_after_failure(self):
        def fail(*args):
            raise RuntimeError("model interrupted")
        with self.assertRaises(RuntimeError):
            grounding._SelectGrounding((512, 1088))(self.executor_for(fail), "x", "t")
        self.assertIs(self.guider.conds, self.conditions)

    def test_rejects_incompatible_guider(self):
        self.guider._step_index = None
        with self.assertRaisesRegex(RuntimeError, "step-aware"):
            grounding._SelectGrounding((512, 1088))(self.executor_for(lambda *a: None), "x", "t")

    def test_missing_resolution_is_not_silently_empty(self):
        with self.assertRaisesRegex(RuntimeError, "no positive"):
            grounding._SelectGrounding((640, 1088))(self.executor_for(lambda *a: None), "x", "t")
        self.assertIs(self.guider.conds, self.conditions)

    def test_tag_does_not_mutate_metadata(self):
        metadata = {"hooks": object()}
        tagged = grounding._tag([["tensor", metadata]], 512)
        self.assertNotIn(grounding._TAG, metadata)
        self.assertIs(tagged[0][1]["hooks"], metadata["hooks"])

    def test_runtime_guard_passthrough(self):
        seen = []
        inspector = types.ModuleType("donut_sda_sampler")
        inspector.inspect_sda_sampler = lambda sampler, names: seen.append((sampler, names))
        class Executor:
            class_obj = "original-preset"
            def __call__(self, *args):
                return args
        args = ("model", [1.0, 0.6, 0.1], {"seed": 42}, object(), "noise", "latent", "mask", True)
        with patch.dict(sys.modules, {"donut_sda_sampler": inspector}):
            result = grounding._SamplingGuard(2)(Executor(), *args)
        self.assertEqual(result, args)
        self.assertEqual(seen, [("original-preset", grounding.SUPPORTED_SAMPLERS)])

    def test_bad_sigma_schedules_rejected_before_execution(self):
        for sigmas in ([1, 0], [1, 1, 0], [1, float("nan"), 0], [1, 0, -1]):
            with self.subTest(sigmas=sigmas), self.assertRaises(ValueError):
                grounding._SamplingGuard(2)(None, None, sigmas, {}, None, None)

    def test_guard_reports_grounding_not_sda(self):
        inspector = types.ModuleType("donut_sda_sampler")
        def reject(*args):
            raise ValueError("SDA cannot verify override_sigmas_opt")
        inspector.inspect_sda_sampler = reject
        executor = types.SimpleNamespace(class_obj="preset")
        with patch.dict(sys.modules, {"donut_sda_sampler": inspector}):
            with self.assertRaisesRegex(ValueError, "Scheduled grounding.*override_sigmas_opt"):
                grounding._SamplingGuard(2)(executor, None, [1, 0.5, 0], {}, None, None)


class PreparationTests(unittest.TestCase):
    def test_encode_deduplicate_metadata_clone_and_negative(self):
        encoded, scaled = [], []
        class Encoder:
            def encode(self, clip, prompt, **kwargs):
                encoded.append((clip, prompt, kwargs))
                return ([[f"{prompt}-{kwargs['grounding_px']}", {"encoder": True}]],)
        class Model:
            def __init__(self):
                self.wrappers = []
            def clone(self):
                return Model()
            def add_wrapper_with_key(self, *args):
                self.wrappers.append(args)
        nodes = types.ModuleType("nodes")
        nodes.NODE_CLASS_MAPPINGS = {"Krea2EditGroundedEncode": Encoder}
        comfy = types.ModuleType("comfy")
        extension = types.ModuleType("comfy.patcher_extension")
        extension.WrappersMP = types.SimpleNamespace(PREDICT_NOISE="predict", SAMPLER_SAMPLE="sample")
        comfy.patcher_extension = extension
        edit = types.ModuleType("krea2_edit_integration")
        def scale(image):
            scaled.append(image)
            return "scaled-" + image
        edit.scale_image_to_megapixels = scale
        variance = types.ModuleType("krea2_variance_integration")
        variance.reapply_edit_variance = lambda cond, donor: [[t, dict(m, donor=donor)] for t, m in cond]
        nag = types.ModuleType("krea2_nag_integration")
        nag.sampler_negative = lambda cond, turbo: [[t, dict(m, turbo=turbo)] for t, m in cond]
        modules = {"nodes": nodes, "comfy": comfy, "comfy.patcher_extension": extension,
                   "krea2_edit_integration": edit, "krea2_variance_integration": variance,
                   "krea2_nag_integration": nag}
        model = Model()
        pos, neg = [["initial-pos", {"original": True}]], [["initial-neg", {}]]
        with patch.dict(sys.modules, modules):
            cloned, positives, negatives = grounding._prepare_conditions(
                request(image_b="image-b", turbo=True), model, pos, neg, (512, 512, 768, 1088),
            )
        self.assertIsNot(cloned, model)
        self.assertEqual(model.wrappers, [])
        self.assertEqual(len(cloned.wrappers), 2)
        self.assertEqual(len(encoded), 4)  # two new resolutions, two polarities
        self.assertEqual(scaled, ["image-a", "image-b"])
        self.assertTrue(all(call[2]["image_b"] == "scaled-image-b" for call in encoded))
        self.assertEqual([m[grounding._TAG] for _, m in positives], [512, 768, 1088])
        self.assertEqual(len(negatives), 3)
        self.assertTrue(positives[0][1]["original"])
        self.assertEqual(positives[1][1]["donor"], "variance-donor")
        self.assertTrue(negatives[1][1]["turbo"])
        self.assertNotIn(grounding._TAG, pos[0][1])


if __name__ == "__main__":
    unittest.main()

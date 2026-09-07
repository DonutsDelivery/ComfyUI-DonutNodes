"""Forward-output regressions for UncensorFix after Krea2 merge bypass.

Uses the real Donut merge node, dynamic swap injection, plan resolver and save
composition with small CPU Linear layers. ComfyUI patcher/adapter/manager APIs
are explicit test doubles. This is not a quantized/GPU generation test.
"""
import contextlib
import copy
import importlib.util
import io
import sys
import types
import unittest
import uuid
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
PREFIX = "diffusion_model."
LORA_KEY = "donut_bypass_lora"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Adapter:
    def __init__(self, loaded_keys=(), weights=()):
        self.loaded_keys, self.weights = set(loaded_keys), weights

    def delta(self):
        up, down, alpha, *_ = self.weights
        return (up @ down) * (alpha / down.shape[0])

    def bypass_forward(self, original, x, *args, **kwargs):
        return original(x, *args, **kwargs) + F.linear(x, self.delta()) * self.multiplier


class Injection:
    def __init__(self, inject, eject):
        self.inject, self.eject = inject, eject


class Manager:
    def __init__(self):
        self.adapters, self.hooks = {}, []

    def add_adapter(self, key, adapter, strength=1.0):
        self.adapters[key.removesuffix(".weight")] = (adapter, strength)

    def create_injections(self, root):
        self.hooks = [(root.get_submodule(path), adapter, strength)
                      for path, (adapter, strength) in self.adapters.items()]
        originals = []

        def inject(patcher):
            for layer, adapter, strength in self.hooks:
                original = layer.forward
                originals.append((layer, original))
                adapter.multiplier = strength
                layer.forward = lambda x, *a, _p=adapter, _f=original, **kw: _p.bypass_forward(_f, x, *a, **kw)

        def eject(patcher):
            for layer, original in reversed(originals):
                layer.forward = original
            originals.clear()

        return [Injection(inject, eject)]

    def get_hook_count(self):
        return len(self.hooks)


class Patcher:
    """Ordered patch lists and shared roots, rather than dict.update mocks."""
    def __init__(self, root):
        self.model = root
        self.raw = {k: v.detach().clone() for k, v in root.state_dict().items()}
        self.patches, self.injections, self.additional_models, self.attachments = {}, {}, {}, {}
        self.model_options = {"transformer_options": {}}
        self.patches_uuid = uuid.uuid4()
        self.is_injected = False
        self.load_device = torch.device("cpu")
        self.reject = set()

    def clone(self):
        n = copy.copy(self)
        n.patches = {k: list(v) for k, v in self.patches.items()}
        n.injections = {k: v.copy() for k, v in self.injections.items()}
        n.attachments = dict(self.attachments)
        n.additional_models = {k: [p.clone() for p in v] for k, v in self.additional_models.items()}
        n.model_options = copy.deepcopy(self.model_options)
        n.is_injected = False
        return n

    def is_clone(self, other):
        return self.model is other.model

    def clone_has_same_weights(self, other):
        # Cache-relevant subset of core ModelPatcher's comparison: it compares
        # attachment/additional-model KEYS, not retained source patch contents.
        if not self.is_clone(other) or self.attachments.keys() != other.attachments.keys():
            return False
        if self.additional_models.keys() != other.additional_models.keys():
            return False
        if self.injections.keys() != other.injections.keys():
            return False
        if not self.patches and not other.patches:
            return True
        return self.patches_uuid == other.patches_uuid and len(self.patches) == len(other.patches)

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        accepted = set(patches).intersection(self.raw) - self.reject
        for key in accepted:
            self.patches.setdefault(key, []).append((strength_patch, patches[key], strength_model))
        self.patches_uuid = uuid.uuid4()
        return list(accepted)

    def effective(self, key):
        value = self.raw[key].clone()
        for strength, patch, model_strength in self.patches.get(key, ()):
            delta = patch.delta() if isinstance(patch, Adapter) else patch
            value = value * model_strength + delta * strength
        return value

    def get_key_patches(self, prefix):
        return {k: self.effective(k) for k in self.raw if k.startswith(prefix)}

    def set_injections(self, key, value):
        self.injections[key] = value

    def remove_injections(self, key):
        self.injections.pop(key, None)

    def set_attachments(self, key, value):
        self.attachments[key] = value

    def remove_attachments(self, key):
        self.attachments.pop(key, None)

    def set_additional_models(self, key, value):
        self.additional_models[key] = value

    def get_additional_models_with_key(self, key):
        return self.additional_models.get(key, [])

    def remove_additional_models(self, key):
        self.additional_models.pop(key, None)

    def model_state_dict_for_saving(self, model, prefix):
        return {k.removeprefix(prefix): self.effective(k) for k in self.raw if k.startswith(prefix)}

    @contextlib.contextmanager
    def activate(self):
        # Minimal CPU load lifecycle: additional sources first, then ordered
        # patches, then existing runtime injections. Always restore shared roots.
        with contextlib.ExitStack() as stack:
            for models in self.additional_models.values():
                for model in models:
                    stack.enter_context(model.activate())
            saved = {k: v.clone() for k, v in self.model.state_dict().items()}
            active = []
            try:
                with torch.no_grad():
                    for key, value in self.model.state_dict().items():
                        value.copy_(self.effective(key))
                for injections in self.injections.values():
                    for inj in injections:
                        active.append(inj)
                        inj.inject(self)
                self.is_injected = bool(active)
                yield
            finally:
                for inj in reversed(active):
                    inj.eject(self)
                self.is_injected = False
                with torch.no_grad():
                    for key, value in self.model.state_dict().items():
                        value.copy_(saved[key])


def make_root(keys, value):
    root = torch.nn.Module()
    for key in keys:
        parts = key.removesuffix(".weight").split(".")
        module = root
        for i, part in enumerate(parts[:-1]):
            if part.isdigit():
                while len(module) <= int(part):
                    module.append(torch.nn.Module())
                module = module[int(part)]
            else:
                if not hasattr(module, part):
                    child = torch.nn.ModuleList() if parts[i + 1].isdigit() else torch.nn.Module()
                    module.add_module(part, child)
                module = getattr(module, part)
        layer = torch.nn.Linear(3, 2, bias=True)
        with torch.no_grad():
            layer.weight.fill_(value)
            layer.bias.fill_(value / 10)
        module.add_module(parts[-1], layer)
    return root


class MergeBypassTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        package_name = "_uncensorfix_merge_tests"
        package = types.ModuleType(package_name)
        package.__path__ = [str(ROOT)]
        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        adapters = types.ModuleType("comfy.weight_adapter")
        adapters.__path__ = []
        adapters.WeightAdapterBase = Adapter
        adapters.BypassInjectionManager = Manager
        comfy.weight_adapter = adapters
        lora = types.ModuleType("comfy.weight_adapter.lora")
        lora.LoRAAdapter = Adapter
        extension = types.ModuleType("comfy.patcher_extension")
        extension.PatcherInjection = Injection
        cls.modules = mock.patch.dict(sys.modules, {
            package_name: package, "comfy": comfy, "comfy.weight_adapter": adapters,
            "comfy.weight_adapter.lora": lora, "comfy.patcher_extension": extension,
            "folder_paths": None, "comfy.lora": None, "safetensors": None,
        })
        cls.modules.start()
        cls.addClassCleanup(cls.modules.stop)
        # Reuse the original suite's tiny base-node definition only; import the
        # real merge and serializer modules without importing pack __init__.
        original_tests = load_module(package_name + ".original_tests", ROOT / "test_krea2_fusion_preset.py")
        base = types.ModuleType(package_name + ".DonutKrea2FusionControl")
        for key, value in original_tests.BASE_NAMES.items():
            setattr(base, key, value)

        class BaseNode:
            def apply(self, **kwargs):
                return (kwargs["model"], kwargs["conditioning_in_1"], None, None, None,
                        "preset_label=Custom settings; preset_is_ui_only=true\nexternal_files_loaded=none")

        base.DonutKrea2FusionControl = BaseNode
        package.DonutKrea2FusionControl = base
        sys.modules[base.__name__] = base
        cls.node = load_module(package_name + ".DonutKrea2FusionPreset", ROOT / "DonutKrea2FusionPreset.py")
        cls.merge = load_module(package_name + ".DonutModelMergeKrea2", ROOT / "DonutModelMergeKrea2.py")
        cls.serialization = load_module(package_name + ".donut_krea2_merge_serialization", ROOT / "donut_krea2_merge_serialization.py")
        weights = load_module(package_name + ".uncensorfix_weights", ROOT / "uncensorfix_weights.py")
        cls.keys = tuple(target[0] for target in weights.TARGETS)
        cls.factors = tuple((k, torch.full((2, 2), .02 + i / 1000),
                            torch.full((2, 3), .1), 2.0) for i, k in enumerate(cls.keys))
        cls.factor_patch = mock.patch.object(cls.node, "_uncensorfix_factors", return_value=cls.factors)
        cls.factor_patch.start()
        cls.addClassCleanup(cls.factor_patch.stop)
        cls.x = torch.tensor([[.3, -.7, 1.1], [.4, .2, -.1]])

    def models(self):
        return Patcher(make_root(self.keys, .2)), Patcher(make_root(self.keys, .8))

    def merged(self, base=None, source=None, ratios=None):
        if base is None:
            base, source = self.models()
        ratios = ratios or {"first.": 1., "txtfusion.": 0.}
        with contextlib.redirect_stdout(io.StringIO()):
            result, = self.merge.DonutModelMergeKrea2().merge(base, source, "Experimental bypass", **ratios)
        return result

    def apply(self, model, strength=1.):
        return self.node._apply_uncensorfix(model, strength)[0]

    def outputs(self, model):
        with model.activate():
            return {key: model.model.get_submodule(key[:-7])(self.x).detach().clone() for key in self.keys}

    def delta_outputs(self, strength=1.):
        return {key: F.linear(self.x, up @ down * (alpha / down.shape[0])) * strength
                for key, up, down, alpha in self.factors}

    def assert_output_delta(self, before, after, strength=1.):
        for key, delta in self.delta_outputs(strength).items():
            torch.testing.assert_close(after[key], before[key] + delta, rtol=1e-5, atol=2e-7)
            self.assertFalse(torch.equal(before[key], after[key]), key)

    def add_bypass(self, model, strength=.35):
        result = model.clone()
        manager = Manager()
        for key, up, down, alpha in self.factors:
            manager.add_adapter(key, Adapter((), (up * 2, down, alpha, None, None, None)), strength)
        result.set_injections(LORA_KEY, manager.create_injections(result.model))
        return result

    def test_reproduces_old_accepted_but_inactive_patches(self):
        merged = self.merged()
        wrong = merged.clone()
        patches = {k: Adapter((), (up, down, a, None, None, None)) for k, up, down, a in self.factors}
        self.assertEqual(len(wrong.add_patches(patches)), 33)
        before, after = self.outputs(merged), self.outputs(wrong)
        for key in self.keys:
            self.assertTrue(torch.equal(before[key], after[key]))

    def test_full_swap_routes_all_33_and_changes_real_forwards(self):
        merged = self.merged()
        self.assertFalse(bool(merged.injections[self.serialization.KREA2_MERGE_INJECTION_KEY]))
        original_source = self.serialization.get_krea2_merge_bypass_info(merged)[0]
        for strength in (.75, 1., -0.5, 2.):
            with self.subTest(strength=strength):
                patched, count, details = self.node._apply_uncensorfix(merged, strength)
                source = self.serialization.get_krea2_merge_bypass_info(patched)[0]
                self.assertEqual(count, 33)
                self.assertEqual(set(source.patches), set(self.keys))
                self.assertFalse(patched.patches)
                self.assertFalse(merged.patches)
                self.assertFalse(original_source.patches)
                self.assertIsNot(source, original_source)
                self.assertIn("uncensorfix_bypass_source_targets=33", details)
                self.assert_output_delta(self.outputs(merged), self.outputs(patched), strength)

    def test_partial_and_unswapped_targets_stay_on_main(self):
        ratios = {"first.": 1., "txtfusion.layerwise_blocks.0.": 0.,
                  "txtfusion.layerwise_blocks.1.": .4, "txtfusion.projector.": 0.}
        merged = self.merged(ratios=ratios)
        before = self.outputs(merged)
        patched = self.apply(merged, .75)
        source, plans, _ = self.serialization.get_krea2_merge_bypass_info(patched)
        swapped = {key for _, key, _ in plans}
        self.assertEqual(len(swapped), 9)
        self.assertEqual(set(source.patches), swapped)
        self.assertFalse(swapped.intersection(patched.patches))
        self.assertEqual(set(patched.patches), set(merged.patches) | (set(self.keys) - swapped))
        self.assert_output_delta(before, self.outputs(patched), .75)

    def test_preserves_source_and_main_overlapping_regular_loras(self):
        base, source = self.models()
        existing = {k: Adapter((), (up * 3, down, a, None, None, None)) for k, up, down, a in self.factors}
        base.add_patches(existing, .2)
        source.add_patches(existing, .3)
        merged = self.merged(base, source, {"first.": 1., "txtfusion.layerwise_blocks.0.": 0.})
        patched = self.apply(merged)
        routed = self.serialization.get_krea2_merge_bypass_info(patched)[0]
        for key in self.keys:
            self.assertEqual(len(base.patches[key]), 1)
            self.assertEqual(len(source.patches[key]), 1)
            self.assertGreaterEqual(len(patched.patches[key]), 1)
            if key.startswith(PREFIX + "txtfusion.layerwise_blocks.0."):
                self.assertEqual(len(routed.patches[key]), 2)
                self.assertEqual(len(patched.patches[key]), 1)
            else:
                self.assertEqual(len(patched.patches[key]), 2)
        self.assert_output_delta(self.outputs(merged), self.outputs(patched))

    def test_prior_bypass_lora_survives_without_hook_replacement(self):
        merged = self.merged()
        lora_model = self.add_bypass(merged)
        patched = self.apply(lora_model, .75)
        self.assertEqual(patched.injections.keys(), lora_model.injections.keys())
        self.assertIs(patched.injections[LORA_KEY][0], lora_model.injections[LORA_KEY][0])
        self.assert_output_delta(self.outputs(lora_model), self.outputs(patched), .75)
        self.assert_output_delta(self.outputs(merged), self.outputs(lora_model), .7)

    def test_bypass_lora_added_after_uncensorfix_remains_additive(self):
        patched = self.apply(self.merged(), .75)
        later = self.add_bypass(patched)
        self.assert_output_delta(self.outputs(patched), self.outputs(later), .7)

    def test_bypass_lora_without_model_merge_keeps_normal_path(self):
        base, _ = self.models()
        lora_model = self.add_bypass(base)
        patched = self.apply(lora_model)
        self.assertEqual(set(patched.patches), set(self.keys))
        self.assertEqual(patched.additional_models, {})
        self.assert_output_delta(self.outputs(lora_model), self.outputs(patched))

    def test_changed_source_invalidates_outer_empty_patch_cache(self):
        merged = self.merged()
        low, high = self.apply(merged, .5), self.apply(merged, 1.)
        self.assertFalse(low.patches)
        self.assertFalse(high.patches)
        self.assertFalse(merged.clone_has_same_weights(low))
        self.assertFalse(low.clone_has_same_weights(high))
        self.assertTrue(high.clone_has_same_weights(high.clone()))
        self.assertNotEqual(merged.patches_uuid, low.patches_uuid)
        self.assert_output_delta(self.outputs(low), self.outputs(high), .5)
        # Return to the old graph/source after executing both patched clones.
        self.assert_output_delta(self.outputs(merged), self.outputs(low), .5)

    def test_later_clone_resolves_its_own_source_at_injection_time(self):
        patched = self.apply(self.merged(), .75)
        clone = patched.clone()
        self.assertIsNot(self.serialization.get_krea2_merge_bypass_info(patched)[0],
                         self.serialization.get_krea2_merge_bypass_info(clone)[0])
        a, b = self.outputs(patched), self.outputs(clone)
        for key in self.keys:
            self.assertTrue(torch.equal(a[key], b[key]))

    def test_sequential_uncensorfix_nodes_append_once_each(self):
        merged = self.merged()
        first = self.apply(merged, .25)
        second = self.apply(first, .75)
        source = self.serialization.get_krea2_merge_bypass_info(second)[0]
        self.assertTrue(all(len(v) == 2 for v in source.patches.values()))
        self.assert_output_delta(self.outputs(merged), self.outputs(second))

    def test_save_composition_sees_routed_weights_and_source_bias(self):
        for ratios in ({"first.": 1., "txtfusion.": 0.},
                       {"first.": 1., "txtfusion.layerwise_blocks.0.": 0., "txtfusion.projector.": .4}):
            patched = self.apply(self.merged(ratios=ratios), .75)
            info = self.serialization.get_krea2_merge_bypass_info(patched)
            regular = self.serialization.clone_without_krea2_merge_runtime(patched, info)
            sd, _, _ = self.serialization.compose_krea2_merge_unet_state_dict(regular, info[0], info[1])
            live = self.outputs(patched)
            for key in self.keys:
                local = key.removeprefix(PREFIX)
                expected = F.linear(self.x, sd[local], sd[local[:-6] + "bias"])
                torch.testing.assert_close(live[key], expected)

    def test_save_composition_keeps_overlapping_bypass_lora_and_uncensorfix(self):
        merged = self.merged(ratios={"first.": 1., "txtfusion.layerwise_blocks.0.": 0.})
        patched = self.apply(self.add_bypass(merged), .75)
        source, plans, _ = self.serialization.get_krea2_merge_bypass_info(patched)
        components = {key: [(Adapter((), (up * 2, down, a, None, None, None)), .35)]
                      for key, up, down, a in self.factors}
        # Same split as Donut's save path: main components stay on main;
        # components on runtime-swapped targets are materialized on source.
        regular = self.serialization.clone_without_krea2_merge_runtime(patched)
        regular = self.serialization.clone_with_regular_components(regular, components)
        source = self.serialization.clone_with_regular_components(
            source, components, {key for _, key, _ in plans})
        sd, _, _ = self.serialization.compose_krea2_merge_unet_state_dict(regular, source, plans)
        live = self.outputs(patched)
        for key in self.keys:
            local = key.removeprefix(PREFIX)
            expected = F.linear(self.x, sd[local], sd[local[:-6] + "bias"])
            torch.testing.assert_close(live[key], expected)

    def test_other_additional_models_and_attachments_are_preserved(self):
        merged = self.merged()
        other, _ = self.models()
        merged.set_additional_models("other", [other])
        merged.set_attachments("keep_me", "retained")
        patched = self.apply(merged)
        self.assertIs(patched.additional_models["other"][0].model, other.model)
        self.assertEqual(patched.attachments["keep_me"], "retained")
        self.assertEqual(merged.additional_models["other"], [other])

    def test_diagnostics_report_main_and_bypass_target_counts(self):
        result = self.node.DonutKrea2FusionControl().apply(
            model=self.merged(), conditioning_in_1=object(),
            compatibility_preset="UncensorFix", tap_strength=.75)
        for text in ("uncensorfix_source=embedded", "uncensorfix_targets=33",
                     "uncensorfix_bypass_source_targets=33", "uncensorfix_model_targets=0",
                     "uncensorfix_strength=0.75", "external_files_loaded=none"):
            self.assertIn(text, result[-1])

    def test_missing_source_is_an_error_not_a_main_model_fallback(self):
        merged = self.merged()
        merged.remove_additional_models(self.serialization.KREA2_MERGE_SOURCE_KEY)
        with self.assertRaisesRegex(RuntimeError, "source, found 0"):
            self.apply(merged)

    def test_missing_plan_is_an_error_even_with_falsey_injection_list(self):
        merged = self.merged()
        merged.attachments.clear()
        with self.assertRaisesRegex(RuntimeError, "metadata is missing"):
            self.apply(merged)

    def test_orphaned_plan_is_not_routed_to_an_inactive_source(self):
        merged = self.merged()
        merged.injections.clear()
        with self.assertRaisesRegex(RuntimeError, "without their runtime injection"):
            self.apply(merged)

    def test_missing_source_target_is_rejected_before_any_clone(self):
        merged = self.merged()
        source = self.serialization.get_krea2_merge_bypass_info(merged)[0]
        with mock.patch.object(source.model, "state_dict", return_value={}):
            with mock.patch.object(merged, "clone", side_effect=AssertionError("clone too soon")):
                with self.assertRaisesRegex(RuntimeError, "merge-bypass model2"):
                    self.apply(merged)

    def test_wrong_source_shape_is_rejected(self):
        merged = self.merged()
        source = self.serialization.get_krea2_merge_bypass_info(merged)[0]
        state = dict(source.model.state_dict())
        state[self.keys[0]] = torch.zeros(1, 1)
        with mock.patch.object(source.model, "state_dict", return_value=state):
            with self.assertRaisesRegex(RuntimeError, "wrong shape on merge-bypass model2"):
                self.apply(merged)

    def test_partial_source_acceptance_is_rejected_and_input_is_unchanged(self):
        merged = self.merged()
        source = self.serialization.get_krea2_merge_bypass_info(merged)[0]
        source.reject = {self.keys[0]}
        with self.assertRaisesRegex(RuntimeError, "model2: 32/33"):
            self.apply(merged)
        self.assertFalse(source.patches)
        self.assertFalse(merged.patches)

    def test_off_preserves_existing_merge_and_bypass_lora_forwards(self):
        upstream = self.add_bypass(self.apply(self.merged(), .75))
        before = self.outputs(upstream)
        result = self.node.DonutKrea2FusionControl().apply(
            model=upstream, conditioning_in_1=object(), compatibility_preset="Off",
            tap_strength=3., fusion_strength=2., projector_strength=2.)
        self.assertIs(result[0], upstream)
        after = self.outputs(result[0])
        for key in self.keys:
            self.assertTrue(torch.equal(before[key], after[key]))

    def test_zero_strength_keeps_identity_and_does_not_inspect_bypass(self):
        merged = self.merged()
        merged.attachments.clear()  # Invalid metadata must not matter at zero.
        with mock.patch.object(self.node, "_uncensorfix_factors", side_effect=AssertionError("decoded")):
            result, count, _ = self.node._apply_uncensorfix(merged, 0.)
        self.assertIs(result, merged)
        self.assertEqual(count, 0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

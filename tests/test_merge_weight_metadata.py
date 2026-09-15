"""CPU regressions for virtual FP8 state keys and Krea2 merge orientation.

Uses real torch modules/FP8 tensors and strict patcher/injection test doubles.
The key-patch contract models ComfyUI 3216c62e, not a full ComfyUI/GPU run.
No private workflows/checkpoints, network access or ComfyUI install required.
"""
import copy
import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "diffusion_model."


def attribute(root, path):
    for part in path.split("."):
        root = getattr(root, part)
    return root


def get_key_weight(root, key):
    # Same load-bearing contract as comfy.model_patcher.get_key_weight:
    # serialized keys are looked up on the live op; retain its converter.
    path, _, name = key.rpartition(".")
    op = attribute(root, path) if path else root
    return getattr(op, name), getattr(op, "set_" + name, None), getattr(op, "convert_" + name, None)


class QuantizedTensor(torch.Tensor):
    """Tensor-subclass layout fixture; storage is real FP8, scale is a sidecar."""
    @staticmethod
    def __new__(cls, value):
        return torch.Tensor._make_subclass(cls, value, False)

    def state_dict(self, prefix):
        return {prefix: self.as_subclass(torch.Tensor).detach(), prefix + "_scale": torch.tensor(.5),
                **{prefix + suffix: torch.tensor(1.) for suffix in getattr(self, "extra", ())}}


class SyntheticLinear(torch.nn.Linear):
    """FP8 storage plus scale exported beside weight, not a module attribute."""
    def __init__(self, value=6.0, extra=()):
        super().__init__(2, 2, bias=True)
        self.quant_format = "float8_e4m3fn"
        self.extra = extra
        self.weight = torch.nn.Parameter(QuantizedTensor(torch.full((2, 2), value * 2).to(torch.float8_e4m3fn)), requires_grad=False)
        self.bias = torch.nn.Parameter(torch.full((2,), value), requires_grad=False)
        self.weight.extra = extra

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        destination.update(self.weight.state_dict(prefix + "weight"))
        destination[prefix + "bias"] = self.bias
        destination[prefix + "comfy_quant"] = torch.tensor(list(b'{"format":"float8_e4m3fn"}'), dtype=torch.uint8)

    def convert_weight(self, weight, **kwargs):
        return weight.float().as_subclass(torch.Tensor) * .5 if isinstance(weight, QuantizedTensor) else weight

    def forward(self, x):
        return torch.nn.functional.linear(x, self.convert_weight(self.weight), self.bias)


def linear(value, synthetic=False):
    if synthetic:
        return SyntheticLinear(value)
    result = torch.nn.Linear(2, 2)
    with torch.no_grad():
        result.weight.fill_(value); result.bias.fill_(value)
    return result


def model(value=2., synthetic=False):
    root = torch.nn.Module(); d = torch.nn.Module(); root.diffusion_model = d
    for name in ("first", "tmlp", "txtmlp", "tproj", "last"):
        setattr(d, name, linear(value, synthetic))
    d.blocks = torch.nn.ModuleList([linear(value, synthetic) for _ in range(28)])
    d.txtfusion = torch.nn.Module()
    d.txtfusion.layerwise_blocks = torch.nn.ModuleList([linear(value, synthetic) for _ in range(2)])
    d.txtfusion.projector = linear(value, synthetic)
    d.txtfusion.refiner_blocks = torch.nn.ModuleList([linear(value, synthetic) for _ in range(2)])
    return root


class Injection:
    def __init__(self, inject, eject): self.inject, self.eject = inject, eject


class Manager:
    def __init__(self): self.adapters = []
    def add_adapter(self, key, adapter, strength): self.adapters.append((key, adapter))
    def get_hook_count(self): return len(self.adapters)
    def create_injections(self, root):
        injections = []
        for key, adapter in self.adapters:
            op = attribute(root, key[:-7]); original = op.forward
            def inject(_, op=op, original=original, adapter=adapter):
                op.forward = lambda x, *a, **kw: adapter.bypass_forward(original, x, *a, **kw)
            def eject(_, op=op, original=original): op.forward = original
            injections.append(Injection(inject, eject))
        return injections


class Patcher:
    def __init__(self, root):
        self.model = root; self.patches = {}; self.backup = {}; self.hook_backup = {}
        self.injections = {}; self.additional = {}; self.attachments = {}; self.load_device = "cpu"
    def clone(self):
        result = copy.copy(self)
        for name in ("patches", "backup", "hook_backup", "injections", "additional", "attachments"):
            setattr(result, name, {k: v.copy() if isinstance(v, list) else v for k, v in getattr(self, name).items()})
        return result
    def model_state_dict(self): return self.model.state_dict()
    def get_key_patches(self, prefix=None):
        result = {}
        for key in self.model_state_dict():
            if prefix is not None and not key.startswith(prefix): continue
            weight, _, convert = get_key_weight(self.model, key)  # Intentionally strict.
            if key in self.backup: weight = self.backup[key].weight
            if key in self.hook_backup: weight = self.hook_backup[key][0]
            result[key] = [(weight, convert or (lambda w, **kw: w))] + self.patches.get(key, [])
        return result
    def add_patches(self, values, strength_patch=1., strength_model=1.):
        accepted = []
        for key, data in values.items():
            if key in self.model_state_dict():
                self.patches.setdefault(key, []).append((strength_patch, data, strength_model, None, None)); accepted.append(key)
        return accepted
    def set_injections(self, key, values): self.injections[key] = values
    def set_additional_models(self, key, values): self.additional[key] = values
    def get_additional_models_with_key(self, key): return self.additional[key]
    def set_attachments(self, key, value): self.attachments[key] = value
    def is_clone(self, other): return self.model is other.model


def materialize(patcher, key):
    value, _, convert = get_key_weight(patcher.model, key)
    value = (convert(value) if convert else value).float()
    for source_strength, data, base_strength, _, _ in patcher.patches.get(key, []):
        source, convert = data[0]
        value = value * base_strength + convert(source).float() * source_strength
    return value


class MergeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        package = types.ModuleType("_donut_metadata_tests"); package.__path__ = [str(ROOT)]
        comfy = types.ModuleType("comfy"); comfy.__path__ = []
        core = types.ModuleType("comfy.model_patcher"); core.get_key_weight = get_key_weight
        quant = types.ModuleType("comfy.quant_ops"); quant.QuantizedTensor = QuantizedTensor
        adapters = types.ModuleType("comfy.weight_adapter")
        adapters.WeightAdapterBase = object; adapters.BypassInjectionManager = Manager
        extensions = types.ModuleType("comfy.patcher_extension"); extensions.PatcherInjection = Injection
        comfy.model_patcher = core; comfy.weight_adapter = adapters; comfy.patcher_extension = extensions; comfy.quant_ops = quant
        cls.modules = patch.dict(sys.modules, {package.__name__: package, "comfy": comfy,
            "comfy.model_patcher": core, "comfy.quant_ops": quant, "comfy.weight_adapter": adapters, "comfy.patcher_extension": extensions})
        cls.modules.start()
        cls.merge = importlib.import_module(package.__name__ + ".DonutModelMergeKrea2")
        cls.weights = cls.merge
        cls.grouped = importlib.import_module(package.__name__ + ".donut_grouped_merge")

    @classmethod
    def tearDownClass(cls): cls.modules.stop()

    def test_01_host_getter_reproduces_reported_error(self):
        source = Patcher(model(synthetic=True))
        with self.assertRaisesRegex(AttributeError, "weight_scale"):
            source.get_key_patches(PREFIX)

    def test_02_recovers_without_fabricating_attributes(self):
        source = Patcher(model(synthetic=True)); op = source.model.diffusion_model.first
        before = source.model_state_dict()
        result = self.weights._get_merge_key_patches(source, PREFIX)
        self.assertEqual(len(result), 76)  # 38 real linear weights + 38 biases.
        self.assertFalse(hasattr(op, "weight_scale")); self.assertFalse(hasattr(op, "comfy_quant"))
        self.assertEqual(set(before), set(source.model_state_dict()))
        self.assertEqual(source.patches, {})
        self.assertNotIn(PREFIX + "first.weight_scale", result)

    def test_03_live_fp8_tensor_and_converter_are_retained(self):
        source = Patcher(model(6., synthetic=True)); op = source.model.diffusion_model.first
        weight, convert = self.weights._get_merge_key_patches(source, PREFIX)[PREFIX + "first.weight"][0]
        self.assertIs(weight, op.weight)
        self.assertEqual(weight.dtype, torch.float8_e4m3fn)
        self.assertIs(convert.__self__, op)
        torch.testing.assert_close(convert(weight), torch.full((2, 2), 6.))

    def test_04_ordinary_host_result_is_untouched(self):
        sentinel = {"host": object()}
        source = types.SimpleNamespace(get_key_patches=lambda prefix: sentinel)
        self.assertIs(self.weights._get_merge_key_patches(source, PREFIX), sentinel)

    def test_05_plain_fp8_without_virtual_metadata_stays_on_host_path(self):
        root = model(); op = root.diffusion_model.first
        op.weight = torch.nn.Parameter(op.weight.to(torch.float8_e4m3fn), requires_grad=False)
        result = self.weights._get_merge_key_patches(Patcher(root), PREFIX)
        self.assertIs(result[PREFIX + "first.weight"][0][0], op.weight)

    def test_06_unknown_missing_parameter_is_not_swallowed(self):
        source = Patcher(model(synthetic=True)); state = source.model_state_dict()
        state[PREFIX + "first.gamma"] = torch.tensor(1.)
        source.model_state_dict = lambda: state
        with self.assertRaisesRegex(AttributeError, "gamma"):
            self.weights._get_merge_key_patches(source, PREFIX)

    def test_07_scale_name_alone_is_not_proof_of_virtual_metadata(self):
        source = Patcher(model()); state = source.model_state_dict()
        state[PREFIX + "first.weight_scale"] = torch.tensor(1.)
        source.model_state_dict = lambda: state
        with self.assertRaisesRegex(AttributeError, "weight_scale"):
            self.weights._get_merge_key_patches(source, PREFIX)

    def test_08_unknown_module_path_is_not_swallowed(self):
        source = Patcher(model(synthetic=True)); state = source.model_state_dict()
        state[PREFIX + "nonexistent.weight"] = torch.tensor(1.)
        source.model_state_dict = lambda: state
        with self.assertRaisesRegex(AttributeError, "nonexistent"):
            self.weights._get_merge_key_patches(source, PREFIX)

    def test_09_explicit_metadata_patches_and_backups_fail_closed(self):
        for field in ("patches", "backup", "hook_backup"):
            with self.subTest(field=field):
                source = Patcher(model(synthetic=True))
                getattr(source, field)[PREFIX + "first.weight_scale"] = object()
                with self.assertRaisesRegex(RuntimeError, "export-only.*metadata"):
                    self.weights._get_merge_key_patches(source, PREFIX)

    def test_10_backup_precedence_and_source_patch_identity(self):
        source = Patcher(model(synthetic=True)); key = PREFIX + "first.weight"
        backup = torch.full((2, 2), 7.); hooked = torch.full((2, 2), 9.)
        source.backup[key] = types.SimpleNamespace(weight=backup)
        opaque_patch = (0.75, object(), 1., None, None); source.patches[key] = [opaque_patch]
        result = self.weights._get_merge_key_patches(source, PREFIX)
        self.assertIs(result[key][0][0], backup); self.assertIs(result[key][1], opaque_patch)
        source.hook_backup[key] = (hooked,)
        result = self.weights._get_merge_key_patches(source, PREFIX)
        self.assertIs(result[key][0][0], hooked)
        self.assertIsNot(result[key], source.patches[key]); self.assertEqual(len(source.patches[key]), 1)

    def test_11_real_scale_attributes_are_not_filtered(self):
        source = Patcher(model(synthetic=True))
        source.model.diffusion_model.first.register_buffer("weight_scale", torch.tensor(.5))
        result = self.weights._get_merge_key_patches(source, PREFIX)
        self.assertIn(PREFIX + "first.weight_scale", result)

    def test_12_additional_tensor_export_scales_are_handled(self):
        root = model(); root.diffusion_model.first = SyntheticLinear(extra=("_scale_2",))
        result = self.weights._get_merge_key_patches(Patcher(root), PREFIX)
        self.assertNotIn(PREFIX + "first.weight_scale_2", result)
        self.assertIn(PREFIX + "first.weight", result)

    def test_13_unrelated_exception_is_not_intercepted(self):
        source = Patcher(model())
        def fail(prefix): raise ValueError("unrelated")
        source.get_key_patches = fail
        with self.assertRaisesRegex(ValueError, "unrelated"):
            self.weights._get_merge_key_patches(source, PREFIX)

    def test_14_prefix_filter_excludes_unrelated_state(self):
        source = Patcher(model(synthetic=True)); state = source.model_state_dict()
        state["unrelated.missing"] = torch.tensor(1.)
        source.model_state_dict = lambda: state
        self.assertTrue(self.weights._get_merge_key_patches(source, PREFIX))

    def recipe(self, source_fp8=False, swapped=False, mode="Comfy patches"):
        fineporn = Patcher(model(2., source_fp8)); turbo = Patcher(model(6.))
        model1, model2 = (turbo, fineporn) if swapped else (fineporn, turbo)
        ratios = {name: (0. if swapped else 1.) for name, spec in self.merge.DonutModelMergeKrea2.INPUT_TYPES()["required"].items() if spec[0] == "FLOAT"}
        # Grouped controls override only their documented component sets.
        output = self.grouped.DonutModelMergeKrea2Grouped().merge_grouped(
            model1=model1, model2=model2, ratio_mode="Grouped", body_ratio=0. if swapped else 1.,
            fusion_ratio=1. if swapped else 0., execution_mode=mode, **ratios)[0]
        return fineporn, turbo, output

    def test_15_regular_recipe_for_both_model_orders(self):
        for fp8 in (False, True):
            for swapped in (False, True):
                with self.subTest(fp8=fp8, swapped=swapped):
                    _, _, output = self.recipe(fp8, swapped)
                    for name in output.model.state_dict():
                        if not name.endswith((".weight", ".bias")): continue
                        expected = 6. if ".txtfusion." in name else 2.
                        torch.testing.assert_close(materialize(output, name), torch.full_like(materialize(output, name), expected))

    def test_16_experimental_recipe_hooks_and_remaining_patches(self):
        for swapped in (False, True):
            with self.subTest(swapped=swapped):
                fineporn, turbo, output = self.recipe(True, swapped, "Experimental bypass")
                plans = next(iter(output.attachments.values()))
                self.assertEqual(len(plans), 33 if swapped else 5)
                self.assertTrue(all((".txtfusion." not in k) == swapped for _, k, _ in plans))
                injection = output.injections[self.merge._INJECTION_KEY][0]
                with torch.no_grad():
                    injection.inject(output)
                    try:
                        for path, _, _ in plans:
                            wanted = 2. if swapped else 6.
                            actual = attribute(output.model, path)(torch.ones(1, 2))
                            torch.testing.assert_close(actual, torch.full((1, 2), wanted * 3))
                    finally: injection.eject(output)
                self.assertEqual(fineporn.patches, {}); self.assertEqual(turbo.patches, {})

    def test_17_partial_fp8_source_blend_uses_converted_weight(self):
        first, second = Patcher(model(2.)), Patcher(model(6., True))
        for mode in ("Comfy patches", "Experimental bypass"):
            with self.subTest(mode=mode):
                result = self.merge.DonutModelMergeKrea2().merge(first, second, mode, **{"first.": .25})[0]
                self.assertFalse(result.injections)
                torch.testing.assert_close(materialize(result, PREFIX + "blocks.0.weight"), torch.full((2, 2), 5.))

    def test_18_unavailable_bypass_uses_same_safe_regular_path(self):
        with patch.object(self.merge, "_BYPASS_MANAGER", None):
            _, _, output = self.recipe(True, True, "Experimental bypass")
            torch.testing.assert_close(materialize(output, PREFIX + "tmlp.weight"), torch.full((2, 2), 2.))

    def test_19_swapping_inputs_without_inverting_ratios_reverses_recipe(self):
        first, second = Patcher(model(6.)), Patcher(model(2., True))
        result = self.grouped.DonutModelMergeKrea2Grouped().merge_grouped(
            model1=first, model2=second, ratio_mode="Grouped", body_ratio=1., fusion_ratio=0.,
            **{"tmlp.": 1., "txtmlp.": 1., "tproj.": 1.})[0]
        torch.testing.assert_close(materialize(result, PREFIX + "blocks.0.weight"), torch.full((2, 2), 6.))
        torch.testing.assert_close(materialize(result, PREFIX + "txtfusion.projector.weight"), torch.full((2, 2), 2.))

    def test_20_same_named_error_on_unrelated_owner_is_not_swallowed(self):
        source = Patcher(model(synthetic=True))
        error = AttributeError("unrelated", name="weight_scale", obj=object())
        def fail(prefix): raise error
        source.get_key_patches = fail
        with self.assertRaises(AttributeError) as caught:
            self.weights._get_merge_key_patches(source, PREFIX)
        self.assertIs(caught.exception, error)


    def test_21_previous_getter_path_crashes_in_both_modes(self):
        with patch.object(self.merge, "_get_merge_key_patches", lambda p, prefix=PREFIX: p.get_key_patches(prefix)):
            for mode in ("Comfy patches", "Experimental bypass"):
                with self.subTest(mode=mode), self.assertRaisesRegex(AttributeError, "weight_scale"):
                    self.merge.DonutModelMergeKrea2().merge(Patcher(model()), Patcher(model(synthetic=True)), mode, **{"first.": 0.})

    def test_22_standalone_module_import_still_works(self):
        # Existing repository tests import this file without a package parent.
        spec = importlib.util.spec_from_file_location("standalone_merge_metadata_test", ROOT / "DonutModelMergeKrea2.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.assertIn("DonutModelMergeKrea2", module.NODE_CLASS_MAPPINGS)


if __name__ == "__main__": unittest.main()

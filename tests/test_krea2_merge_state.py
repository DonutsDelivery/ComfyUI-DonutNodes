"""CPU regression tests for Krea2 merge key collection and operand direction.

Real torch FP8/BF16 storage and linear arithmetic; a small patcher/packed-weight
adapter models the relevant Comfy 3216c62e contracts. Not a ComfyUI/GPU test.
See comfy/model_patcher.py:get_key_patches and comfy/ops.py's mixed_precision_ops.
"""
from collections import OrderedDict
from contextlib import nullcontext
from copy import copy
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "diffusion_model."


class PackedWeight:
    """Test stand-in for Comfy's runtime QuantizedTensor (not a new format)."""
    def __init__(self, data=2.0, scale=3.0):
        self.data = torch.full((2, 2), data, dtype=torch.float8_e4m3fn)
        self.scale = torch.tensor(scale, dtype=torch.float32)
        self.shape = self.data.shape

    def state_dict(self, name):
        return {name: self.data, name + "_scale": self.scale}

    def dequantize(self):
        return self.data.float() * self.scale


class Linear(torch.nn.Module):
    # Matches the documented conservative comfy.ops.Linear bypass contract.
    __module__ = "comfy.ops"

    def __init__(self, value, packed=False, dtype=torch.bfloat16):
        super().__init__()
        self.in_features = self.out_features = 2
        self.weight = PackedWeight(value) if packed else torch.full((2, 2), value, dtype=dtype)
        self.bias = torch.full((2,), value, dtype=torch.float32)
        self.quant_format = "float8_e4m3fn" if packed else None
        self.extra_export = {}

    def state_dict(self, *args, destination=None, prefix="", **kwargs):
        result = destination if destination is not None else OrderedDict()
        result[prefix + "bias"] = self.bias
        if isinstance(self.weight, PackedWeight):
            result.update(self.weight.state_dict(prefix + "weight"))
            result[prefix + "comfy_quant"] = torch.tensor([1], dtype=torch.uint8)
        else:
            result[prefix + "weight"] = self.weight
        if hasattr(self, "weight_scale"):
            result[prefix + "weight_scale"] = self.weight_scale
        result.update({prefix + key: value for key, value in self.extra_export.items()})
        return result

    def convert_weight(self, weight, **kwargs):
        return weight.dequantize() if isinstance(weight, PackedWeight) else weight

    def _forward(self, x, weight, bias):
        return torch.nn.functional.linear(x, weight, bias)

    def forward(self, x):
        return self._forward(x, self.convert_weight(self.weight).float(), self.bias)


def module_at(root, path):
    for part in path.split("."):
        root = getattr(root, part)
    return root


class Patcher:
    def __init__(self, model):
        self.model = model
        self.patches, self.backup, self.hook_backup = {}, {}, {}
        self.injections, self.additional_models, self.attachments = {}, {}, {}
        self.load_device = "cpu"
        self.patches_uuid = "original"
        self.clones = []
        self.collect_calls = 0
        self.is_injected = False

    def clone(self):
        result = copy(self)
        for name in ("patches", "injections", "additional_models", "attachments"):
            setattr(result, name, {k: copy(v) for k, v in getattr(self, name).items()})
        self.clones.append(result)
        return result

    def is_clone(self, other):
        return self.model is other.model

    def model_state_dict(self, filter_prefix=None):
        return OrderedDict((k, v) for k, v in self.model.state_dict().items()
                           if filter_prefix is None or k.startswith(filter_prefix))

    def get_key_patches(self, filter_prefix=None):
        # Importantly: do NOT use the serialized raw tensors as patch weights.
        # Comfy enumerates state_dict but reads live attributes plus converters.
        self.collect_calls += 1
        result = {}
        for key in self.model_state_dict():
            if filter_prefix is not None and not key.startswith(filter_prefix):
                continue
            path, field = key.rsplit(".", 1)
            op = module_at(self.model, path)
            value = getattr(op, field)  # Reproduces the reported weight_scale error.
            converter = getattr(op, "convert_" + field, None)
            if key in self.backup:
                value = self.backup[key].weight
            if key in self.hook_backup:
                value = self.hook_backup[key][0]
            if converter is None:
                converter = lambda x, **kwargs: x
            result[key] = [(value, converter)] + self.patches.get(key, [])
        return result

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        accepted = []
        state = self.model.state_dict()
        for key, value in patches.items():
            if key in state:
                self.patches.setdefault(key, []).append((strength_patch, value, strength_model, None, None))
                accepted.append(key)
        return accepted

    def set_injections(self, key, value):
        self.injections[key] = value

    def set_additional_models(self, key, value):
        self.additional_models[key] = value

    def get_additional_models_with_key(self, key):
        return self.additional_models.get(key, [])

    def set_attachments(self, key, value):
        self.attachments[key] = value


class Injection:
    def __init__(self, inject, eject):
        self.inject, self.eject = inject, eject


class Manager:
    def __init__(self):
        self.adapters = []

    def add_adapter(self, key, adapter, strength):
        self.adapters.append((key, adapter))

    def get_hook_count(self):
        return len(self.adapters)

    def create_injections(self, root):
        injections = []
        for key, adapter in self.adapters:
            op = module_at(root, key[:-7])
            original = op.forward
            def inject(_patcher, op=op, original=original, adapter=adapter):
                op.forward = lambda x, *a, **kw: adapter.bypass_forward(original, x, *a, **kw)
            def eject(_patcher, op=op, original=original):
                op.forward = original
            injections.append(Injection(inject, eject))
        return injections


def make_model(value, packed=False, dtype=torch.bfloat16):
    model = torch.nn.Module()
    dm = model.diffusion_model = torch.nn.Module()
    dm.first = Linear(value, packed, dtype)
    dm.tmlp = Linear(value, packed, dtype)
    dm.txtmlp = Linear(value, packed, dtype)
    dm.tproj = Linear(value, packed, dtype)
    dm.blocks = torch.nn.ModuleList([Linear(value, packed, dtype) for _ in range(28)])
    dm.txtfusion = torch.nn.Module()
    dm.txtfusion.layerwise_blocks = torch.nn.ModuleList([Linear(value, packed, dtype) for _ in range(2)])
    dm.txtfusion.projector = Linear(value, packed, dtype)
    dm.txtfusion.refiner_blocks = torch.nn.ModuleList([Linear(value, packed, dtype) for _ in range(2)])
    dm.last = Linear(value, packed, dtype)
    model.unrelated = Linear(99)
    return Patcher(model)


def evaluate_patches(base, entries):
    value = base.float().clone()
    for strength_patch, incoming, strength_model, offset, function in entries:
        weight, converter = incoming[0]
        incoming_value = converter(weight).float()
        incoming_value = evaluate_patches(incoming_value, incoming[1:])
        value = value * strength_model + incoming_value * strength_patch
    return value


def effective_value(patcher, key):
    op = module_at(patcher.model, key.rsplit(".", 1)[0])
    field = key.rsplit(".", 1)[1]
    value = getattr(op, field)
    converter = getattr(op, "convert_" + field, lambda value: value)
    return evaluate_patches(converter(value), patcher.patches.get(key, []))


def load_modules():
    comfy = types.ModuleType("comfy"); comfy.__path__ = []
    adapter = types.ModuleType("comfy.weight_adapter")
    adapter.WeightAdapterBase = object; adapter.BypassInjectionManager = Manager
    ext = types.ModuleType("comfy.patcher_extension"); ext.PatcherInjection = Injection
    quant = types.ModuleType("comfy.quant_ops"); quant.QuantizedTensor = PackedWeight
    comfy.weight_adapter = adapter; comfy.patcher_extension = ext; comfy.quant_ops = quant
    package = types.ModuleType("_donut_merge_state_tests"); package.__path__ = [str(ROOT)]
    env = {"comfy": comfy, "comfy.weight_adapter": adapter, "comfy.patcher_extension": ext,
           "comfy.quant_ops": quant, package.__name__: package}
    with patch.dict(sys.modules, env):
        for filename in ("DonutModelMergeKrea2", "donut_grouped_merge"):
            name = package.__name__ + "." + filename
            spec = importlib.util.spec_from_file_location(name, ROOT / (filename + ".py"))
            mod = importlib.util.module_from_spec(spec); env[name] = mod; sys.modules[name] = mod
            spec.loader.exec_module(mod)
    return env, env[package.__name__ + ".DonutModelMergeKrea2"], mod


ENV, MERGE, GROUPED = load_modules()


class MergeStateTests(unittest.TestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules, ENV); self.modules.start()
        self.addCleanup(self.modules.stop)
        self.weight_key = PREFIX + "first.weight"

    def test_native_collection_reproduces_exact_missing_weight_scale(self):
        model = make_model(2, packed=True)
        with self.assertRaisesRegex(AttributeError, "'Linear' object has no attribute 'weight_scale'"):
            model.get_key_patches(PREFIX)

    def test_packed_live_weight_and_nonunit_scale_are_retained(self):
        model = make_model(2, packed=True)
        op = model.model.diffusion_model.first
        records = MERGE._get_merge_key_patches(model)
        weight, convert = records[self.weight_key][0]
        self.assertIs(weight, op.weight)
        torch.testing.assert_close(convert(weight), torch.full((2, 2), 6.0))
        self.assertEqual(op.weight.scale.item(), 3.0)
        self.assertFalse(hasattr(op, "weight_scale"))

    def test_only_virtual_keys_are_excluded_not_real_parameters(self):
        model = make_model(2, packed=True)
        keys = set(model.model_state_dict(PREFIX))
        records = MERGE._get_merge_key_patches(model)
        missing = keys - records.keys()
        self.assertEqual(missing, {key for key in keys if key.endswith((".weight_scale", ".comfy_quant"))})
        self.assertEqual(len(records), 76)
        self.assertTrue(all(key.startswith(PREFIX) for key in records))
        self.assertEqual(keys, set(model.model_state_dict(PREFIX)))

    def test_real_scale_attribute_is_not_silently_removed(self):
        model = make_model(2, packed=True)
        model.model.diffusion_model.first.weight_scale = torch.tensor(7.0)
        records = MERGE._get_merge_key_patches(model)
        self.assertIn(PREFIX + "first.weight_scale", records)
        self.assertEqual(records[PREFIX + "first.weight_scale"][0][0].item(), 7.0)

    def test_plain_fp8_uses_unchanged_native_path(self):
        model = make_model(2, dtype=torch.float8_e4m3fn)
        records = MERGE._get_merge_key_patches(model)
        self.assertEqual(model.collect_calls, 1)
        self.assertEqual(model.clones, [])
        self.assertEqual(records[self.weight_key][0][0].dtype, torch.float8_e4m3fn)

    def test_plain_bf16_uses_unchanged_native_path(self):
        model = make_model(2)
        records = MERGE._get_merge_key_patches(model)
        self.assertEqual(records[self.weight_key][0][0].dtype, torch.bfloat16)
        self.assertEqual(model.clones, [])

    def test_source_patch_order_and_backup_precedence_are_native(self):
        model = make_model(2, packed=True)
        bk = PackedWeight(3, 4); hook = PackedWeight(4, 5)
        entry1 = (0.1, [(torch.ones(2, 2), lambda w: w)], 1, None, None)
        entry2 = (0.2, [(torch.ones(2, 2), lambda w: w)], 1, None, None)
        model.backup[self.weight_key] = types.SimpleNamespace(weight=bk)
        model.hook_backup[self.weight_key] = (hook,)
        model.patches[self.weight_key] = [entry1, entry2]
        records = MERGE._get_merge_key_patches(model)
        self.assertIs(records[self.weight_key][0][0], hook)
        self.assertIs(records[self.weight_key][1], entry1)
        self.assertIs(records[self.weight_key][2], entry2)
        del model.hook_backup[self.weight_key]
        self.assertIs(MERGE._get_merge_key_patches(model)[self.weight_key][0][0], bk)

    def test_input_and_reader_have_no_persistent_method_override(self):
        model = make_model(2, packed=True)
        before = dict(model.__dict__)
        MERGE._get_merge_key_patches(model)
        self.assertNotIn("model_state_dict", model.__dict__)
        self.assertTrue(all("model_state_dict" not in c.__dict__ for c in model.clones))
        for name in ("model", "patches", "backup", "hook_backup", "attachments", "injections"):
            self.assertIs(getattr(model, name), before[name])

    def test_unknown_missing_attribute_still_raises(self):
        model = make_model(2)
        model.model.diffusion_model.first.extra_export["missing_tensor"] = torch.tensor(1)
        with self.assertRaisesRegex(AttributeError, "missing_tensor"):
            MERGE._get_merge_key_patches(model)

    def test_unknown_error_is_not_hidden_by_virtual_key_repair(self):
        model = make_model(2, packed=True)
        model.model.diffusion_model.first.extra_export["missing_tensor"] = torch.tensor(1)
        with self.assertRaisesRegex(AttributeError, "missing_tensor"):
            MERGE._get_merge_key_patches(model)
        self.assertTrue(all("model_state_dict" not in c.__dict__ for c in model.clones))

    def test_virtual_patch_and_backup_conflicts_fail_closed(self):
        for name in ("patches", "backup", "hook_backup"):
            with self.subTest(field=name):
                model = make_model(2, packed=True)
                getattr(model, name)[PREFIX + "first.weight_scale"] = [1]
                with self.assertRaisesRegex(RuntimeError, "serialization-only keys"):
                    MERGE._get_merge_key_patches(model)

    def test_missing_quant_api_preserves_original_exception(self):
        model = make_model(2, packed=True)
        with patch.dict(sys.modules, {"comfy.quant_ops": None}):
            with self.assertRaisesRegex(AttributeError, "weight_scale"):
                MERGE._get_merge_key_patches(model)

    def test_reader_must_be_an_independent_clone(self):
        model = make_model(2, packed=True)
        model.clone = lambda: model
        with self.assertRaisesRegex(RuntimeError, "independent patcher clone"):
            MERGE._get_merge_key_patches(model)

    def test_regular_blend_direction_both_input_orders(self):
        for reverse in (False, True):
            for ratio in (0, 0.25, 1):
                with self.subTest(reverse=reverse, ratio=ratio):
                    a, b = make_model(2, packed=True), make_model(10)
                    if reverse: a, b = b, a
                    expected = effective_value(a, self.weight_key) * ratio + effective_value(b, self.weight_key) * (1-ratio)
                    result = MERGE.DonutModelMergeKrea2().merge(a, b, **{"first.": ratio})[0]
                    torch.testing.assert_close(effective_value(result, self.weight_key), expected)
                    self.assertEqual(a.patches, {}); self.assertEqual(b.patches, {})

    def test_grouped_recipe_keeps_every_non_txtfusion_component(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                fine, turbo = make_model(2, packed=True), make_model(10)
                a, b = (turbo, fine) if reverse else (fine, turbo)
                body, fusion = (0, 1) if reverse else (1, 0)
                result = GROUPED.DonutModelMergeKrea2Grouped().merge_grouped(
                    model1=a, model2=b, ratio_mode="Grouped", body_ratio=body, fusion_ratio=fusion,
                    **{"tmlp.": body, "txtmlp.": body, "tproj.": body})[0]
                for key in MERGE._get_merge_key_patches(fine):
                    source = turbo if key.startswith(PREFIX + "txtfusion.") else fine
                    torch.testing.assert_close(effective_value(result, key), effective_value(source, key))

    def test_experimental_swap_uses_original_scaled_source_forward(self):
        base, source = make_model(10), make_model(2, packed=True)
        result = MERGE.DonutModelMergeKrea2().merge(base, source, execution_mode="Experimental bypass", **{"first.": 0})[0]
        self.assertEqual(result.patches, {})
        retained = result.additional_models[MERGE._SOURCE_MODELS_KEY][0]
        self.assertIs(retained.model, source.model)
        injections = list(result.injections[MERGE._INJECTION_KEY])
        x = torch.tensor([[1.0, 2.0]])
        expected = source.model.diffusion_model.first(x)
        for injection in injections: injection.inject(result)
        try:
            torch.testing.assert_close(result.model.diffusion_model.first(x), expected)
            self.assertEqual(source.model.diffusion_model.first.weight.scale.item(), 3.0)
            self.assertFalse(hasattr(source.model.diffusion_model.first, "weight_scale"))
        finally:
            for injection in reversed(injections): injection.eject(result)
        torch.testing.assert_close(base.model.diffusion_model.first(x), torch.tensor([[40.0, 40.0]]))

    def test_experimental_partial_ratios_remain_materialized(self):
        base, source = make_model(10), make_model(2, packed=True)
        result = MERGE.DonutModelMergeKrea2().merge(base, source, execution_mode="Experimental bypass", **{"first.": 0.25})[0]
        self.assertEqual(result.injections, {})
        self.assertEqual(len(result.patches), 76)
        torch.testing.assert_close(effective_value(result, self.weight_key), torch.full((2, 2), 7.0))

    def test_experimental_recipe_both_input_orders_preserves_fusion_and_body(self):
        for reverse in (False, True):
            fine, turbo = make_model(2, packed=True), make_model(10)
            a, b = (turbo, fine) if reverse else (fine, turbo)
            body, fusion = (0, 1) if reverse else (1, 0)
            result = GROUPED.DonutModelMergeKrea2Grouped().merge_grouped(
                model1=a, model2=b, execution_mode="Experimental bypass", ratio_mode="Grouped",
                body_ratio=body, fusion_ratio=fusion, **{"tmlp.": body, "txtmlp.": body, "tproj.": body})[0]
            injection = list(result.injections[MERGE._INJECTION_KEY])[0]
            x = torch.tensor([[1.0, 2.0]])
            # Capture both donors before shared base forwards are temporarily wrapped.
            expected = {path: module_at(turbo.model if path.startswith(PREFIX+'txtfusion.') else fine.model, path)(x)
                        for path, module in fine.model.named_modules() if path.startswith(PREFIX) and isinstance(module, Linear)}
            injection.inject(result)
            try:
                for path, value in expected.items():
                    torch.testing.assert_close(module_at(result.model, path)(x), value)
            finally:
                injection.eject(result)

    def test_fallback_without_bypass_api_uses_repaired_regular_path(self):
        base, source = make_model(10), make_model(2, packed=True)
        with patch.object(MERGE, "_BYPASS_MANAGER", None):
            result = MERGE.DonutModelMergeKrea2().merge(base, source, execution_mode="Experimental bypass", **{"first.": 0})[0]
        self.assertEqual(result.injections, {})
        torch.testing.assert_close(effective_value(result, self.weight_key), torch.full((2, 2), 6.0))

    def test_grouped_schema_order_defaults_unchanged_and_tooltips_explicit(self):
        schema = GROUPED.DonutModelMergeKrea2Grouped.INPUT_TYPES()
        self.assertEqual(list(schema['optional']), ['execution_mode', 'ratio_mode', 'body_ratio', 'fusion_ratio'])
        for name in ('body_ratio', 'fusion_ratio'):
            self.assertEqual(schema['optional'][name][1]['default'], 1.0)
            self.assertIn('1 keeps model1; 0 uses model2', schema['optional'][name][1]['tooltip'])

    def test_native_override_is_not_lost_on_reader_clone(self):
        model = make_model(2, packed=True)
        original = model.model_state_dict
        override = lambda prefix=None: original(prefix)
        model.model_state_dict = override
        MERGE._get_merge_key_patches(model)
        self.assertIs(model.model_state_dict, override)
        self.assertTrue(all(c.model_state_dict is override for c in model.clones))


if __name__ == '__main__':
    unittest.main()

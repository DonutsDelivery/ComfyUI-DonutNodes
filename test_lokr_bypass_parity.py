"""CPU hook, stacking, and save parity for LoRA/LoCon, LoHa, and LoKr.

Run with python -m unittest test_lokr_bypass_parity -v.
Set COMFYUI_ROOT to test a different ComfyUI checkout.
"""
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from test_safe_lora_stack import _load_module


class LoKrBypassParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(os.environ.get('COMFYUI_ROOT', '/home/user/Programs/ComfyUI-new/ComfyUI'))
        source = root / 'comfy/weight_adapter'
        if not (source / 'lokr.py').exists():
            raise unittest.SkipTest('Set COMFYUI_ROOT to an installed ComfyUI checkout')
        comfy = types.ModuleType('comfy')
        management = types.ModuleType('comfy.model_management')
        management.cast_to_device = lambda value, device, dtype: value.to(device=device, dtype=dtype)
        management.get_torch_device = lambda: torch.device("cpu")
        comfy.model_management = management
        extension = types.ModuleType("comfy.patcher_extension")
        extension.PatcherInjection = lambda **kwargs: types.SimpleNamespace(**kwargs)
        package = types.ModuleType('_donut_native_adapter_test')
        package.__path__ = [str(source)]
        with patch.dict(sys.modules, {'comfy': comfy, 'comfy.model_management': management,
                                     package.__name__: package, 'comfy.patcher_extension': extension}):
            for name in ('base', 'lokr', 'lora', 'loha', 'bypass'):
                spec = importlib.util.spec_from_file_location(package.__name__ + '.' + name, source / (name + '.py'))
                loaded = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = loaded
                spec.loader.exec_module(loaded)
                for export in ('WeightAdapterBase', 'LoKrAdapter', 'LoRAAdapter', 'LoHaAdapter',
                               'BypassInjectionManager', 'BypassForwardHook'):
                    if hasattr(loaded, export):
                        setattr(package, export, getattr(loaded, export))
            cls.adapter_type = package.LoKrAdapter
        cls.native = package
        cls.safe = _load_module(package)
        cls.safe.torch = torch
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.addClassCleanup(torch.set_num_threads, previous_threads)

    def check_parity(self, weights):
        adapter = self.adapter_type(set(), weights)
        w1, w2, _, a, b, c, d, *_ = weights
        shape1 = w1.shape if w1 is not None else (a.shape[0], b.shape[1])
        shape2 = w2.shape if w2 is not None else (c.shape[0], d.shape[1])
        layer = torch.nn.Linear(shape1[1] * shape2[1], shape1[0] * shape2[0], bias=False)
        self.assertIsNone(self.safe._bypass_compatibility_error(adapter, layer))
        adapter = self.safe._LinearLoKrBypassAdapter(adapter)
        for strength in (0.0, -0.65, 1.7):
            adapter.multiplier = strength
            delta = adapter.calculate_weight(torch.zeros_like(layer.weight), 'test', strength, 1.0, None, lambda x: x)
            for batch in ((), (2,), (2, 3)):
                x = torch.randn(*batch, layer.in_features)
                torch.testing.assert_close(adapter.h(x, None), F.linear(x, delta), atol=2e-4, rtol=2e-4)

    def test_direct_and_decomposed_parity(self):
        torch.manual_seed(71)
        for first, second in ((False, False), (True, False), (False, True), (True, True)):
            for alpha in (None, 0.0, 0.7, 4.0):
                with self.subTest(first=first, second=second, alpha=alpha):
                    self.check_parity((None if first else torch.randn(3, 2),
                                       None if second else torch.randn(5, 4), alpha,
                                       torch.randn(3, 2) if first else None,
                                       torch.randn(2, 2) if first else None,
                                       torch.randn(5, 2) if second else None,
                                       torch.randn(2, 4) if second else None, None, None))

    def test_lokr_bypass_runs_on_retained_merge_source(self):
        from test_model_merge_krea2 import module as merge
        from test_uncensorfix_merge_bypass import Patcher, make_root
        body = 'diffusion_model.blocks.0.proj.weight'
        fusion = 'diffusion_model.txtfusion.layerwise_blocks.0.attn.wq.weight'
        base = Patcher(make_root([body, fusion], .2))
        source = Patcher(make_root([body, fusion], .8))
        merged, = merge.DonutModelMergeKrea2().merge(base, source,
            'Experimental bypass', **{'first.': 1., 'txtfusion.': 0.})
        adapter = self.adapter_type(set(), (torch.ones(1, 1), None, 1., None, None,
            torch.full((2, 1), .3), torch.full((1, 3), .2), None, None))
        x = torch.tensor([[.3, -.7, 1.1]])
        with merged.activate():
            before = {key: merged.model.get_submodule(key[:-7])(x).clone() for key in (body, fusion)}
        for strength in (1., -.5, 2.):
            patched = self.safe._apply_bypass_components(merged,
                {key: [(adapter, strength)] for key in (body, fusion)})
            active_source = patched.get_additional_models_with_key(merge._SOURCE_MODELS_KEY)[0]
            self.assertIn('donut_bypass_lora', active_source.injections)
            self.assertIn('donut_bypass_lora', patched.injections)
            self.assertFalse(active_source.patches)
            self.assertFalse(patched.patches)
            self.assertFalse(source.injections)
            self.assertFalse(merged.clone_has_same_weights(patched))
            delta = adapter.calculate_weight(torch.zeros(2, 3), fusion, strength, 1., None, lambda v: v)
            with patched.activate():
                for key in (body, fusion):
                    torch.testing.assert_close(patched.model.get_submodule(key[:-7])(x),
                        before[key] + F.linear(x, delta))
            with merged.activate():
                for key in (body, fusion):
                    torch.testing.assert_close(merged.model.get_submodule(key[:-7])(x), before[key])

    def test_downstream_edit_adapter_preserves_existing_bypass(self):
        from test_uncensorfix_merge_bypass import Patcher, make_root
        import donut_bypass_materialization as metadata
        key = 'diffusion_model.blocks.0.proj.weight'
        base = Patcher(make_root([key], .2))
        adapter = self.adapter_type(set(), (torch.ones(1, 1), torch.full((2, 3), .3),
            1., None, None, None, None, None, None))
        first = self.safe._apply_bypass_components(base, {key: [(adapter, .5)]})
        helper_name = self.safe.__package__ + '.donut_bypass_materialization'
        with patch.dict(sys.modules, {helper_name: metadata}):
            second = self.safe._apply_bypass_components(first, {key: [(adapter, .7)]})
        self.assertFalse(second.patches)
        self.assertIn('donut_bypass_lora', second.injections)
        self.assertEqual(len(metadata.get_bypass_components(second)[key]), 2)
        self.assertEqual(len(metadata.get_bypass_components(first)[key]), 1)
        x = torch.tensor([[.3, -.7, 1.1]])
        expected = base.model.get_submodule(key[:-7])(x).detach().clone() + F.linear(x, torch.full((2, 3), .36))
        with second.activate():
            torch.testing.assert_close(second.model.get_submodule(key[:-7])(x), expected)

    def test_lokr_coverage_counts_executed_components_only(self):
        adapter = self.adapter_type(set(), (torch.ones(1, 1), torch.ones(2, 3),
            None, None, None, None, None, None, None))
        first = self.safe._LinearLoKrBypassAdapter(adapter)
        second = self.safe._LinearLoKrBypassAdapter(adapter)
        stacked = self.safe._CompositeBypassAdapter([(first, 1.)])
        manager = types.SimpleNamespace(adapters={'body': (stacked, 1.), 'fusion': (second, 1.)})
        pending = self.safe._trace_lokr_calls(manager)
        self.assertEqual(len(pending), 2)
        x = torch.ones(1, 3)
        stacked.h(x, None)
        self.assertEqual(pending, {('fusion', 0)})
        stacked.h(x, None)
        self.assertEqual(len(pending), 1)
        second.h(x, None)
        self.assertFalse(pending)

    def test_unsupported_decompositions_remain_regular(self):
        valid = [torch.randn(3, 2), None, 4.0, None, None, torch.randn(5, 2), torch.randn(2, 4), None, None]
        for change, reason in (({6: None}, 'two matrix'), ({6: torch.randn(3, 4)}, 'incompatible'),
                               ({7: torch.randn(2, 2, 1, 1)}, 'Tucker'), ({8: torch.ones(15)}, 'DoRA')):
            weights = valid.copy()
            for index, value in change.items():
                weights[index] = value
            adapter = self.adapter_type(set(), weights)
            self.assertIn(reason, self.safe._bypass_compatibility_error(adapter, torch.nn.Linear(8, 15)))


    def test_unequal_lokr_ranks(self):
        for alpha in (None, 0.0, 0.8, 4.0):
            self.check_parity((None, None, alpha, torch.randn(3, 3), torch.randn(3, 2),
                               torch.randn(5, 2), torch.randn(2, 4), None, None))

    def assert_hook_parity(self, layer, components, x):
        root = torch.nn.Module()
        root.layer = layer
        for adapter, _ in components:
            self.assertIsNone(self.safe._bypass_compatibility_error(adapter, layer))
        manager = self.native.BypassInjectionManager()
        self.safe._register_bypass_adapters(manager, {'layer.weight': components})
        injections = manager.create_injections(root)
        weight = layer.weight.detach().clone()
        for adapter, strength in components:
            weight = adapter.calculate_weight(weight, 'layer.weight', strength, 1.0, None, lambda x: x)
        if isinstance(layer, torch.nn.Linear):
            expected = F.linear(x, weight, layer.bias)
        else:
            expected = layer._conv_forward(x, weight, layer.bias)
        original = layer(x)
        try:
            injections[0].inject(None)
            torch.testing.assert_close(layer(x), expected, atol=3e-4, rtol=3e-4)
            # Registered adapters must still materialize correctly for saves.
            registered, strength = manager.adapters['layer']
            parts = getattr(registered, 'components', [(registered, strength)])
            restored = layer.weight.detach().clone()
            for adapter, strength in parts:
                restored = adapter.calculate_weight(restored, 'layer.weight', strength, 1.0, None, lambda x: x)
            torch.testing.assert_close(restored, weight)
        finally:
            injections[0].eject(None)
        torch.testing.assert_close(layer(x), original)

    def test_loha_and_mixed_linear_stack(self):
        layer = torch.nn.Linear(8, 15)
        for alpha in (None, 0.0, 0.7, 4.0):
            loha = self.native.LoHaAdapter(set(), (torch.randn(15, 2), torch.randn(2, 8), alpha,
                                                  torch.randn(15, 3), torch.randn(3, 8), None, None, None))
            lora = self.native.LoRAAdapter(set(), (torch.randn(15, 2), torch.randn(2, 8), 0.8, None, None, None))
            lokr = self.adapter_type(set(), (None, None, alpha, torch.randn(3, 3), torch.randn(3, 2),
                                             torch.randn(5, 2), torch.randn(2, 4), None, None))
            for batch in ((), (2,), (2, 3)):
                x = torch.randn(*batch, 8)
                self.assert_hook_parity(layer, [(loha, -0.6)], x)
                self.assert_hook_parity(layer, [(loha, -0.6), (lora, 0.7), (lokr, 1.3)], x)

    def test_convolutional_lora_stack(self):
        for dim, conv in enumerate((torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d), 1):
            for kernel, stride, padding, dilation in ((1, 1, 0, 1), (3, 2, 2, 2), (3, 1, 'same', 1)):
                layer = conv(3, 5, kernel, stride=stride, padding=padding, dilation=dilation)
                for flat in (False, True):
                    components = []
                    for alpha, strength in ((None, -0.6), (0.7, 1.3), (0.0, 0.5)):
                        up = torch.randn(5, 2, *([1] * dim))
                        down = torch.randn(2, 3, *([kernel] * dim))
                        if flat:
                            up, down = up.flatten(1), down.flatten(1)
                        components.append((self.native.LoRAAdapter(set(), (up, down, alpha, None, None, None)), strength))
                    x = torch.randn(2, 3, *([9] * dim))
                    with self.subTest(dim=dim, kernel=kernel, flat=flat):
                        self.assert_hook_parity(layer, components[:1], x)
                        self.assert_hook_parity(layer, components, x)

    def test_unsupported_loha_and_convolutions(self):
        weights = [torch.randn(5, 2), torch.randn(2, 3), 1.0, torch.randn(5, 2), torch.randn(2, 3), None, None, None]
        for index, value, reason in ((5, torch.ones(2, 2, 1, 1), 'Tucker'),
                                     (7, torch.ones(5), 'DoRA'), (4, torch.ones(7, 3), 'incompatible')):
            invalid = weights.copy()
            invalid[index] = value
            adapter = self.native.LoHaAdapter(set(), invalid)
            self.assertIn(reason, self.safe._bypass_compatibility_error(adapter, torch.nn.Linear(3, 5)))
        lora = self.native.LoRAAdapter(set(), (torch.randn(4, 2), torch.randn(2, 4), 1.0, None, None, None))
        for layer in (torch.nn.Conv2d(4, 4, 1, groups=2), torch.nn.Conv2d(4, 4, 1, padding_mode='reflect')):
            root = types.SimpleNamespace(layer=layer)
            bypass, regular, reasons = self.safe._partition_bypass_targets(root, {'layer.weight'}, {'layer.weight': [(lora, 1.0)]})
            self.assertFalse(bypass)
            self.assertIn('layer.weight', regular)
            self.assertIn('convolutions', reasons['layer.weight'][0])


    def apply_lokr(self, root, injections=None):
        import copy

        class Patcher:
            def __init__(self, model):
                self.model = model
                self.injections = dict(injections or {})
                self.attachments = {}
                self.is_injected = False

            def clone(self):
                result = copy.copy(self)
                result.injections = dict(self.injections)
                result.attachments = dict(self.attachments)
                return result

            def set_injections(self, key, value):
                self.injections[key] = value

            def get_injections(self, key):
                return self.injections.get(key)

            def get_attachment(self, key):
                return self.attachments.get(key)

            def set_attachments(self, key, value):
                self.attachments[key] = value

            def inject(self):
                for group in self.injections.values():
                    for injection in group:
                        injection.inject(self)

            def eject(self):
                # ComfyUI eject_model uses the same order as inject_model.
                for group in self.injections.values():
                    for injection in group:
                        injection.eject(self)

        adapter = self.adapter_type(set(), (torch.ones(3, 2), None, 1.0, None, None,
                                            torch.ones(5, 2), torch.ones(2, 4), None, None))
        loader = types.SimpleNamespace(load_lbw=lambda *args, **kwargs: ({'layer.weight': (adapter, 1.)}, [], '1'))
        with patch.object(self.safe, 'LoraLoaderBlockWeight', loader):
            result = self.safe._apply_bypass_applications(Patcher(root), [({}, 1., '1')])
        return result, adapter

    def test_bypass_rebinds_to_model_copy(self):
        root = torch.nn.Module()
        root.layer = torch.nn.Linear(8, 15, bias=False)
        model, adapter = self.apply_lokr(root)
        delegate = model.clone()
        delegate.model = torch.nn.Module()
        delegate.model.layer = torch.nn.Linear(8, 15, bias=False)
        x = torch.ones(1, 8)
        before = root.layer(x)
        delegate_before = delegate.model.layer(x)
        expected_delta = F.linear(x, adapter.calculate_weight(torch.zeros(15, 8), 'layer.weight', 1., 1., None, lambda x: x))
        try:
            delegate.inject()
            torch.testing.assert_close(delegate.model.layer(x), delegate_before + expected_delta)
            torch.testing.assert_close(root.layer(x), before)
        finally:
            delegate.eject()
        torch.testing.assert_close(delegate.model.layer(x), delegate_before)

    def test_eject_merge_then_lora_does_not_restore_stale_merge(self):
        class Swap(self.native.WeightAdapterBase):
            weights = ()
            loaded_keys = set()

            def bypass_forward(self, original, x, *args, **kwargs):
                return torch.full((*x.shape[:-1], 15), 7.)

        class Composable(list):
            def __bool__(self):
                return False

        root = torch.nn.Module()
        root.layer = torch.nn.Linear(8, 15, bias=False)
        manager = self.native.BypassInjectionManager()
        manager.add_adapter('layer.weight', Swap())
        swap_injections = manager.create_injections(root)
        model, _ = self.apply_lokr(root, {'merge': Composable(swap_injections)})
        x = torch.ones(1, 8)
        original = root.layer(x)
        for _ in range(3):
            try:
                model.inject()
                torch.testing.assert_close(root.layer(x), torch.full((1, 15), 15.))
            finally:
                model.eject()
            torch.testing.assert_close(root.layer(x), original)


    def test_bypass_cleanup_when_sampling_copy_is_discarded(self):
        import gc
        import weakref
        root = torch.nn.Module()
        root.layer = torch.nn.Linear(8, 15, bias=False)
        model, _ = self.apply_lokr(root)
        x = torch.ones(1, 8)
        original = root.layer(x)
        copy_model = model.clone()
        copy_model.inject()
        self.assertFalse(torch.equal(root.layer(x), original))
        reference = weakref.ref(copy_model)
        del copy_model
        gc.collect()
        self.assertIsNone(reference())
        torch.testing.assert_close(root.layer(x), original)

    def test_shared_root_copy_does_not_double_apply_or_eject_newer_copy(self):
        root = torch.nn.Module()
        root.layer = torch.nn.Linear(8, 15, bias=False)
        model, _ = self.apply_lokr(root)
        newer = model.clone()
        x = torch.ones(1, 8)
        original = root.layer(x)
        try:
            model.inject()
            expected = root.layer(x)
            newer.inject()
            torch.testing.assert_close(root.layer(x), expected)
            model.eject()
            torch.testing.assert_close(root.layer(x), expected)
        finally:
            newer.eject()
            model.eject()
        torch.testing.assert_close(root.layer(x), original)

    def test_rebinding_keeps_components_available_for_save(self):
        from donut_bypass_materialization import get_bypass_components
        root = torch.nn.Module()
        root.layer = torch.nn.Linear(8, 15, bias=False)
        model, adapter = self.apply_lokr(root)
        before = get_bypass_components(model)
        self.assertEqual(set(before), {'layer.weight'})
        try:
            model.inject()
            after = get_bypass_components(model)
            self.assertEqual(after, before)
            saved, strength = after['layer.weight'][0]
            self.assertEqual(strength, 1.)
            expected = adapter.calculate_weight(torch.zeros(15, 8), 'layer.weight', 1., 1., None, lambda x: x)
            actual = saved.calculate_weight(torch.zeros(15, 8), 'layer.weight', strength, 1., None, lambda x: x)
            torch.testing.assert_close(actual, expected)
        finally:
            model.eject()


if __name__ == '__main__':
    unittest.main()

import importlib.util
import sys
import types
import unittest
import uuid
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
MODULE_PATH = ROOT / 'DonutModelMergeKrea2.py'


class PatcherInjection:
    def __init__(self, inject, eject):
        self.inject = inject
        self.eject = eject


class WeightAdapterBase:
    def h(self, x, base_out):
        return torch.zeros_like(base_out)

    def g(self, value):
        return value

    def bypass_forward(self, original_forward, x, *args, **kwargs):
        base_out = original_forward(x, *args, **kwargs)
        return self.g(base_out + self.h(x, base_out))


class BypassInjectionManager:
    def __init__(self):
        self.adapters = {}
        self.hooks = []

    def add_adapter(self, key, adapter, strength=1.0):
        module_path = key[:-7] if key.endswith('.weight') else key
        self.adapters[module_path] = (adapter, strength)

    @staticmethod
    def _resolve(root, path):
        module = root
        for part in path.split('.'):
            module = module[int(part)] if part.isdigit() else getattr(module, part)
        return module

    def create_injections(self, root):
        hook_specs = []
        for path, (adapter, strength) in self.adapters.items():
            module = self._resolve(root, path)
            hook_specs.append((module, adapter, strength))
        self.hooks = hook_specs

        originals = []

        def inject(_patcher):
            for module, adapter, strength in hook_specs:
                adapter.multiplier = strength
                original = module.forward
                originals.append((module, original))

                def wrapped(x, *args, _original=original, _adapter=adapter, **kwargs):
                    return _adapter.bypass_forward(_original, x, *args, **kwargs)

                module.forward = wrapped

        def eject(_patcher):
            for module, original in reversed(originals):
                module.forward = original
            originals.clear()

        return [PatcherInjection(inject, eject)]

    def get_hook_count(self):
        return len(self.hooks)


def load_module():
    comfy = types.ModuleType('comfy')
    comfy.__path__ = []
    weight_adapter = types.ModuleType('comfy.weight_adapter')
    weight_adapter.WeightAdapterBase = WeightAdapterBase
    weight_adapter.BypassInjectionManager = BypassInjectionManager
    patcher_extension = types.ModuleType('comfy.patcher_extension')
    patcher_extension.PatcherInjection = PatcherInjection
    comfy.weight_adapter = weight_adapter
    comfy.patcher_extension = patcher_extension

    injected = {
        'comfy': comfy,
        'comfy.weight_adapter': weight_adapter,
        'comfy.patcher_extension': patcher_extension,
    }
    old = {name: sys.modules.get(name) for name in injected}
    sys.modules.update(injected)
    try:
        spec = importlib.util.spec_from_file_location('donut_model_merge_krea2_tested', MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in old.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


module = load_module()


class TinyKrea(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = torch.nn.Module()
        self.diffusion_model.first = torch.nn.Linear(2, 2, bias=True)
        self.diffusion_model.first.register_buffer('weight_scale', torch.tensor(1.0))
        self.diffusion_model.blocks = torch.nn.ModuleList([torch.nn.Module()])
        self.diffusion_model.blocks[0].proj = torch.nn.Linear(2, 2, bias=False)
        self.diffusion_model.register_parameter('raw_scale', torch.nn.Parameter(torch.ones(2)))


class FakePatcher:
    def __init__(self, model, key_patches=None):
        self.model = model
        self._key_patches = key_patches or {}
        self.added = []
        self.injections = {}
        self.additional_models = {}
        self.attachments = {}
        self.is_injected = False
        self.load_device = torch.device('cpu')
        self.patches_uuid = uuid.uuid4()

    def clone(self):
        clone = FakePatcher(self.model, self._key_patches)
        clone.injections = {key: list(value) for key, value in self.injections.items()}
        clone.patches_uuid = self.patches_uuid
        clone.attachments = dict(self.attachments)
        clone.additional_models = {
            key: [model.clone() for model in models]
            for key, models in self.additional_models.items()
        }
        return clone

    def is_clone(self, other):
        return self.model is other.model

    def get_key_patches(self, prefix):
        return {
            key: value
            for key, value in self._key_patches.items()
            if key.startswith(prefix)
        }

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        self.added.append((patches, strength_patch, strength_model))

    def set_injections(self, key, injections):
        self.injections[key] = injections

    def set_additional_models(self, key, models):
        self.additional_models[key] = models

    def get_additional_models_with_key(self, key):
        return self.additional_models.get(key, [])

    def set_attachments(self, key, value):
        self.attachments[key] = value


class InputSchemaTests(unittest.TestCase):
    def test_matches_core_krea2_controls_and_appends_execution_mode(self):
        inputs = module.DonutModelMergeKrea2.INPUT_TYPES()
        required = list(inputs['required'])
        expected = [
            'model1',
            'model2',
            'first.',
            'tmlp.',
            'txtmlp.',
            'tproj.',
            'txtfusion.layerwise_blocks.0.',
            'txtfusion.layerwise_blocks.1.',
            'txtfusion.projector.',
            'txtfusion.refiner_blocks.0.',
            'txtfusion.refiner_blocks.1.',
            *[f'blocks.{index}.' for index in range(28)],
            'last.',
        ]
        self.assertEqual(required, expected)
        self.assertEqual(inputs['optional']['execution_mode'][1]['default'], 'Comfy patches')


class AdapterTests(unittest.TestCase):
    def test_blends_model1_and_model2_forward_outputs(self):
        base = torch.nn.Linear(2, 1)
        source = torch.nn.Linear(2, 1)
        with torch.no_grad():
            base.weight[:] = torch.tensor([[1.0, 0.0]])
            base.bias[:] = torch.tensor([1.0])
            source.weight[:] = torch.tensor([[0.0, 2.0]])
            source.bias[:] = torch.tensor([-1.0])

        source_root = torch.nn.Module()
        source_root.layer = source
        source_patcher = types.SimpleNamespace(model=source_root)
        adapter = module._ModelBlendBypassAdapter(source_patcher, 'layer', 0.25)
        x = torch.tensor([[2.0, 3.0]])

        actual = adapter.bypass_forward(base.forward, x)
        expected = 0.25 * base(x) + 0.75 * source(x)
        self.assertTrue(torch.allclose(actual, expected))

    def test_ratio_zero_does_not_call_model1_forward(self):
        source = torch.nn.Linear(1, 1, bias=False)
        source.weight.data.fill_(3.0)
        root = torch.nn.Module()
        root.layer = source
        adapter = module._ModelBlendBypassAdapter(types.SimpleNamespace(model=root), 'layer', 0.0)

        def fail_base(_x):
            raise AssertionError('model1 should not execute at ratio 0')

        self.assertTrue(torch.equal(adapter.bypass_forward(fail_base, torch.ones(1, 1)), torch.tensor([[3.0]])))


class MergeTests(unittest.TestCase):
    def test_regular_mode_uses_core_ratio_orientation_and_longest_prefix(self):
        root1 = TinyKrea()
        root2 = TinyKrea()
        patches = {
            'diffusion_model.blocks.0.proj.weight': object(),
            'diffusion_model.raw_scale': object(),
        }
        model1 = FakePatcher(root1)
        model2 = FakePatcher(root2, patches)

        (merged,) = module.DonutModelMergeKrea2().merge(
            model1,
            model2,
            execution_mode='Comfy patches',
            **{'first.': 0.9, 'blocks.0.': 0.2, 'last.': 0.7},
        )

        by_key = {next(iter(call[0])): call[1:] for call in merged.added}
        self.assertEqual(by_key['diffusion_model.blocks.0.proj.weight'], (0.8, 0.2))
        self.assertAlmostEqual(by_key['diffusion_model.raw_scale'][0], 0.1)
        self.assertEqual(by_key['diffusion_model.raw_scale'][1], 0.9)

    def test_experimental_mode_bypasses_all_direct_linear_state(self):
        root1 = TinyKrea()
        root2 = TinyKrea()
        with torch.no_grad():
            root1.diffusion_model.first.weight.copy_(torch.eye(2))
            root1.diffusion_model.first.bias.zero_()
            root2.diffusion_model.first.weight.copy_(2.0 * torch.eye(2))
            root2.diffusion_model.first.bias.fill_(1.0)

        patches = {
            'diffusion_model.first.weight': object(),
            'diffusion_model.first.bias': object(),
            'diffusion_model.first.weight_scale': object(),
            'diffusion_model.raw_scale': object(),
        }
        model1 = FakePatcher(root1)
        model2 = FakePatcher(root2, patches)

        (merged,) = module.DonutModelMergeKrea2().merge(
            model1,
            model2,
            execution_mode='Experimental bypass',
            **{'first.': 0.25, 'last.': 1.0},
        )

        regular_keys = {next(iter(call[0])) for call in merged.added}
        self.assertEqual(regular_keys, {'diffusion_model.raw_scale'})
        self.assertIn(module._SOURCE_MODELS_KEY, merged.additional_models)
        self.assertIn(module._INJECTION_KEY, merged.injections)
        self.assertNotEqual(merged.patches_uuid, model1.patches_uuid)
        identity_keys = [key for key in merged.attachments if key.startswith(module._CACHE_ATTACHMENT_PREFIX)]
        self.assertEqual(len(identity_keys), 1)

        outer = merged.injections[module._INJECTION_KEY][0]
        outer.inject(merged)
        try:
            x = torch.tensor([[2.0, 4.0]])
            actual = merged.model.diffusion_model.first(x)
            # Explicit expected output: .25*x + .75*(2*x + 1)
            expected = 0.25 * x + 0.75 * (2.0 * x + 1.0)
            self.assertTrue(torch.allclose(actual, expected))
        finally:
            outer.eject(merged)

    def test_patchless_bypass_gets_a_unique_cache_identity_attachment(self):
        root1 = TinyKrea()
        root2 = TinyKrea()
        patches = {
            'diffusion_model.first.weight': object(),
            'diffusion_model.first.bias': object(),
            'diffusion_model.first.weight_scale': object(),
        }
        model1 = FakePatcher(root1)
        model2 = FakePatcher(root2, patches)

        (merged,) = module.DonutModelMergeKrea2().merge(
            model1,
            model2,
            execution_mode='Experimental bypass',
            **{'first.': 0.5},
        )

        self.assertEqual(merged.added, [])
        identity_keys = [
            key for key in merged.attachments
            if key.startswith(module._CACHE_ATTACHMENT_PREFIX)
        ]
        self.assertEqual(len(identity_keys), 1)
        cloned = merged.clone()
        self.assertEqual(set(cloned.attachments), set(merged.attachments))

    def test_shared_model_falls_back_to_regular_patches(self):
        root = TinyKrea()
        patches = {'diffusion_model.first.weight': object()}
        model1 = FakePatcher(root)
        model2 = FakePatcher(root, patches)

        (merged,) = module.DonutModelMergeKrea2().merge(
            model1,
            model2,
            execution_mode='Experimental bypass',
            **{'first.': 0.5},
        )
        self.assertEqual(len(merged.added), 1)
        self.assertFalse(merged.injections)
        self.assertFalse(merged.additional_models)

    def test_existing_injection_falls_back_to_regular_patches(self):
        root1 = TinyKrea()
        root2 = TinyKrea()
        patches = {'diffusion_model.first.weight': object()}
        model1 = FakePatcher(root1)
        model1.injections['existing'] = [object()]
        model2 = FakePatcher(root2, patches)

        (merged,) = module.DonutModelMergeKrea2().merge(
            model1,
            model2,
            execution_mode='Experimental bypass',
            **{'first.': 0.5},
        )
        self.assertEqual(len(merged.added), 1)
        self.assertEqual(set(merged.injections), {'existing'})
        self.assertFalse(merged.additional_models)


if __name__ == '__main__':
    unittest.main()

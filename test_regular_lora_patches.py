"""Regular LBW routing through real Donut merge hooks on small CPU layers."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from test_model_merge_krea2 import module as merge
from test_uncensorfix_merge_bypass import Adapter, Patcher, make_root

ROOT = Path(__file__).resolve().parent
BODY = 'diffusion_model.blocks.0.proj.weight'
FUSION = 'diffusion_model.txtfusion.layerwise_blocks.0.attn.wq.weight'
TEXT = 'diffusion_model.text_embedding.weight'
CLIP = 'clip.proj.weight'


def load_lbw_module():
    name = '_donut_regular_patch_tests'
    package = types.ModuleType(name)
    package.__path__ = [str(ROOT)]
    comfy = types.ModuleType('comfy')
    comfy.__path__ = []
    comfy.utils = types.ModuleType('comfy.utils')
    comfy.lora = types.ModuleType('comfy.lora')
    comfy.lora.model_lora_keys_unet = lambda model: {}
    comfy.lora.model_lora_keys_clip = lambda clip, keys: keys
    # These tests start at parsed adapter objects; native LoKr tensor parsing
    # and arithmetic are covered by test_lokr_bypass_parity.
    comfy.lora.load_lora = lambda lora, keys, **kwargs: dict(lora)
    cli = types.ModuleType('comfy.cli_args')
    cli.args = types.SimpleNamespace()
    folders = types.ModuleType('folder_paths')
    folders.models_dir = '/tmp'
    server = types.ModuleType('server')
    server.PromptServer = object
    libs = types.ModuleType(name + '.libs')
    utils = types.ModuleType(name + '.libs.utils')
    utils.add_folder_path_and_extensions = lambda *args: None
    libs.utils = utils
    injected = {name: package, 'comfy': comfy, 'comfy.utils': comfy.utils,
                'comfy.lora': comfy.lora, 'comfy.cli_args': cli,
                'folder_paths': folders, 'nodes': types.ModuleType('nodes'),
                'server': server, libs.__name__: libs, utils.__name__: utils}
    with patch.dict(sys.modules, injected):
        spec = importlib.util.spec_from_file_location(name + '.lora_block_weight', ROOT / 'lora_block_weight.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


class Clip:
    def __init__(self, patcher=None):
        self.patcher = patcher or Patcher(make_root([CLIP], .4))
        self.cond_stage_model = self.patcher.model

    def clone(self):
        return Clip(self.patcher.clone())

    def add_patches(self, patches, strength):
        return self.patcher.add_patches(patches, strength)


class RegularPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lbw = load_lbw_module()
        cls.adapter = Adapter((), (torch.full((2, 1), .3), torch.full((1, 3), .2), 1., None, None, None))
        cls.x = torch.tensor([[.3, -.7, 1.1]])

    def models(self):
        return Patcher(make_root([BODY, FUSION, TEXT], .2)), Patcher(make_root([BODY, FUSION, TEXT], .8))

    def merged(self, mode='Experimental bypass', fusion_ratio=0.):
        base, source = self.models()
        result, = merge.DonutModelMergeKrea2().merge(base, source, mode, **{'first.': 1., 'txtfusion.': fusion_ratio})
        return result

    def apply(self, model, patches, strength=1., clip=None, clip_strength=0., vector=','.join(['1'] * 29)):
        return self.lbw.LoraLoaderBlockWeight.load_lora_for_models(
            model, clip, patches, strength, clip_strength, False, 0, 1., 1., vector)

    def output(self, model, key):
        with model.activate():
            return model.model.get_submodule(key[:-7])(self.x).detach().clone()

    def test_regular_lora_after_runtime_merge_changes_active_source(self):
        original = self.merged()
        for strength in (1., -.5, 2.):
            with self.subTest(strength=strength):
                patched, _, _ = self.apply(original, {BODY: self.adapter, FUSION: self.adapter}, strength)
                for key in (BODY, FUSION):
                    expected = self.output(original, key) + F.linear(self.x, self.adapter.delta()) * strength
                    torch.testing.assert_close(self.output(patched, key), expected)
                source = patched.get_additional_models_with_key(merge._SOURCE_MODELS_KEY)[0]
                self.assertIn(FUSION, source.patches)
                self.assertNotIn(FUSION, patched.patches)
                self.assertFalse(original.patches)
                self.assertFalse(original.get_additional_models_with_key(merge._SOURCE_MODELS_KEY)[0].patches)

    def test_diffusion_text_named_layer_is_not_mistaken_for_clip(self):
        base, _ = self.models()
        patched, _, _ = self.apply(base, {TEXT: self.adapter})
        torch.testing.assert_close(self.output(patched, TEXT), self.output(base, TEXT) + F.linear(self.x, self.adapter.delta()))

    def test_clip_routing_uses_actual_target_and_its_own_strength(self):
        base, _ = self.models()
        clip = Clip()
        patched, clip_result, _ = self.apply(base, {TEXT: self.adapter, CLIP: self.adapter}, 1.5, clip, -.25)
        self.assertIn(TEXT, patched.patches)
        self.assertNotIn(CLIP, patched.patches)
        self.assertEqual(clip_result.patcher.patches[CLIP][0][0], -.25)
        self.assertFalse(clip.patcher.patches)

    def test_zero_strength_does_not_register_weight_patches(self):
        base, _ = self.models()
        patched, _, _ = self.apply(base, {BODY: self.adapter}, strength=0.)
        self.assertFalse(patched.patches)

    def test_apply_saved_lbw_uses_same_routing(self):
        base = self.merged()
        patched, _ = self.lbw.ApplyLBW.doit(base, Clip(), 2., -.5,
                                            {'blocks': {FUSION: (self.adapter, .25), TEXT: (self.adapter, .5)}, 'muted': []})
        torch.testing.assert_close(self.output(patched, FUSION), self.output(base, FUSION) + F.linear(self.x, self.adapter.delta()) * .5)
        torch.testing.assert_close(self.output(patched, TEXT), self.output(base, TEXT) + F.linear(self.x, self.adapter.delta()))


    def test_regular_merge_then_lora_preserves_blend_and_delta(self):
        for ratio in (0., .3, 1.):
            base = self.merged('Comfy patches', ratio)
            patched, _, _ = self.apply(base, {BODY: self.adapter, FUSION: self.adapter})
            for key in (BODY, FUSION):
                torch.testing.assert_close(self.output(patched, key), self.output(base, key) + F.linear(self.x, self.adapter.delta()))
            self.assertFalse(patched.injections)

    def test_grouped_node_comfy_patches_then_lora(self):
        name = self.lbw.__package__ + '.donut_grouped_merge'
        spec = importlib.util.spec_from_file_location(name, ROOT / 'donut_grouped_merge.py')
        grouped = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {self.lbw.__package__ + '.DonutModelMergeKrea2': merge}):
            spec.loader.exec_module(grouped)
        for ratio in (0., .3, 1.):
            model1, model2 = self.models()
            base, = grouped.DonutModelMergeKrea2Grouped().merge_grouped(
                model1=model1, model2=model2, ratio_mode='Grouped',
                body_ratio=ratio, fusion_ratio=ratio, execution_mode='Comfy patches')
            patched, _, _ = self.apply(base, {BODY: self.adapter, FUSION: self.adapter})
            for key in (BODY, FUSION):
                expected = self.output(model1, key) * ratio + self.output(model2, key) * (1. - ratio)
                expected += F.linear(self.x, self.adapter.delta())
                torch.testing.assert_close(self.output(patched, key), expected)
            self.assertFalse(patched.injections)

    def test_source_stack_preserves_existing_patches_vectors_and_cache_identity(self):
        base = self.merged()
        first, _, _ = self.apply(base, {FUSION: self.adapter}, strength=.5)
        vector = ','.join(['0.5', '0.25'] + ['1'] * 27)
        second, _, _ = self.apply(first, {BODY: self.adapter, FUSION: self.adapter}, strength=2., vector=vector)
        source = second.get_additional_models_with_key(merge._SOURCE_MODELS_KEY)[0]
        self.assertEqual([p[0] for p in source.patches[FUSION]], [.5, 1.])
        self.assertEqual(second.patches[BODY][0][0], .5)
        self.assertFalse(base.clone_has_same_weights(first))
        self.assertTrue(first.clone_has_same_weights(first.clone()))
        torch.testing.assert_close(self.output(second, FUSION), self.output(base, FUSION) + F.linear(self.x, self.adapter.delta()) * 1.5)
        self.assertEqual(len(first.get_additional_models_with_key(merge._SOURCE_MODELS_KEY)[0].patches[FUSION]), 1)

    def test_muted_source_and_zero_clip_strength_create_no_patches(self):
        base = self.merged()
        patched, clip, _ = self.apply(base, {BODY: self.adapter, FUSION: self.adapter, CLIP: self.adapter},
                                      clip=Clip(), vector=','.join(['0'] * 29))
        self.assertFalse(patched.patches)
        self.assertFalse(clip.patcher.patches)
        self.assertFalse(patched.get_additional_models_with_key(merge._SOURCE_MODELS_KEY)[0].patches)
        self.assertTrue(base.clone_has_same_weights(patched))

    def test_rejected_active_source_patch_raises_instead_of_silent_noop(self):
        base = self.merged()
        source = base.get_additional_models_with_key(merge._SOURCE_MODELS_KEY)[0]
        source.reject.add(FUSION)
        with self.assertRaisesRegex(RuntimeError, 'active Krea2 merge source'):
            self.apply(base, {FUSION: self.adapter})
        self.assertFalse(source.patches)

    def test_load_lbw_preserves_offsets_and_functions_for_all_groups(self):
        model, _ = self.models()
        for target in (BODY, FUSION, 'diffusion_model.input_blocks.0.proj.weight'):
            key = (target, (0, 0, 1), lambda value: value * .5)
            weights, _, _ = self.lbw.LoraLoaderBlockWeight.load_lbw(model, None, {key: self.adapter}, False, 0, 1., 1., ','.join(['1'] * 29))
            self.assertEqual(set(weights), {key})

    def test_saved_merge_uses_regular_patches_on_retained_source(self):
        from donut_krea2_merge_serialization import get_krea2_merge_bypass_info, compose_krea2_merge_unet_state_dict
        base = self.merged()
        patched, _, _ = self.apply(base, {BODY: self.adapter, FUSION: self.adapter})
        source, plans, _ = get_krea2_merge_bypass_info(patched)
        state, _, _ = compose_krea2_merge_unet_state_dict(patched, source, plans)
        for key in (BODY, FUSION):
            local = key.removeprefix('diffusion_model.')
            expected = self.output(patched, key)
            actual = F.linear(self.x, state[local], state[local.removesuffix('weight') + 'bias'])
            torch.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()

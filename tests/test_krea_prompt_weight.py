"""Synthetic model/encoder contract; no user backend, checkpoint or GPU."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys
import types


class Clip:
    def __init__(self):
        self.encoded = []

    def tokenize(self, text):
        ids = [151644, 872, 198, *map(ord, text), 151645]
        return {"qwen3vl_4b": [[(i, 1) for i in ids]]}

    def encode_from_tokens_scheduled(self, tokens):
        self.encoded.append(tokens)
        return [[SimpleNamespace(shape=(1, len(tokens['qwen3vl_4b'][0]), 4)), {}]]


class Model:
    def __init__(self):
        self.model_options = {'transformer_options': {'existing': 'retained'}}
        self.patches = {}
        self.blocks = [SimpleNamespace(attn=SimpleNamespace()) for _ in range(2)]

    def clone(self):
        result = Model()
        result.model_options = self.model_options.copy()
        return result

    def get_model_object(self, name):
        assert name == 'diffusion_model'
        return SimpleNamespace(blocks=self.blocks)

    def add_object_patch(self, name, value):
        self.patches[name] = value


class WeightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'donut_krea_prompt_weight.py'
        spec = importlib.util.spec_from_file_location('weight_under_test', path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    # AC: @create-krea-prompt-weights ac-execution
    def test_existing_conditioning_and_model_are_preserved_while_cloning_weights(self):
        clip, model = Clip(), Model()
        conditioning = [[SimpleNamespace(shape=(1, 8, 4)), {'variance': 'retained'}]]
        weighted, cond = self.module.DonutKrea2PromptWeight().encode(
            clip, model, '(cat:1.5) (dog:-1)', 1, conditioning)
        self.assertIs(cond, conditioning)
        self.assertFalse(clip.encoded)
        self.assertIsNot(weighted, model)
        weights = weighted.model_options['transformer_options']['krea2_token_weights']
        self.assertEqual(weights, [(i, 1, 1) for i in (0, 1, 2)] + [(i, -1, 0) for i in (3, 4, 5, 6)])
        self.assertEqual(weighted.model_options['transformer_options']['existing'], 'retained')
        self.assertNotIn('krea2_token_weights', model.model_options['transformer_options'])
        self.assertEqual(len(weighted.patches), 2)

    # AC: @create-krea-prompt-weights ac-control
    def test_disabled_and_unweighted_are_noop_with_existing_conditioning(self):
        for text, strength in [('(cat:2)', 0), ('a cat', 1), ('(cat:1)', 1)]:
            clip, model = Clip(), Model()
            cond = [[SimpleNamespace(shape=(1, 5, 4)), {}]]
            result, returned = self.module.DonutKrea2PromptWeight().encode(clip, model, text, strength, cond)
            self.assertIs(result, model)
            self.assertIs(returned, cond)
            self.assertFalse(clip.encoded)

    # AC: @create-krea-prompt-weights ac-execution
    def test_standalone_encode_removes_syntax_and_scales_strength(self):
        clip, model = Clip(), Model()
        weighted, cond = self.module.DonutKrea2PromptWeight().encode(clip, model, '(cat:2)', .5)
        self.assertEqual(len(clip.encoded), 1)
        self.assertEqual(weighted.model_options['transformer_options']['krea2_token_weights'],
                         [(i, 1, 1) for i in (3, 4, 5)])

    # AC: @create-krea-prompt-weights ac-execution
    def test_conditioning_encoder_cleans_scene_before_variance_and_keeps_face_separate(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location('prompt_under_test', root / 'donut_prompt.py')
        prompt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prompt)
        nodes = types.ModuleType('nodes')
        class Encode:
            def encode(self, clip, text):
                return (clip.encode_from_tokens_scheduled(clip.tokenize(text)),)
        class Zero:
            def zero_out(self, cond):
                return (cond,)
        nodes.CLIPTextEncode, nodes.ConditioningZeroOut = Encode, Zero
        def variance(positive, face, **options):
            positive[0][1]['variance'] = 'retained'
            return positive, face
        text = '(cat:1.5) without (fog:-1)'
        with patch.dict(sys.modules, {'nodes': nodes}), patch.object(prompt, 'enhance_prompt_pair', variance), patch.object(prompt, 'wildcard_roots', lambda: ()):
            for enabled in (False, True):
                clip = Clip()
                result = prompt.DonutPromptConditioning().encode(clip, 'Calm eyes. ', text, '', krea_prompt_weights=enabled)['result']
                self.assertEqual(result[0], 'Calm eyes. ' + text)
                expected = 'Calm eyes. ' + ('cat without fog' if enabled else text)
                self.assertEqual(clip.encoded[0], clip.tokenize(expected))
                self.assertEqual(clip.encoded[1], clip.tokenize('Calm eyes. '))
                self.assertEqual(result[3][0][1]['variance'], 'retained')
                if enabled:
                    model, cond = self.module.DonutKrea2PromptWeight().encode(clip, Model(), result[0], 1, result[3])
                    self.assertIs(cond, result[3])
                    weights = model.model_options['transformer_options']['krea2_token_weights']
                    self.assertEqual([pos for pos, value, _ in weights if value == 1], [13, 14, 15, 16])
                    self.assertEqual(len(clip.encoded), 3)

    # AC: @create-krea-prompt-weights ac-errors
    def test_invalid_strength_and_non_krea_encoder_are_actionable(self):
        node = self.module.DonutKrea2PromptWeight()
        for value in (True, -1, 5, float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, 'strength'):
                node.encode(Clip(), Model(), '(cat:2)', value)
        clip = Clip()
        clip.tokenize = lambda _: {'clip_l': [[(1, 1)]]}
        with self.assertRaisesRegex(ValueError, 'Qwen3'):
            node.encode(clip, Model(), '(cat:2)', 1)

    # AC: @create-krea-prompt-weights ac-execution
    def test_attention_emphasis_and_removal_change_numeric_cpu_output(self):
        import torch
        modules = {name: types.ModuleType(name) for name in (
            'comfy', 'comfy.ldm', 'comfy.ldm.modules', 'comfy.ldm.modules.attention',
            'comfy.ldm.flux', 'comfy.ldm.flux.math')}
        def attention(q, k, v, heads, mask=None, **kwargs):
            result = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            return result.transpose(1, 2).reshape(result.shape[0], result.shape[2], -1)
        modules['comfy.ldm.modules.attention'].optimized_attention = attention
        modules['comfy.ldm.modules.attention'].attention_pytorch = attention
        modules['comfy.ldm.flux.math'].apply_rope = lambda q, k, freqs: (q, k)
        attn = SimpleNamespace(heads=1, kvheads=1, wq=torch.zeros_like, wk=torch.zeros_like,
                               wv=lambda x: x, gate=torch.zeros_like, wo=lambda x: x,
                               qknorm=lambda q, k: (q, k))
        value = torch.tensor([[[1.], [3.], [5.]]])
        with patch.dict(sys.modules, modules):
            normal = self.module.krea2_attn_forward_weight(attn, value)
            emphasized = self.module.krea2_attn_forward_weight(attn, value,
                transformer_options={'krea2_token_weights': [(2, 1, 2)]})
            removed = self.module.krea2_attn_forward_weight(attn, value,
                transformer_options={'krea2_token_weights': [(2, -1, 0)]})
        torch.testing.assert_close(normal, torch.full_like(normal, 1.5))
        torch.testing.assert_close(removed, torch.full_like(removed, -1/6))
        expected = (torch.softmax(torch.tensor([0., 0., 2.]), 0) * value[0, :, 0]).sum() * .5
        torch.testing.assert_close(emphasized, torch.full_like(emphasized, expected))


if __name__ == '__main__':
    unittest.main()

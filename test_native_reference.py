import json
import unittest
from unittest.mock import Mock, call, patch

import torch
import nodes
import donut_prompt
import krea2_variance_integration as variance
import test_edit_studio as edit_tests

module = edit_tests.module


class ReferenceImageTests(unittest.TestCase):
    setUp = edit_tests.EditStudioTests.setUp
    reference = edit_tests.EditStudioTests.reference

    def test_native_images_keep_full_geometry_and_are_independent(self):
        a, b = self.reference((83, 51)), self.reference((39, 72))
        result = module.DonutReferenceStudio().prepare(True, a, b, True)
        self.assertEqual(tuple(result[0].shape), (1, 51, 83, 3))
        self.assertEqual(tuple(result[1].shape), (1, 72, 39, 3))
        self.assertTrue(result[2])
        module.nodes.LoraLoaderModelOnly.assert_not_called()

    def test_off_or_edit_active_does_not_open_references(self):
        with patch.object(module, '_open_reference', side_effect=AssertionError('must not load')):
            for enabled, editing in [(False, False), (True, True)]:
                self.assertEqual(module.DonutReferenceStudio().prepare(enabled, 'missing', edit_active=editing), (None, None, False))
                self.assertEqual(module.DonutReferenceStudio.IS_CHANGED(enabled, 'missing', edit_active=editing), 'disabled')


class NativeConditioningTests(unittest.TestCase):
    def test_native_vision_preserves_main_tokens_but_face_and_negative_are_text_only(self):
        clip = Mock()
        clip.tokenize.return_value = {'qwen3vl_4b': object()}
        vision = [[torch.ones(1, 320, 96), {'vision_metadata': 'keep'}]]
        clip.encode_from_tokens_scheduled.return_value = vision
        ordinary = [[torch.ones(1, 8, 96), {}]]
        image = torch.zeros(1, 80, 120, 3)
        with patch.object(nodes.CLIPTextEncode, 'encode', return_value=(ordinary,)) as plain:
            out = donut_prompt.DonutPromptConditioning().encode(clip, 'same', '', 'same', native_reference_enabled=True, native_reference_a=image)['result']
        # Identical full/face/negative text must not reuse the vision cache entry.
        self.assertIs(out[3], vision)
        self.assertIs(out[4], ordinary)
        self.assertIs(out[6], ordinary)
        self.assertEqual(out[3][0][0].shape[1], 320)
        self.assertEqual(out[3][0][1]['vision_metadata'], 'keep')
        self.assertNotIn('vision_metadata', out[4][0][1])
        self.assertEqual(out[4][0][0].shape[1], 8)
        self.assertEqual(torch.count_nonzero(out[5][0][0]).item(), 0)
        plain.assert_called_once_with(clip, 'same')
        clip.tokenize.assert_called_once()
        from comfy.text_encoders.krea2 import KREA2_TEMPLATE
        args, kwargs = clip.tokenize.call_args
        self.assertIn('<|image_pad|>', args[0]); self.assertEqual(kwargs['llama_template'], KREA2_TEMPLATE)
        self.assertIs(kwargs['images'][0], image)

    def test_two_references_only_reach_the_full_prompt(self):
        clip = Mock()
        clip.tokenize.return_value = {'qwen3vl_4b': object()}
        vision = [[torch.ones(1, 320, 96), {'vision_metadata': 'keep'}]]
        clip.encode_from_tokens_scheduled.return_value = vision
        face = [[torch.ones(1, 8, 96), {'text': 'face'}]]
        negative = [[torch.ones(1, 6, 96), {'text': 'negative'}]]
        a, b = torch.zeros(1, 80, 120, 3), torch.ones(1, 40, 60, 3)
        with patch.object(nodes.CLIPTextEncode, 'encode', side_effect=[(face,), (negative,)]) as plain:
            out = donut_prompt.DonutPromptConditioning().encode(
                clip, 'face', 'scene', 'negative', separator='\n',
                native_reference_enabled=True, native_reference_a=a, native_reference_b=b,
            )['result']
        self.assertEqual(out[:3], ('face\nscene', 'face', ''))
        self.assertIs(out[3], vision)
        self.assertIs(out[4], face)
        self.assertIs(out[6], negative)
        self.assertEqual(plain.call_args_list, [call(clip, 'face'), call(clip, 'negative')])
        clip.tokenize.assert_called_once()
        text = clip.tokenize.call_args.args[0]
        self.assertIn('Reference A:', text)
        self.assertIn('Reference B:', text)
        self.assertTrue(text.endswith('face\nscene'))
        images = clip.tokenize.call_args.kwargs['images']
        self.assertEqual(len(images), 2)
        self.assertIs(images[0], a)
        self.assertIs(images[1], b)
        self.assertEqual(torch.count_nonzero(a).item(), 0)
        self.assertTrue(torch.all(b == 1).item())

    def test_b_only_reference_keeps_face_text_only(self):
        clip = Mock()
        clip.tokenize.return_value = {'qwen3vl_4b': object()}
        vision = [[torch.ones(1, 32, 8), {'vision_metadata': 'B'}]]
        clip.encode_from_tokens_scheduled.return_value = vision
        text = [[torch.ones(1, 8, 8), {}]]
        b = torch.ones(1, 40, 60, 3)
        with patch.object(nodes.CLIPTextEncode, 'encode', return_value=(text,)):
            out = donut_prompt.DonutPromptConditioning().encode(
                clip, 'face', 'scene', 'negative',
                native_reference_enabled=True, native_reference_b=b,
            )['result']
        self.assertIs(out[3], vision)
        self.assertIs(out[4], text)
        clip.tokenize.assert_called_once()
        self.assertEqual(len(clip.tokenize.call_args.kwargs['images']), 1)
        self.assertIs(clip.tokenize.call_args.kwargs['images'][0], b)

    def test_changing_reference_does_not_change_face_conditioning(self):
        clip = Mock()
        clip.tokenize.return_value = {'qwen3vl_4b': object()}
        first = [[torch.ones(1, 32, 8), {'vision_metadata': 'first'}]]
        second = [[torch.zeros(1, 32, 8), {'vision_metadata': 'second'}]]
        clip.encode_from_tokens_scheduled.side_effect = [first, second]
        text = [[torch.ones(1, 8, 8), {}]]
        node = donut_prompt.DonutPromptConditioning()
        with patch.object(nodes.CLIPTextEncode, 'encode', return_value=(text,)):
            outputs = [node.encode(
                clip, 'same', '', 'same', native_reference_enabled=True,
                native_reference_a=image,
            )['result'] for image in (torch.zeros(1, 8, 8, 3), torch.ones(1, 8, 8, 3))]
        self.assertIs(outputs[0][3], first)
        self.assertIs(outputs[1][3], second)
        self.assertIs(outputs[0][4], text)
        self.assertIs(outputs[1][4], text)
        self.assertEqual(clip.tokenize.call_count, 2)

    def test_reference_toggle_does_not_leave_vision_in_face_cache(self):
        clip = Mock()
        clip.tokenize.return_value = {'qwen3vl_4b': object()}
        vision = [[torch.ones(1, 32, 8), {'vision_metadata': 'keep'}]]
        clip.encode_from_tokens_scheduled.return_value = vision
        text = [[torch.ones(1, 8, 8), {}]]
        node = donut_prompt.DonutPromptConditioning()
        with patch.object(nodes.CLIPTextEncode, 'encode', return_value=(text,)):
            for enabled in (True, False, True):
                out = node.encode(
                    clip, 'same', '', 'same', native_reference_enabled=enabled,
                    native_reference_a=torch.ones(1, 8, 8, 3),
                )['result']
                self.assertIs(out[3], vision if enabled else text)
                self.assertIs(out[4], text)
        self.assertEqual(clip.tokenize.call_count, 2)

    def test_variance_receives_separate_vision_and_text_conditioning(self):
        clip = Mock()
        clip.tokenize.return_value = {'qwen3vl_4b': object()}
        vision = [[torch.ones(1, 32, 8), {'vision_metadata': 'keep'}]]
        clip.encode_from_tokens_scheduled.return_value = vision
        text = [[torch.ones(1, 8, 8), {}]]
        settings_seen = []
        def apply(conditioning, settings):
            settings_seen.append((conditioning, settings))
            return [[tensor, {**metadata, variance.VARIANCE_KEY: dict(settings)}]
                    for tensor, metadata in conditioning]
        with patch.object(nodes.CLIPTextEncode, 'encode', return_value=(text,)), \
                patch.object(variance, 'apply_seed_variance', side_effect=apply):
            out = donut_prompt.DonutPromptConditioning().encode(
                clip, 'same', '', 'same', native_reference_enabled=True,
                native_reference_a=torch.ones(1, 8, 8, 3),
                variance_enabled=True, variance_seed=42,
            )['result']
        self.assertEqual(len(settings_seen), 2)
        self.assertIs(settings_seen[0][0], vision)
        self.assertIs(settings_seen[1][0], text)
        self.assertEqual([settings['seed'] for _, settings in settings_seen], [42, 43])
        self.assertEqual(out[3][0][1]['vision_metadata'], 'keep')
        self.assertNotIn('vision_metadata', out[4][0][1])
        self.assertEqual(out[4][0][1][variance.VARIANCE_KEY]['seed'], 43)
        self.assertNotIn(variance.VARIANCE_KEY, out[6][0][1])

    def test_selected_prompt_set_keeps_face_and_edit_text_without_vision(self):
        clip = Mock()
        clip.tokenize.return_value = {'qwen3vl_4b': object()}
        vision = [[torch.ones(1, 32, 8), {'vision_metadata': 'keep'}]]
        clip.encode_from_tokens_scheduled.return_value = vision
        text = [[torch.ones(1, 8, 8), {}]]
        variants = json.dumps([{'face': '{selected|selected}', 'scene': '', 'negative': 'selected'}])
        with patch.object(nodes.CLIPTextEncode, 'encode', return_value=(text,)) as plain:
            out = donut_prompt.DonutPromptConditioning().encode(
                clip, 'base face', 'base scene', 'base negative', edit_negative='edit negative',
                native_reference_enabled=True, native_reference_a=torch.ones(1, 8, 8, 3),
                prompt_sets_json=variants, prompt_set_index=2,
            )['result']
        self.assertEqual(out[:3], ('selected', 'selected', 'edit negative'))
        self.assertIs(out[3], vision)
        self.assertIs(out[4], text)
        plain.assert_called_once_with(clip, 'selected')
        clip.tokenize.assert_called_once()

    def test_disabled_reference_uses_normal_text_path(self):
        clip = Mock(); cond = [[torch.ones(1, 8, 96), {}]]
        with patch.object(nodes.CLIPTextEncode, 'encode', return_value=(cond,)) as plain:
            result = donut_prompt.DonutPromptConditioning().encode(clip, 'same', '', 'same', native_reference_a=torch.ones(1, 1, 1, 3))['result']
        clip.tokenize.assert_not_called(); self.assertIs(result[3], cond)
        # Full and face share the same ordinary conditioning cache entry.
        self.assertEqual(plain.call_count, 1)

    def test_missing_images_and_wrong_encoder_fail_clearly(self):
        node = donut_prompt.DonutPromptConditioning()
        with self.assertRaisesRegex(ValueError, 'Add an image'):
            node.encode(Mock(), '', '', '', native_reference_enabled=True)
        clip = Mock(); clip.tokenize.return_value = {'l': []}
        with self.assertRaisesRegex(ValueError, 'Qwen3-VL'):
            node.encode(clip, '', '', '', native_reference_enabled=True, native_reference_a=torch.zeros(1, 2, 2, 3))

if __name__ == '__main__': unittest.main()

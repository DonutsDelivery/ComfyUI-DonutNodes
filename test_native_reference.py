import unittest
from unittest.mock import Mock, patch

import torch
import nodes
import donut_prompt
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
    def test_native_vision_preserves_all_tokens_and_metadata_positives_only(self):
        clip = Mock()
        clip.tokenize.return_value = {'qwen3vl_4b': object()}
        vision = [[torch.ones(1, 320, 96), {'vision_metadata': 'keep'}]]
        clip.encode_from_tokens_scheduled.return_value = vision
        ordinary = [[torch.ones(1, 8, 96), {}]]
        image = torch.zeros(1, 80, 120, 3)
        with patch.object(nodes.CLIPTextEncode, 'encode', return_value=(ordinary,)) as plain:
            out = donut_prompt.DonutPromptConditioning().encode(clip, 'same', '', 'same', native_reference_enabled=True, native_reference_a=image)['result']
        self.assertIs(out[3], vision); self.assertIs(out[4], vision); self.assertIs(out[6], ordinary)
        self.assertEqual(out[3][0][0].shape[1], 320)
        self.assertEqual(out[3][0][1]['vision_metadata'], 'keep')
        plain.assert_called_once_with(clip, 'same')
        clip.tokenize.assert_called_once()
        from comfy.text_encoders.krea2 import KREA2_TEMPLATE
        args, kwargs = clip.tokenize.call_args
        self.assertIn('<|image_pad|>', args[0]); self.assertEqual(kwargs['llama_template'], KREA2_TEMPLATE)
        self.assertIs(kwargs['images'][0], image)

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

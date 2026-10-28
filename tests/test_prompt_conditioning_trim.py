"""DonutPromptConditioning skips encodes for outputs nothing consumes."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import types
import unittest
from unittest.mock import patch


class Clip:
    def __init__(self):
        self.encoded = []

    def tokenize(self, text):
        return {"qwen3vl_4b": [[(ord(c), 1) for c in text]]}

    def encode_from_tokens_scheduled(self, tokens):
        self.encoded.append("".join(chr(i) for i, _ in tokens["qwen3vl_4b"][0]))
        return [[SimpleNamespace(text=self.encoded[-1]), {}]]


def _load():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("prompt_trim_under_test", root / "donut_prompt.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _nodes():
    nodes = types.ModuleType("nodes")

    class Encode:
        def encode(self, clip, text):
            return (clip.encode_from_tokens_scheduled(clip.tokenize(text)),)

    class Zero:
        def zero_out(self, cond):
            return ([[SimpleNamespace(text="zero:" + cond[0][0].text), {}]],)

    nodes.CLIPTextEncode, nodes.ConditioningZeroOut = Encode, Zero
    return nodes


def _prompt(*consumers):
    """API prompt where node "7" feeds the listed output slots."""
    prompt = {"7": {"class_type": "DonutPromptConditioning", "inputs": {"clip": ["1", 1]}}}
    for index, slot in enumerate(consumers):
        prompt[f"c{index}"] = {"class_type": "Sink", "inputs": {"value": ["7", slot]}}
    return prompt


class PromptConditioningTrimTests(unittest.TestCase):
    def setUp(self):
        self.module = _load()

    def _encode(self, running_prompt, unique_id="7", **options):
        clip = Clip()
        with patch.dict(sys.modules, {"nodes": _nodes()}), \
                patch.object(self.module, "_running_prompt", lambda: running_prompt), \
                patch.object(self.module, "wildcard_roots", lambda: ()):
            result = self.module.DonutPromptConditioning().encode(
                clip, "FACE", "SCENE", "NEG", separator=" ", unique_id=unique_id, **options)["result"]
        return clip.encoded, result

    def test_consumed_outputs_reads_links_to_this_node_only(self):
        prompt = _prompt(3, 6, 6)
        prompt["other"] = {"inputs": {"value": ["8", 4], "text": "7"}}
        self.assertEqual(self.module.consumed_outputs(prompt, "7"), (3, 6))
        self.assertEqual(self.module.consumed_outputs(_prompt(), "7"), ())
        self.assertIsNone(self.module.consumed_outputs(prompt, "missing"))
        self.assertIsNone(self.module.consumed_outputs(None, "7"))
        self.assertIsNone(self.module.consumed_outputs(prompt, None))

    def test_only_consumed_conditioning_is_encoded(self):
        encoded, result = self._encode(_prompt(3, 6))
        self.assertEqual(encoded, ["FACE SCENE", "NEG"])
        self.assertEqual(result[3][0][0].text, "FACE SCENE")
        self.assertIsNone(result[4])
        self.assertEqual(result[6][0][0].text, "NEG")
        self.assertEqual(result[5][0][0].text, "zero:NEG")
        self.assertEqual(result[:3], ("FACE SCENE", "FACE", ""))

    def test_zeroed_negative_alone_still_encodes_the_negative(self):
        encoded, result = self._encode(_prompt(1, 5))
        self.assertEqual(encoded, ["NEG"])
        self.assertIsNone(result[3])
        self.assertEqual(result[5][0][0].text, "zero:NEG")

    def test_unknown_prompt_or_id_encodes_everything(self):
        for running, unique_id in ((None, "7"), (_prompt(3), None), (_prompt(3), "missing")):
            with self.subTest(running=running is not None, unique_id=unique_id):
                encoded, result = self._encode(running, unique_id)
                self.assertEqual(encoded, ["FACE SCENE", "FACE", "NEG"])
                self.assertTrue(all(value is not None for value in result))

    def test_variance_skips_a_missing_side_and_keeps_face_seed(self):
        calls = []

        def vary(conditioning, settings):
            calls.append((conditioning[0][0].text, settings["seed"]))
            return conditioning

        variance = sys.modules[self.module.enhance_prompt_pair.__module__]
        with patch.object(variance, "apply_seed_variance", vary):
            encoded, result = self._encode(_prompt(4), variance_enabled=True, variance_seed=10)
        self.assertEqual(encoded, ["FACE"])
        self.assertIsNone(result[3])
        self.assertEqual(calls, [("FACE", 11)])

    def test_is_changed_tracks_consumed_slots(self):
        def changed(running):
            with patch.object(self.module, "_running_prompt", lambda: running), \
                    patch.object(self.module, "directory_fingerprint", lambda: "dirs"), \
                    patch.object(self.module, "wildcard_roots", lambda: ()):
                return self.module.DonutPromptConditioning.IS_CHANGED(
                    face="FACE", scene="SCENE", negative="NEG", unique_id="7")

        self.assertEqual(changed(_prompt(3, 6)), changed(_prompt(6, 3)))
        self.assertNotEqual(changed(_prompt(3, 6)), changed(_prompt(3, 4, 6)))
        self.assertIsInstance(changed(_prompt(3))[-1], tuple)


if __name__ == "__main__":
    unittest.main()

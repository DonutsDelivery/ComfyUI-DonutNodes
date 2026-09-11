"""Tests for the path-wide Donut LoRA execution-mode policy."""

import copy
import types
import unittest

import donut_lora_execution as module


class ExecutionModeTests(unittest.TestCase):
    def test_requested_mode_is_used_when_model_has_no_selection(self):
        model = types.SimpleNamespace(model_options={})
        self.assertEqual(
            module.resolve_execution_mode(model, "Experimental bypass"),
            "Experimental bypass",
        )

    def test_existing_selection_is_authoritative(self):
        model = types.SimpleNamespace(model_options={
            module.MODEL_OPTIONS_KEY: "Comfy patches",
        })
        self.assertEqual(
            module.resolve_execution_mode(model, "Experimental bypass"),
            "Comfy patches",
        )

    def test_publish_mode_is_local_to_the_model_options(self):
        original = types.SimpleNamespace(model_options={"other": 1})
        clone = types.SimpleNamespace(model_options=copy.deepcopy(original.model_options))
        module.publish_execution_mode(clone, "Experimental bypass")
        self.assertNotIn(module.MODEL_OPTIONS_KEY, original.model_options)
        self.assertEqual(
            clone.model_options[module.MODEL_OPTIONS_KEY],
            "Experimental bypass",
        )

    def test_publish_mode_creates_options_when_output_has_none(self):
        output = types.SimpleNamespace()
        self.assertIs(module.publish_execution_mode(output, "Comfy patches"), output)
        self.assertEqual(output.model_options[module.MODEL_OPTIONS_KEY], "Comfy patches")

    def test_invalid_mode_still_fails_without_inherited_selection(self):
        with self.assertRaisesRegex(ValueError, "Unknown LoRA execution mode"):
            module.resolve_execution_mode(types.SimpleNamespace(model_options={}), "invalid")


if __name__ == "__main__":
    unittest.main()

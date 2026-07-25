import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).with_name("DonutPromptReceiver.py")


def load_prompt_receiver_module():
    module_name = "donut_prompt_receiver_under_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


class PromptReceiverStartupTests(unittest.TestCase):
    def test_import_does_not_start_http_server(self):
        with mock.patch("http.server.HTTPServer") as http_server:
            load_prompt_receiver_module()

        http_server.assert_not_called()

    def test_node_execution_starts_configured_server(self):
        module = load_prompt_receiver_module()
        module.PromptReceiverServer._instance = None

        with mock.patch.object(module.PromptReceiverServer, "start") as start:
            module.DonutPromptReceiver().run(
                9123,
                workflow={"receiver": {"inputs": {}}},
                node_id="receiver",
            )

        start.assert_called_once_with(9123)


if __name__ == "__main__":
    unittest.main()

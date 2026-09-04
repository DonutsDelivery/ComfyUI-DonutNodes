import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODULE_PATH = ROOT / "DonutModelMergeKrea2.py"


class PatcherInjection:
    def __init__(self, inject, eject):
        self.inject = inject
        self.eject = eject


class WeightAdapterBase:
    pass


class BypassInjectionManager:
    pass


def load_module():
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    weight_adapter = types.ModuleType("comfy.weight_adapter")
    weight_adapter.WeightAdapterBase = WeightAdapterBase
    weight_adapter.BypassInjectionManager = BypassInjectionManager
    patcher_extension = types.ModuleType("comfy.patcher_extension")
    patcher_extension.PatcherInjection = PatcherInjection
    comfy.weight_adapter = weight_adapter
    comfy.patcher_extension = patcher_extension

    injected = {
        "comfy": comfy,
        "comfy.weight_adapter": weight_adapter,
        "comfy.patcher_extension": patcher_extension,
    }
    old = {name: sys.modules.get(name) for name in injected}
    sys.modules.update(injected)
    try:
        spec = importlib.util.spec_from_file_location(
            "donut_krea2_merge_lora_composition_tested", MODULE_PATH
        )
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


class InjectionCompositionTests(unittest.TestCase):
    def test_merge_injection_is_falsey_for_lora_compatibility_guard(self):
        injections = module._ComposableInjectionList([object()])
        self.assertFalse(injections)
        self.assertFalse(any({module._INJECTION_KEY: injections}.values()))
        self.assertEqual(len(injections), 1)

    def test_clone_copy_preserves_falsey_compatibility_marker(self):
        injections = module._ComposableInjectionList([object()])
        cloned = injections.copy()
        self.assertIsInstance(cloned, module._ComposableInjectionList)
        self.assertFalse(cloned)
        self.assertEqual(len(cloned), 1)

    def test_model_merge_still_detects_compatible_injection_as_runtime_injection(self):
        model = types.SimpleNamespace(
            is_injected=False,
            injections={
                module._INJECTION_KEY: module._ComposableInjectionList([object()])
            },
        )
        self.assertTrue(module._has_runtime_injections(model))


if __name__ == "__main__":
    unittest.main()

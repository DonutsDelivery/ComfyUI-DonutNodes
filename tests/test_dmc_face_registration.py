"""Import the real pack initializer with synthetic ComfyUI/dependency modules."""
import ast
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "synthetic_donutnodes_face_registration"


def load(name, path, *, package=False):
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(ROOT)] if package else None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FaceRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.nodes = types.ModuleType("nodes")
        self.nodes.NODE_CLASS_MAPPINGS = {}
        sys.modules["nodes"] = self.nodes
        self.calls = []
        calls = self.calls

        class Upstream:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"image": ("IMAGE",)},
                        "optional": {"edit_mode": ("BOOLEAN", {"default": False})}}

            def doit(self, image, *, edit_mode=False, face_reference=None, wildcard="", **options):
                calls.append((image, edit_mode, face_reference, wildcard, options))
                return ("synthetic legacy output",) * 6

        self.upstream = Upstream
        # The real initializer and dependency-isolation helper execute. Every
        # other component is a stub, so no torch, private settings or routes load.
        dependency = load(PACKAGE + ".donut_dependencies", ROOT / "donut_dependencies.py")
        dependency.installed_versions = lambda: {}
        tree = ast.parse((ROOT / "__init__.py").read_text())
        values = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                  and node.targets[0].id in {"_NODE_MODULES", "_REQUIRED_OVERRIDES"}}
        for name in (*values["_NODE_MODULES"], "shared.server_routes", "donut_wildcards", "donut_model_downloads"):
            if name == "dmc_face_detailer":
                continue
            module = types.ModuleType(PACKAGE + "." + name)
            module.NODE_CLASS_MAPPINGS = {
                key: type(key, (), {}) for key in values["_REQUIRED_OVERRIDES"].get(name, ())}
            module.NODE_DISPLAY_NAME_MAPPINGS = {}
            if name == "DonutFaceDetailer":
                module.NODE_CLASS_MAPPINGS = {"DonutFaceDetailer": Upstream}
            sys.modules[module.__name__] = module
        self.dependency = dependency

    def import_pack(self):
        return load(PACKAGE, ROOT / "__init__.py", package=True)

    # AC: @create-multiple-face-prompts ac-existing-backend-node
    def test_actual_initializer_exports_wrapper_before_comfy_registers_upstream(self):
        pack = self.import_pack()
        self.assertEqual(self.nodes.NODE_CLASS_MAPPINGS, {})
        self.assertEqual(self.dependency.IMPORT_FAILURES, {})
        self.assertIn("DMCFaceDetailer", pack.NODE_CLASS_MAPPINGS)
        self.assertEqual(pack.NODE_DISPLAY_NAME_MAPPINGS["DMCFaceDetailer"],
                         "DMC Face Detailer · Ordered Characters")
        # ComfyUI merges the pack mappings after importing its initializer.
        self.nodes.NODE_CLASS_MAPPINGS.update(pack.NODE_CLASS_MAPPINGS)
        schema = pack.NODE_CLASS_MAPPINGS["DMCFaceDetailer"].INPUT_TYPES()
        self.assertEqual(schema["optional"]["face_prompts_json"][0], "STRING")
        self.assertEqual(schema["optional"]["require_single_face"], ("BOOLEAN", {"default": False}))
        self.assertNotIn("face_prompts_json", self.upstream.INPUT_TYPES()["optional"])

    # AC: @create-multiple-face-prompts ac-existing-backend-node
    def test_wrapper_can_import_before_upstream_then_resolve_registered_schema(self):
        wrapper = load(PACKAGE + ".dmc_face_detailer", ROOT / "dmc_face_detailer.py")
        self.assertEqual(self.nodes.NODE_CLASS_MAPPINGS, {})
        pack = self.import_pack()
        self.nodes.NODE_CLASS_MAPPINGS.update(pack.NODE_CLASS_MAPPINGS)
        self.assertIs(pack.NODE_CLASS_MAPPINGS["DMCFaceDetailer"], wrapper.DMCFaceDetailer)
        self.assertIs(wrapper.DMCFaceDetailer.upstream(), self.upstream)
        self.assertEqual(wrapper.DMCFaceDetailer.INPUT_TYPES()["optional"]["face_prompts_json"][0], "STRING")

    # AC: @create-multiple-face-prompts ac-single-face-compatible
    def test_empty_plan_delegates_original_inputs_without_altering_upstream(self):
        pack = self.import_pack()
        self.nodes.NODE_CLASS_MAPPINGS.update(pack.NODE_CLASS_MAPPINGS)
        node = pack.NODE_CLASS_MAPPINGS["DMCFaceDetailer"]()
        image, reference = object(), object()
        self.assertEqual(node.doit(image, edit_mode=True, face_reference=reference,
                                   wildcard="synthetic legacy wildcard", seed=123),
                         ("synthetic legacy output",) * 6)
        self.assertEqual(self.calls, [(image, True, reference, "synthetic legacy wildcard", {"seed": 123})])


if __name__ == "__main__":
    unittest.main()

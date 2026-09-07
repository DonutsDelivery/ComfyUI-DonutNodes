"""CPU dependency/registration regressions; no ComfyUI server or GPU required.

Registration tests execute the real initializer with stand-ins for node modules.
The shared utility tests below use real Torch/NumPy/Pillow when available.
"""

import ast
import builtins
from contextlib import redirect_stdout
import importlib.util
import io
import math
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_donut_dependency_tests", ROOT / "donut_dependencies.py")
deps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deps)


class DependencyTests(unittest.TestCase):
    def setUp(self):
        deps.IMPORT_FAILURES.clear()
        self.logging = mock.patch.object(deps.LOGGER, "error")
        self.logging.start()
        self.addCleanup(self.logging.stop)

    def test_failed_module_does_not_remove_unrelated_nodes(self):
        existing = object()
        classes, names = {"working": existing}, {"working": "Working"}
        with mock.patch.object(deps.importlib, "import_module", side_effect=ImportError("numpy.core.multiarray failed to import")):
            result = deps.import_component("test", "Broken", classes, names)
        self.assertIsNone(result)
        self.assertIs(classes["working"], existing)
        self.assertEqual(names, {"working": "Working"})
        self.assertTrue(deps.IMPORT_FAILURES["Broken"]["numpy_abi_error"])

    def test_all_common_numpy_abi_signatures_are_recognized(self):
        for message in deps.ABI_MARKERS:
            with self.subTest(message=message):
                record = deps._failure_record("test", ImportError(message))
                self.assertTrue(record["numpy_abi_error"])

    def test_chained_abi_failure_retains_original_traceback(self):
        try:
            try:
                raise AttributeError("_ARRAY_API not found")
            except AttributeError as error:
                raise ImportError("outer import failed") from error
        except ImportError as error:
            record = deps._failure_record("cv2", error)
        self.assertTrue(record["numpy_abi_error"])
        self.assertIn("outer import failed", record["traceback"])
        self.assertIn("_ARRAY_API not found", record["traceback"])
        self.assertIn("test_chained_abi_failure", record["traceback"])

    def test_programming_errors_are_not_mislabelled_numpy_errors(self):
        with mock.patch.object(deps.importlib, "import_module", side_effect=ValueError("bad node schema")):
            deps.import_component("test", "Broken", {})
        record = deps.IMPORT_FAILURES["Broken"]
        self.assertFalse(record["numpy_abi_error"])
        self.assertIn("ValueError", record["traceback"])

    def test_successful_import_merges_mappings_and_clears_stale_failure(self):
        deps.IMPORT_FAILURES["Good"] = {"old": "failure"}
        module = types.SimpleNamespace(NODE_CLASS_MAPPINGS={"A": object()}, NODE_DISPLAY_NAME_MAPPINGS={"A": "Alias"})
        classes, names = {}, {}
        with mock.patch.object(deps.importlib, "import_module", return_value=module):
            self.assertIs(deps.import_component("test", "Good", classes, names), module)
        self.assertIs(classes["A"], module.NODE_CLASS_MAPPINGS["A"])
        self.assertEqual(names["A"], "Alias")
        self.assertNotIn("Good", deps.IMPORT_FAILURES)

    def test_display_mapping_is_optional(self):
        module = types.SimpleNamespace(NODE_CLASS_MAPPINGS={"A": object()})
        classes, names = {}, {}
        with mock.patch.object(deps.importlib, "import_module", return_value=module):
            deps.import_component("test", "Good", classes, names)
        self.assertIn("A", classes)
        self.assertEqual(names, {})

    def test_invalid_display_mapping_does_not_partially_register(self):
        module = types.SimpleNamespace(NODE_CLASS_MAPPINGS={"A": object()}, NODE_DISPLAY_NAME_MAPPINGS=None)
        classes = {}
        with mock.patch.object(deps.importlib, "import_module", return_value=module):
            deps.import_component("test", "Bad", classes, {})
        self.assertEqual(classes, {})

    def test_failed_required_override_is_not_silently_replaced_by_base(self):
        classes, names = {"Override": object(), "keep": object()}, {"Override": "Old"}
        with mock.patch.object(deps.importlib, "import_module", side_effect=ImportError("missing dependency")):
            deps.import_component("test", "OverrideModule", classes, names, overrides=("Override",))
        self.assertNotIn("Override", classes)
        self.assertNotIn("Override", names)
        self.assertIn("keep", classes)

    def test_missing_required_export_fails_closed(self):
        module = types.SimpleNamespace(NODE_CLASS_MAPPINGS={"wrong": object()})
        classes = {"Required": object()}
        with mock.patch.object(deps.importlib, "import_module", return_value=module):
            deps.import_component("test", "BadOverride", classes, {}, overrides=("Required",))
        self.assertEqual(classes, {})
        self.assertIn("missing required overrides", deps.IMPORT_FAILURES["BadOverride"]["error"])

    def test_interrupt_exit_and_out_of_memory_are_not_swallowed(self):
        for error in (KeyboardInterrupt(), SystemExit(), MemoryError()):
            with self.subTest(error=type(error).__name__):
                with mock.patch.object(deps.importlib, "import_module", side_effect=error):
                    with self.assertRaises(type(error)):
                        deps.import_component("test", "Broken", {})

    def test_non_node_component_can_register_routes(self):
        module = types.SimpleNamespace()
        with mock.patch.object(deps.importlib, "import_module", return_value=module):
            self.assertIs(deps.import_component("test", "routes"), module)

    def test_metadata_checks_do_not_import_binary_packages(self):
        def version(name):
            if name == "numpy":
                return "2.5.1"
            raise deps.metadata.PackageNotFoundError(name)
        with mock.patch.object(deps.metadata, "version", side_effect=version), mock.patch.object(
            deps.importlib, "import_module", side_effect=AssertionError("binary import")
        ):
            versions = deps.installed_versions()
        self.assertEqual(versions["numpy"], "2.5.1")
        self.assertIsNone(versions["scipy"])

    def test_opencv_conflict_checks_all_four_distributions(self):
        for other in deps.OPENCV_DISTRIBUTIONS[1:]:
            with self.subTest(other=other):
                warning = deps.opencv_conflict({"opencv-python": "4.13", other: "4.13"})
                self.assertIn(other, warning)
                self.assertIn("share the cv2 namespace", warning)
        self.assertIsNone(deps.opencv_conflict({"opencv-python": "4.13"}))
        self.assertIsNone(deps.opencv_conflict({}))

    def test_cv2_is_returned_without_changing_image_arithmetic(self):
        cv2 = object()
        with mock.patch.object(deps.importlib, "import_module", return_value=cv2) as loader:
            self.assertIs(deps.require_cv2("Masks"), cv2)
        loader.assert_called_once_with("cv2")

    def test_cv2_failure_is_actionable_and_preserves_cause(self):
        for error in (ImportError("missing"), AttributeError("_ARRAY_API not found"),
                      ValueError("numpy.dtype size changed"), OSError("DLL load failed")):
            with self.subTest(error=error):
                with mock.patch.object(deps.importlib, "import_module", side_effect=error):
                    with self.assertRaisesRegex(RuntimeError, "Donut Dependency Check") as raised:
                        deps.require_cv2("Mask dilation/erosion")
                self.assertIs(raised.exception.__cause__, error)
                self.assertIn("Mask dilation/erosion", str(raised.exception))

    def test_cv2_helper_does_not_catch_unrelated_type_error(self):
        with mock.patch.object(deps.importlib, "import_module", side_effect=TypeError("bug")):
            with self.assertRaises(TypeError):
                deps.require_cv2("Mask")

    def test_unknown_probe_never_starts_a_process(self):
        with mock.patch.object(deps.subprocess, "run", side_effect=AssertionError("spawned")):
            with self.assertRaises(KeyError):
                deps.probe_dependency("arbitrary code")

    def test_probes_use_running_python_no_shell_and_bounded_timeout(self):
        result = types.SimpleNamespace(returncode=0, stdout="", stderr="")
        with mock.patch.object(deps.subprocess, "run", return_value=result) as runner:
            checked = deps.probe_dependency("numpy", timeout=7)
        arguments, options = runner.call_args
        self.assertEqual(arguments[0][0], sys.executable)
        self.assertIn("-c", arguments[0])
        self.assertNotIn("pip", arguments[0])
        self.assertFalse(options["shell"])
        self.assertEqual(options["timeout"], 7)
        self.assertEqual(options["env"]["MPLBACKEND"], "Agg")
        self.assertEqual(checked["status"], "ok")

    def test_probes_preserve_portable_no_user_site_flag(self):
        result = types.SimpleNamespace(returncode=0, stdout="", stderr="")
        with mock.patch.object(deps.sys, "flags", types.SimpleNamespace(no_user_site=True)), mock.patch.object(
            deps.subprocess, "run", return_value=result
        ) as runner:
            deps.probe_dependency("numpy")
        self.assertEqual(runner.call_args.args[0][:2], [sys.executable, "-s"])

    def test_timeout_and_native_crash_are_reported(self):
        with mock.patch.object(deps.subprocess, "run", side_effect=subprocess.TimeoutExpired(["python"], 2)):
            self.assertEqual(deps.probe_dependency("numpy", timeout=2)["status"], "timeout")
        with mock.patch.object(deps.subprocess, "run", return_value=types.SimpleNamespace(returncode=-11, stdout="", stderr="crashed")):
            result = deps.probe_dependency("numpy")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["returncode"], -11)

    def test_spawn_failure_is_reported(self):
        with mock.patch.object(deps.subprocess, "run", side_effect=OSError("permission denied")):
            self.assertEqual(deps.probe_dependency("numpy")["status"], "error")

    def test_success_exit_with_numpy_abi_warning_is_not_a_pass(self):
        with mock.patch.object(deps.subprocess, "run", return_value=types.SimpleNamespace(
            returncode=0, stdout="", stderr="A module that was compiled using NumPy 1.x cannot be run"
        )):
            self.assertEqual(deps.probe_dependency("numpy")["status"], "failed")

    def test_report_default_never_spawns_processes(self):
        with mock.patch.object(deps, "installed_versions", return_value={"numpy": "2.5.1"}), mock.patch.object(
            deps.subprocess, "run", side_effect=AssertionError("subprocess")
        ):
            report = deps.dependency_report()
        self.assertIn(sys.executable, report)
        self.assertIn("NumPy 2.x version range cannot", report)
        self.assertIn("Import checks not run", report)
        self.assertIn("pip", report)

    def test_report_probes_are_fixed_allowlist(self):
        with mock.patch.object(deps, "installed_versions", return_value={}), mock.patch.object(
            deps, "probe_dependency", return_value={"status": "ok", "returncode": 0, "output": ""}
        ) as probe:
            report = deps.dependency_report(True)
        self.assertEqual([c.args[0] for c in probe.call_args_list], list(deps._PROBES))
        self.assertIn("torch_numpy: ok", report)

    def test_diagnostic_node_has_visible_text_and_string_output(self):
        with mock.patch.object(deps, "dependency_report", return_value="test report"), redirect_stdout(io.StringIO()):
            output = deps.DonutDependencyCheck().check()
        self.assertEqual(output["ui"]["text"], ["test report"])
        self.assertEqual(output["result"], ("test report",))
        self.assertTrue(deps.DonutDependencyCheck.OUTPUT_NODE)
        self.assertTrue(math.isnan(deps.DonutDependencyCheck.IS_CHANGED()))

    def test_requirements_have_binary_floors_but_no_global_numpy_pin(self):
        text = (ROOT / "requirements.txt").read_text()
        requirements = [line for line in text.splitlines() if line and not line.startswith("#")]
        self.assertIn("opencv-python-headless>=4.10.0.84", requirements)
        self.assertIn("scipy>=1.13.1", requirements)
        self.assertIn("matplotlib>=3.9.2", requirements)
        self.assertFalse(any(line.lower().startswith(("numpy", "torch")) for line in requirements))


class BootstrapTests(unittest.TestCase):
    def boot(self, failing=()):
        package = "_donut_boot_test"
        spec = importlib.util.spec_from_file_location(package, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
        module = importlib.util.module_from_spec(spec)
        loaded = []
        base_class, replacement_class = type("Base", (), {}), type("Override", (), {})

        def importer(name, parent):
            name = name[1:]
            loaded.append(name)
            if name in failing:
                raise ImportError("numpy.core.multiarray failed to import")
            mapping = {name: type(name, (), {})}
            if name == "DonutKrea2FusionControl":
                mapping = {"DonutKrea2FusionControl": base_class}
            elif name == "DonutKrea2FusionPreset":
                mapping = {"DonutKrea2FusionControl": replacement_class}
            elif name == "donut_lora_nodes":
                mapping = {"DonutApplyLoRAStack": base_class, "DonutFiller": type("Filler", (), {})}
            elif name == "DonutSafeApplyLoRAStack":
                mapping = {"DonutApplyLoRAStack": replacement_class}
            return types.SimpleNamespace(NODE_CLASS_MAPPINGS=mapping, NODE_DISPLAY_NAME_MAPPINGS={})

        with mock.patch.dict(sys.modules, {package: module, package + ".donut_dependencies": deps}), mock.patch.object(
            deps.importlib, "import_module", side_effect=importer
        ), mock.patch.object(deps, "installed_versions", return_value={}), mock.patch.object(
            deps.LOGGER, "error"
        ), mock.patch.object(deps.LOGGER, "warning"):
            spec.loader.exec_module(module)
        return module, loaded, replacement_class

    def test_real_initializer_preserves_order_and_required_overrides(self):
        module, loaded, replacement = self.boot()
        self.assertEqual(loaded, ["shared.server_routes", *module._NODE_MODULES])
        self.assertEqual(len(module._NODE_MODULES), 38)
        for key in ("DonutApplyLoRAStack", "DonutKrea2FusionControl"):
            self.assertIs(module.NODE_CLASS_MAPPINGS[key], replacement)
        self.assertIn("DonutDependencyCheck", module.NODE_CLASS_MAPPINGS)
        self.assertEqual(module.WEB_DIRECTORY, "./web")
        self.assertEqual(module.NODE_DISPLAY_NAME_MAPPINGS["DonutFiller"], "Donut Filler (Model + CLIP)")

    def test_broken_analysis_and_server_routes_do_not_hide_lora_or_fusion(self):
        module, loaded, _ = self.boot({"DonutFrequencyAnalysis", "DonutSpectralNoiseSharpener", "shared.server_routes"})
        for key in ("DonutApplyLoRAStack", "DonutKrea2FusionControl", "DonutImageAdjust", "DonutDependencyCheck"):
            self.assertIn(key, module.NODE_CLASS_MAPPINGS)
        self.assertEqual(len(deps.IMPORT_FAILURES), 3)
        self.assertIn("DonutImageAdjust", loaded)

    def test_failed_fusion_or_safe_lora_override_does_not_offer_wrong_base_node(self):
        module, _, _ = self.boot({"DonutKrea2FusionPreset", "DonutSafeApplyLoRAStack"})
        self.assertNotIn("DonutKrea2FusionControl", module.NODE_CLASS_MAPPINGS)
        self.assertNotIn("DonutApplyLoRAStack", module.NODE_CLASS_MAPPINGS)
        self.assertIn("DonutDependencyCheck", module.NODE_CLASS_MAPPINGS)
        self.assertIn("DonutImageAdjust", module.NODE_CLASS_MAPPINGS)

    def test_diagnostics_survive_when_all_real_components_fail(self):
        source = ast.parse((ROOT / "__init__.py").read_text())
        modules = next(ast.literal_eval(node.value) for node in source.body if isinstance(node, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "_NODE_MODULES" for t in node.targets))
        module, _, _ = self.boot(set(modules) | {"shared.server_routes"})
        self.assertEqual(set(module.NODE_CLASS_MAPPINGS), {"DonutDependencyCheck"})

    def test_deprecated_flags_still_apply_to_healthy_nodes(self):
        module, _, _ = self.boot()
        self.assertTrue(module.NODE_CLASS_MAPPINGS["DonutAutoGamma"].DEPRECATED)
        self.assertTrue(module.NODE_DISPLAY_NAME_MAPPINGS["DonutAutoGamma"].endswith("(DEPRECATED)"))


class SharedUtilsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            import numpy
            import PIL
        except ImportError as error:
            raise unittest.SkipTest(f"Real utility tests need Torch, NumPy and Pillow: {error}")
        cls.torch = torch
        cls.numpy = numpy

    def load_utils(self):
        package_name = "_donut_utils_test"
        package = types.ModuleType(package_name)
        package.__path__ = [str(ROOT)]
        libs = types.ModuleType(package_name + ".libs")
        libs.__path__ = [str(ROOT / "libs")]
        folders = types.ModuleType("folder_paths")
        folders.folder_names_and_paths = {}
        folders.add_model_folder_path = mock.Mock()
        context = mock.patch.dict(sys.modules, {
            package_name: package, libs.__name__: libs,
            package_name + ".donut_dependencies": deps,
            "folder_paths": folders,
        })
        context.start()
        self.addCleanup(context.stop)
        spec = importlib.util.spec_from_file_location(libs.__name__ + ".utils", ROOT / "libs" / "utils.py")
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result, folders

    def test_real_shared_utils_and_folder_setup_work_with_opencv_import_blocked(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name == "cv2" or name.startswith("cv2."):
                raise ImportError("_ARRAY_API not found")
            return original(name, *args, **kwargs)
        with mock.patch("builtins.__import__", side_effect=guarded):
            utils, folders = self.load_utils()
            utils.add_folder_path_and_extensions("lbw_models", ["/models/lbw"], {".safetensors"})
            self.assertEqual(utils.empty_latent().shape, (1, 4, 8, 8))
            self.assertEqual(folders.folder_names_and_paths["lbw_models"], (["/models/lbw"], {".safetensors"}))

    def test_dilation_fails_at_use_not_import_with_clear_message(self):
        utils, _ = self.load_utils()
        with mock.patch.object(deps.importlib, "import_module", side_effect=ImportError("_ARRAY_API not found")):
            with self.assertRaisesRegex(RuntimeError, "Mask dilation/erosion"):
                utils.dilate_mask(self.torch.ones((1, 4, 4)), 1)

    def test_dilation_erosion_arithmetic_matches_previous_opencv_path(self):
        try:
            import cv2
        except ImportError as error:
            self.skipTest(f"Working OpenCV not installed: {error}")
        utils, _ = self.load_utils()
        mask = self.torch.tensor([[[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]]])
        for factor in (-2., -1., -.5, 0., .5, 1., 2.):
            with self.subTest(factor=factor):
                k = abs(int(factor * 2) + 1)
                kernel = self.numpy.ones((k, k), self.numpy.uint8)
                operation = cv2.dilate if factor > 0 else cv2.erode
                reference = self.torch.stack([self.torch.from_numpy(operation(m, kernel, iterations=1)) for m in mask.numpy()])
                self.assertTrue(self.torch.equal(utils.dilate_mask(mask, factor), reference))


if __name__ == "__main__":
    unittest.main()

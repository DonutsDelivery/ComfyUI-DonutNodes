import os
import tempfile
import types
import unittest
from pathlib import Path

from donut_save_path import get_model_save_path


class RejectingFolderPaths:
    @staticmethod
    def get_save_image_path(filename_prefix, output_dir):
        raise Exception("Saving image outside the output folder is not allowed")


class AcceptingFolderPaths:
    @staticmethod
    def get_save_image_path(filename_prefix, output_dir):
        return (output_dir, filename_prefix, 7, "", filename_prefix)


class DonutSavePathTests(unittest.TestCase):
    def test_normal_comfy_path_is_used_unchanged(self):
        result = get_model_save_path(AcceptingFolderPaths, "ComfyUI", "/tmp/output")
        self.assertEqual(result, ("/tmp/output", "ComfyUI", 7, "", "ComfyUI"))

    def test_existing_output_symlink_can_point_to_external_ssd(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "output"
            ssd = root / "external-ssd"
            output.mkdir()
            ssd.mkdir()
            (output / "diffusion_models").symlink_to(ssd, target_is_directory=True)

            result = get_model_save_path(
                RejectingFolderPaths,
                "diffusion_models/KreaMerge",
                str(output),
            )

            folder, filename, counter, subfolder, _ = result
            self.assertEqual(Path(folder), output / "diffusion_models")
            self.assertEqual(Path(folder).resolve(), ssd.resolve())
            self.assertEqual(filename, "KreaMerge")
            self.assertEqual(counter, 1)
            self.assertEqual(subfolder, "diffusion_models")

    def test_counter_scans_existing_safetensors_through_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "output"
            ssd = root / "ssd"
            output.mkdir()
            ssd.mkdir()
            (output / "models").symlink_to(ssd, target_is_directory=True)
            (ssd / "Thing_00001_.safetensors").write_bytes(b"")
            (ssd / "Thing_00009_.safetensors").write_bytes(b"")

            result = get_model_save_path(
                RejectingFolderPaths,
                "models/Thing",
                str(output),
            )
            self.assertEqual(result[2], 10)

    def test_parent_traversal_is_rejected_even_with_symlink_present(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "output"
            ssd = root / "ssd"
            output.mkdir()
            ssd.mkdir()
            (output / "models").symlink_to(ssd, target_is_directory=True)

            with self.assertRaises(Exception):
                get_model_save_path(
                    RejectingFolderPaths,
                    "../escape/model",
                    str(output),
                )

    def test_absolute_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "output"
            output.mkdir()
            with self.assertRaises(Exception):
                get_model_save_path(
                    RejectingFolderPaths,
                    "/mnt/elsewhere/model",
                    str(output),
                )

    def test_no_symlink_does_not_weaken_comfy_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "output"
            output.mkdir()
            with self.assertRaisesRegex(Exception, "outside the output"):
                get_model_save_path(
                    RejectingFolderPaths,
                    "ordinary/subdir/model",
                    str(output),
                )


if __name__ == "__main__":
    unittest.main()

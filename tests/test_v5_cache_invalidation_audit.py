"""Known disk-file invalidation gap from the V5 model-cache audit.

Uses the real IS_CHANGED body, isolated from ComfyUI startup. The expected
failure records an unfixed issue; it is not a successful generation test.
"""
import ast
import hashlib
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class EditFileInvalidation(unittest.TestCase):
    @unittest.expectedFailure
    def test_replacing_edit_lora_invalidates_cached_edit_studio_output(self):
        tree = ast.parse((ROOT / 'DonutEditStudio.py').read_text())
        node = next(item for item in tree.body
                    if isinstance(item, ast.ClassDef) and item.name == 'DonutEditStudio')
        method = next(item for item in node.body
                      if isinstance(item, ast.FunctionDef) and item.name == 'IS_CHANGED')
        method.decorator_list = []
        with tempfile.TemporaryDirectory() as directory:
            lora = Path(directory) / 'edit.safetensors'
            lora.write_bytes(b'old adapter')
            environment = {
                'hashlib': hashlib,
                'directory_fingerprint': lambda: (),
                '_reference_path': lambda _: Path(directory) / 'absent-reference',
            }
            exec(compile(ast.Module(body=[method], type_ignores=[]),
                         'DonutEditStudio.py', 'exec'), environment)
            def signature():
                return environment['IS_CHANGED'](
                    None, enabled=True, lora_name=str(lora), lora_strength=1.0)
            before = signature()
            lora.write_bytes(b'replacement adapter with different content and length')
            self.assertNotEqual(signature(), before)


if __name__ == '__main__':
    unittest.main()

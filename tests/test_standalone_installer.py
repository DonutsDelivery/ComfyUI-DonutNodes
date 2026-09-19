import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from urllib.request import Request

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('installer', ROOT / 'model-installer/install_models.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class Tests(unittest.TestCase):
    def test_catalog(self):
        self.assertTrue(installer.catalog(ROOT / 'model_sources.json'))

    def test_download_verify_skip_and_preserve_conflicts(self):
        data = b'model bytes'
        entry = dict(folder='vae', filename='test.safetensors', size=len(data),
                     sha256=hashlib.sha256(data).hexdigest(), url='https://huggingface.co/test')
        class Opener:
            calls = 0
            def open(self, *args, **kwargs):
                self.calls += 1
                return io.BytesIO(data)
        with tempfile.TemporaryDirectory() as temp:
            root, opener = Path(temp), Opener()
            installer.install(entry, root, opener, {})
            installer.install(entry, root, opener, {})
            self.assertEqual(opener.calls, 1)
            target = root / 'models/vae/test.safetensors'
            target.write_bytes(b'other')
            with self.assertRaises(ValueError):
                installer.install(entry, root, opener, {})
            self.assertEqual(target.read_bytes(), b'other')
            target.unlink()
            with self.assertRaises(ValueError):
                installer.install(dict(entry, sha256='0' * 64), root, opener, {})
            self.assertFalse(target.exists())
            self.assertEqual(list(root.rglob('*.part')), [])

    def test_redirect_does_not_leak_token(self):
        req = Request('https://huggingface.co/test', headers={'Authorization': 'Bearer secret'})
        redirected = installer.Redirects().redirect_request(req, None, 302, '', {}, 'https://cdn.hf.co/file')
        self.assertFalse(redirected.has_header('Authorization'))
        with self.assertRaises(ValueError):
            installer.Redirects().redirect_request(req, None, 302, '', {}, 'https://example.com/file')

    def test_comfyui_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'ComfyUI'
            root.mkdir()
            (root / 'main.py').touch()
            (root / 'models').mkdir()
            self.assertEqual(installer.find_root(temp), root.resolve())


if __name__ == '__main__':
    unittest.main()

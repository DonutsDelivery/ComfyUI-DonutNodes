import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PKG = "_donut_download_test"
package = types.ModuleType(PKG); package.__path__ = [str(ROOT)]
shared = types.ModuleType(PKG + ".shared"); shared.__path__ = [str(ROOT / "shared")]
sys.modules[PKG] = package; sys.modules[shared.__name__] = shared
server = types.ModuleType("server")
server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=types.SimpleNamespace(
    get=lambda route: lambda fn: fn, post=lambda route: lambda fn: fn)))
with patch.dict(sys.modules, {"server":server}):
    spec = importlib.util.spec_from_file_location(PKG + ".donut_model_downloads", ROOT / "donut_model_downloads.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)


class Response:
    status_code = 200
    is_redirect = False
    headers = {}

    def __init__(self, content=b"model bytes", callback=None):
        self.content, self.callback = content, callback

    def iter_content(self, size):
        yield self.content
        if self.callback:
            self.callback()

    def close(self):
        pass


class ModelDownloadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.models = self.root / "models"; self.models.mkdir()
        self.entry = {"folder":"loras", "filename":"krea2/test.safetensors", "url":"https://huggingface.co/donut/model/resolve/revision/test.safetensors",
                      "sha256":hashlib.sha256(b"model bytes").hexdigest(), "size":11}
        self.manager = mod.ModelDownloads()
        patches = [
            patch.object(mod.folder_paths, "get_folder_paths", return_value=[str(self.models)]),
            patch.object(mod.folder_paths, "get_user_directory", return_value=str(self.root / "user")),
            patch.object(mod.folder_paths, "folder_names_and_paths", {"loras":([str(self.models)], {".safetensors"})}),
            patch.object(mod.folder_paths, "get_filename_list", side_effect=lambda folder:[p.relative_to(self.models).as_posix() for p in self.models.rglob("*.safetensors")]),
            patch.object(mod.folder_paths, "get_full_path", side_effect=lambda folder,name:str(self.models / name) if (self.models / name).is_file() else None),
            patch.object(mod, "get_civitai_api_key", return_value=""),
        ]
        for mocked in patches:
            mocked.start(); self.addCleanup(mocked.stop)

    def run_models(self, refs=None):
        self.manager.update(results=[], completed=0, running=True)
        self.manager.run(refs or [{"folder":"loras", "name":self.entry["filename"]}], [self.entry])
        return self.manager.snapshot()

    def test_missing_download_is_verified_and_second_run_does_not_download(self):
        with patch.object(mod.requests, "get", return_value=Response()) as get:
            result = self.run_models()
            self.assertEqual(result["results"][0]["status"], "downloaded")
            self.assertEqual((self.models / self.entry["filename"]).read_bytes(), b"model bytes")
            self.assertEqual(self.run_models()["results"][0]["status"], "verified")
            get.assert_called_once()
        self.assertEqual(list(self.models.rglob("*.part")), [])

    def test_same_size_wrong_hash_is_not_installed(self):
        with patch.object(mod.requests, "get", return_value=Response(b"wrong bytes")):
            result = self.run_models()
        self.assertIn("SHA-256", result["results"][0]["message"])
        self.assertFalse((self.models / self.entry["filename"]).exists())
        self.assertEqual(list(self.models.rglob("*.part")), [])

    def test_short_or_oversized_response_is_not_installed(self):
        for body in (b"short", b"longer than expected model bytes"):
            with self.subTest(body=body), patch.object(mod.requests, "get", return_value=Response(body)):
                self.assertEqual(self.run_models()["results"][0]["status"], "error")
            self.assertFalse((self.models / self.entry["filename"]).exists())
            self.assertEqual(list(self.models.rglob("*.part")), [])

    def test_existing_wrong_file_is_preserved(self):
        path = self.models / self.entry["filename"]; path.parent.mkdir(); path.write_bytes(b"wrong bytes")
        with patch.object(mod.requests, "get") as get:
            result = self.run_models(); get.assert_not_called()
        self.assertEqual(result["results"][0]["status"], "error")
        self.assertEqual(path.read_bytes(), b"wrong bytes")

    def test_renamed_matching_file_is_reused_by_hash(self):
        path = self.models / "renamed.safetensors"; path.write_bytes(b"model bytes")
        with patch.object(mod.requests, "get") as get:
            result = self.run_models(); get.assert_not_called()
        self.assertEqual(result["results"][0]["resolved_name"], "renamed.safetensors")

    def test_cache_does_not_accept_changed_file(self):
        path = self.models / "renamed.safetensors"; path.write_bytes(b"model bytes")
        self.assertEqual(self.manager.digest(path), self.entry["sha256"])
        path.write_bytes(b"wrong bytes")
        self.assertNotEqual(self.manager.digest(path), self.entry["sha256"])

    def test_cancel_does_not_publish_partial_file(self):
        with patch.object(mod.requests, "get", return_value=Response(callback=self.manager.cancelled.set)):
            result = self.run_models()
        self.assertEqual(result["state"], "cancelled")
        self.assertFalse((self.models / self.entry["filename"]).exists())
        self.assertEqual(list(self.models.rglob("*.part")), [])

    def test_destination_created_during_download_is_not_overwritten(self):
        path = self.models / self.entry["filename"]
        with patch.object(mod.requests, "get", return_value=Response(callback=lambda:path.write_bytes(b"other file"))):
            result = self.run_models()
        self.assertEqual(result["results"][0]["status"], "error")
        self.assertEqual(path.read_bytes(), b"other file")

    def test_unlisted_missing_model_never_downloads(self):
        with patch.object(mod.requests, "get") as get:
            result = self.run_models([{"folder":"loras", "name":"unknown.safetensors"}]); get.assert_not_called()
        self.assertIn("No upstream source", result["results"][0]["message"])

    def test_paths_and_redirects_stay_in_allowed_locations(self):
        for name in ("../outside", "/etc/passwd", "C:\\outside", "a/../outside"):
            with self.assertRaises(ValueError): mod.model_name(name)
        outside = self.root / "outside"; outside.mkdir()
        (self.models / "krea2").symlink_to(outside)
        with self.assertRaises(ValueError): self.manager.destination(self.entry)
        for url in ("http://huggingface.co/file", "https://huggingface.co.evil.example/file", "https://127.0.0.1/file", "https://user@huggingface.co/file"):
            with self.assertRaises(ValueError): mod.download_url(url)

    def test_credentials_are_not_forwarded_to_download_cdn(self):
        redirect = Response(); redirect.is_redirect = True
        redirect.headers = {"Location":"https://us.aws.cdn.hf.co/model"}
        with patch.dict(mod.os.environ, {"HF_TOKEN":"test-secret"}), patch.object(mod.requests, "get", side_effect=[redirect, Response()]) as get:
            self.assertEqual(self.run_models()["results"][0]["status"], "downloaded")
        self.assertEqual(get.call_args_list[0].kwargs["headers"]["Authorization"], "Bearer test-secret")
        self.assertNotIn("Authorization", get.call_args_list[1].kwargs["headers"])

    def test_private_redirect_is_rejected_before_following_it(self):
        redirect = Response(); redirect.is_redirect = True
        redirect.headers = {"Location":"https://127.0.0.1/private"}
        with patch.object(mod.requests, "get", return_value=redirect) as get:
            self.assertEqual(self.run_models()["results"][0]["status"], "error")
            get.assert_called_once()

    def test_registry_controls_source_and_requires_complete_integrity_metadata(self):
        catalog = mod.load_catalog()
        self.assertGreaterEqual(len(catalog), 9)
        for entry in catalog:
            self.assertEqual(len(entry["sha256"]), 64)
            self.assertGreater(entry["size"], 0)
        self.assertIsNone(mod.catalog_entry(catalog, "loras", "unknown.safetensors"))


if __name__ == "__main__": unittest.main()

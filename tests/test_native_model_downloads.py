"""Offline integration tests: actual downloader and staging, tiny mocked files.

No network, model downloads, CUDA inference or running ComfyUI are required.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativeModelDownloads(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.models = self.root / 'models'
        self.nodes = types.ModuleType('nodes')
        self.nodes.NODE_CLASS_MAPPINGS = {name: object for name in (
            'SeedVR2Preprocess', 'SeedVR2Conditioning', 'SeedVR2PostProcessing',
            'LoadBackgroundRemovalModel', 'RemoveBackground', 'SAM3_Detect')}
        self.folders = types.ModuleType('folder_paths')
        self.folders.models_dir = str(self.models)
        self.folders.filename_list_cache = {}
        self.folders.folder_names_and_paths = {
            name: ([str(self.models / name)], {'.safetensors'})
            for name in ('diffusion_models', 'vae', 'background_removal', 'loras', 'checkpoints')}
        self.folders.get_folder_paths = lambda name: self.folders.folder_names_and_paths[name][0]
        self.folders.get_user_directory = lambda: str(self.root / 'user')
        self.folders.get_filename_list = lambda folder: [
            path.relative_to(self.models / folder).as_posix()
            for path in (self.models / folder).rglob('*.safetensors')]
        self.folders.get_full_path = lambda folder, name: (
            str(self.models / folder / name) if (self.models / folder / name).is_file() else None)
        package = types.ModuleType('_donut_native_download_test'); package.__path__ = [str(ROOT)]
        shared = types.ModuleType(package.__name__ + '.shared'); shared.__path__ = []
        config = types.ModuleType(shared.__name__ + '.config'); config.get_civitai_api_key = lambda: ''
        server = types.ModuleType('server')
        server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(
            routes=types.SimpleNamespace(get=lambda _: lambda f: f, post=lambda _: lambda f: f)))
        mods = {package.__name__: package, shared.__name__: shared, config.__name__: config,
                'folder_paths': self.folders, 'server': server, 'nodes': self.nodes}
        context = patch.dict(sys.modules, mods); context.start(); self.addCleanup(context.stop)
        self.mod = load_module(package.__name__ + '.donut_model_downloads', ROOT / 'donut_model_downloads.py')
        self.manager = self.mod.ModelDownloads()
        self.catalog = self.mod.load_catalog()
        self.native = [entry for entry in self.catalog if entry.get('requires_nodes')]

    def toy_entries(self):
        entries = []
        for entry in self.native:
            payload = entry['filename'].encode()
            entries.append({**entry, 'size': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()})
        return entries

    def run_downloads(self, entries, refs=None):
        self.manager.update(results=[], completed=0, running=True)
        refs = refs or [{'folder': entry['folder'], 'name': entry['filename']} for entry in entries]
        self.manager.run(refs, entries)
        return self.manager.snapshot()

    def test_catalog_has_verified_native_models(self):
        self.assertEqual(len(self.native), 5)
        expected = {
            'seedvr2_3b_int8_convrot.safetensors': (3458259704, 'c3dec8bcc5916843a8a858572970597462e1f2dc598d6dfd818f6cd40f53a157'),
            'seedvr2_7b_int8_convrot.safetensors': (8334897976, '5aa0d25fc9d35e449b659d0c9a5dcb22e2a4fa04032101b95a39da42b32c1be6'),
            'seedvr2_ema_vae_fp16.safetensors': (501324814, '20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1'),
            'birefnet.safetensors': (444473596, '9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154'),
            'sam3.1_multiplex_fp16.safetensors': (1745546848, '9ba99c92703c2e8b4f47de2d34a539bb8e18923049e238b780d70dbe6368eb03'),
        }
        for entry in self.native:
            self.assertEqual((entry['size'], entry['sha256']), expected[entry['filename']])
            self.assertRegex(entry['url'], r'^https://huggingface.co/Comfy-Org/(SeedVR2|BiRefNet|sam3\.1)/resolve/[a-f0-9]{40}/')
        self.assertIn('background_removal', self.mod.FOLDERS)

    def test_download_install_verify_and_reuse_all_native_model_types(self):
        entries = self.toy_entries()
        self.folders.filename_list_cache = {entry['folder']: 'stale' for entry in entries}
        def respond(url, **kwargs):
            entry = next(entry for entry in entries if entry['url'] == url)
            return Response(entry['filename'].encode())
        with patch.object(self.mod.requests, 'get', side_effect=respond) as get:
            result = self.run_downloads(entries)
            self.assertEqual([item['status'] for item in result['results']], ['downloaded'] * len(entries))
            self.assertEqual(get.call_count, len(entries))
            for entry in entries:
                self.assertEqual((self.models / entry['folder'] / entry['filename']).read_bytes(), entry['filename'].encode())
            self.assertEqual(self.folders.filename_list_cache, {})
            result = self.run_downloads(entries)
            self.assertEqual([item['status'] for item in result['results']], ['verified'] * len(entries))
            self.assertEqual(get.call_count, len(entries))
        self.assertEqual(list(self.models.rglob('*.part')), [])

    def test_authentication_error_identifies_provider_and_local_configuration(self):
        response = Response(b'')
        response.status_code = 401
        entry = {**self.toy_entries()[0], 'url': 'https://civitai.com/api/download/models/1'}
        with patch.object(self.mod.requests, 'get', return_value=response):
            result = self.run_downloads([entry])
        failure = result['results'][0]
        self.assertEqual(failure['status'], 'error')
        self.assertIn('civitai.api_key', failure['message'])
        self.assertIn('HTTP 401', failure['message'])
        self.assertEqual(list(self.models.rglob('*.safetensors')), [])

    def test_missing_native_nodes_stop_before_network_or_file_write(self):
        self.nodes.NODE_CLASS_MAPPINGS.clear()
        with patch.object(self.mod.requests, 'get') as get:
            result = self.run_downloads(self.toy_entries()); get.assert_not_called()
        self.assertEqual(len(result['results']), len(self.native))
        for item in result['results']:
            self.assertEqual(item['status'], 'error')
            self.assertIn('Update ComfyUI and restart', item['message'])
        self.assertFalse(self.models.exists())

    def test_older_core_background_folder_error_explains_update(self):
        self.nodes.NODE_CLASS_MAPPINGS.clear()
        del self.folders.folder_names_and_paths['background_removal']
        with patch.object(self.mod.requests, 'get') as get:
            result = self.run_downloads(self.toy_entries()[-1:]); get.assert_not_called()
        self.assertIn('Missing native nodes', result['results'][0]['message'])

    def test_unrelated_models_keep_working_without_native_nodes(self):
        self.nodes.NODE_CLASS_MAPPINGS.clear()
        entry = {'folder': 'loras', 'filename': 'legacy.safetensors', 'size': 3,
                 'sha256': hashlib.sha256(b'old').hexdigest(), 'url': 'https://huggingface.co/legacy/model'}
        with patch.object(self.mod.requests, 'get', return_value=Response(b'old')):
            self.assertEqual(self.run_downloads([entry])['results'][0]['status'], 'downloaded')

    def test_wrong_hash_never_publishes_file(self):
        entry = self.toy_entries()[-1]
        with patch.object(self.mod.requests, 'get', return_value=Response(b'x' * entry['size'])):
            result = self.run_downloads([entry])
        self.assertIn('SHA-256', result['results'][0]['message'])
        self.assertFalse((self.models / entry['folder'] / entry['filename']).exists())
        self.assertEqual(list(self.models.rglob('*.part')), [])

    def test_existing_different_file_is_preserved(self):
        entry = self.toy_entries()[-1]
        dest = self.models / entry['folder'] / entry['filename']
        dest.parent.mkdir(parents=True); dest.write_bytes(b'custom model')
        with patch.object(self.mod.requests, 'get') as get:
            result = self.run_downloads([entry]); get.assert_not_called()
        self.assertEqual(result['results'][0]['status'], 'error')
        self.assertEqual(dest.read_bytes(), b'custom model')

    def test_renamed_native_file_is_returned_for_widget_rebinding(self):
        entry = self.toy_entries()[-1]
        path = self.models / entry['folder'] / 'my/renamed.safetensors'
        path.parent.mkdir(parents=True); path.write_bytes(entry['filename'].encode())
        with patch.object(self.mod.requests, 'get') as get:
            result = self.run_downloads([entry]); get.assert_not_called()
        self.assertEqual(result['results'][0]['resolved_name'], 'my/renamed.safetensors')

    def test_extra_model_paths_destination_is_honored(self):
        entry = next(e for e in self.native if e['folder'] == 'background_removal')
        custom = self.root / 'shared-matting'
        self.folders.folder_names_and_paths['background_removal'][0].insert(0, str(custom))
        self.assertEqual(self.manager.destination(entry), custom / entry['filename'])

    def test_hf_xet_redirect_is_allowed_without_forwarding_credentials(self):
        entry = self.toy_entries()[0]
        redirect = Response(b''); redirect.is_redirect = True
        redirect.headers = {'Location': 'https://cas-bridge.xethub.hf.co/xet-file'}
        with patch.dict(self.mod.os.environ, {'HF_TOKEN': 'test-only'}), patch.object(
                self.mod.requests, 'get', side_effect=[redirect, Response(entry['filename'].encode())]) as get:
            result = self.run_downloads([entry])
        self.assertEqual(result['results'][0]['status'], 'downloaded')
        self.assertIn('Authorization', get.call_args_list[0].kwargs['headers'])
        self.assertNotIn('Authorization', get.call_args_list[1].kwargs['headers'])

    def test_malicious_paths_and_hostnames_remain_rejected(self):
        for name in ('../escape', '/tmp/model', 'C:\\model', 'ok/../model'):
            with self.subTest(name=name), self.assertRaises(ValueError): self.mod.model_name(name)
        for url in ('http://huggingface.co/file', 'https://huggingface.co.evil.test/file',
                    'https://127.0.0.1/file', 'https://user@hf.co/file'):
            with self.subTest(url=url), self.assertRaises(ValueError): self.mod.download_url(url)

    def test_native_requirement_schema_rejects_malformed_data(self):
        file = self.root / 'bad.json'
        for required in ('SeedVR2Preprocess', [None], ['not/a/node'], ['bad-name']):
            file.write_text(json.dumps({'version': 1, 'models': [{**self.native[0], 'requires_nodes': required}]}))
            with patch.object(self.mod, 'CATALOG_PATH', file), self.assertRaises(ValueError): self.mod.load_catalog()

    def test_start_accepts_new_folder_deduplicates_and_only_schedules_on_request(self):
        reference = {'folder': 'background_removal', 'name': 'birefnet.safetensors'}
        with patch.object(self.mod.threading, 'Thread') as thread:
            self.assertEqual(self.manager.snapshot()['state'], 'idle')
            thread.assert_not_called()
            self.manager.start([reference, reference])
            self.assertEqual(thread.call_args.kwargs['args'][0], [reference])
            thread.return_value.start.assert_called_once()


class Response:
    status_code = 200
    is_redirect = False
    headers = {}
    def __init__(self, data): self.data = data
    def iter_content(self, _size): yield self.data
    def close(self): pass


class RegistryStaging(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.mod = load_module('_registry_native_download_tests', ROOT / 'tools/prepare_registry.py')
        self.catalog = json.loads((ROOT / 'model_sources.json').read_text())

    def archive(self, catalog=None, extra=None, helper=None):
        archive = self.root / 'node.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            for name in ('assets/uncensorfix.f32', 'uncensorfix_weights.py', 'pyproject.toml',
                         'web/donut_model_downloads.js', 'web/donut_layout.js'):
                z.writestr(name, 'fixture')
            z.writestr('web/donut_model_requirements.js', helper if helper is not None else (ROOT / 'web/donut_model_requirements.js').read_bytes())
            z.writestr('model_sources.json', json.dumps(catalog or self.catalog))
            for name, data in (extra or {}).items(): z.writestr(name, data)
        return archive

    def test_staging_embeds_archive_catalog_and_replaces_only_download_panel(self):
        # Deliberately use a different archive catalog, proving no lookup of the
        # current working tree or a remote service supplies the release links.
        selected = {'version': 1, 'models': self.catalog['models'][-4:]}
        destination = self.root / 'stage'
        self.mod.prepare(self.archive(selected), destination)
        generated = (destination / 'web/donut_registry_catalog.js').read_text()
        actual = json.loads(generated.split('export const MODEL_CATALOG = ', 1)[1].removesuffix(';\n'))
        self.assertEqual(actual, selected['models'])
        self.assertFalse((destination / 'donut_model_downloads.py').exists())
        panel = (destination / 'web/donut_model_downloads.js').read_text()
        self.assertEqual(panel, (ROOT / 'distribution/registry/donut_model_downloads.js').read_text())
        self.assertIn('Download from upstream', panel)
        self.assertNotIn('/donut/models/download', panel)
        self.assertNotIn('fetch(', panel)
        self.assertIn('requires_nodes', generated)

    def test_packed_downloaders_and_standalone_installers_are_rejected(self):
        forbidden = ('donut_model_downloads.py', 'install_models.py', 'install-models.sh',
                     'install-models.bat', 'model-installer/install_models.py')
        for name in forbidden:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Non-registry file'):
                self.mod.prepare(self.archive(extra={name: 'forbidden'}), self.root / f"stage-{name.replace('/', '-')}")

    def test_unsafe_archive_paths_are_still_rejected(self):
        for name in ('../outside', '/absolute', 'bad\\path'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Unsafe archive path'):
                self.mod.prepare(self.archive(extra={name: 'no'}), self.root / 'stage')

    def test_stale_staging_directory_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'new staging directory'):
            self.mod.prepare(self.archive(), self.root)

    def test_unsafe_catalog_entries_are_rejected_before_extracting(self):
        original = self.catalog['models'][-1]
        changes = [{'url': 'javascript:alert(1)'}, {'url': 'https://user:pass@hf.co/file'},
                   {'filename': '../escape'}, {'folder': '../escape'}, {'sha256': 'bad'},
                   {'size': True}, {'requires_nodes': ['bad/node']}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.mod.prepare(self.archive({'version': 1, 'models': [{**original, **change}]}), self.root / 'stage')
            self.assertFalse((self.root / 'stage').exists())

    def test_stale_requirements_module_is_rejected_before_extraction(self):
        with self.assertRaisesRegex(ValueError, 'outdated'):
            self.mod.prepare(self.archive(helper='export function modelBindings() {}'), self.root / 'stage')
        self.assertFalse((self.root / 'stage').exists())

    def test_invalid_catalog_shapes_are_rejected(self):
        for value in ([], None, {'version': 2, 'models': []}, {'version': 1, 'models': 'bad'}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.mod.registry_catalog(value)

    def test_duplicate_catalog_entries_are_rejected(self):
        entry = self.catalog['models'][-1]
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.mod.registry_catalog({'version': 1, 'models': [entry, entry]})


if __name__ == '__main__': unittest.main()

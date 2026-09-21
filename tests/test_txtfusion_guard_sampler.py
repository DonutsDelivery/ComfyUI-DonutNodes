"""Dispatch/schema/cancellation tests; Comfy sampling is an interface double."""
import copy
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import donut_txtfusion_guard as guard


class Cancelled(BaseException):
    pass


class ParentSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'seed': ('INT', {}), 'steps': ('INT', {})},
                'optional': {'old_flag': ('BOOLEAN', {'default':False}),
                             'grounding_end_px': ('INT', {'default':1088})}}
    def sample(self, model, seed=123, steps=8, mode='simple', edit_mode=False,
               sda_enabled=False, **nag_options):
        self.received = (model, seed, steps, dict(nag_options))
        # In real DonutSampler the model at run_simple already carries NAG.
        return self.run_simple(model, seed, steps, nag_options)
    def run_simple(self, model, *args, **kwargs):
        self.executed = (model, args, kwargs)
        if getattr(self, 'cancel', False): raise Cancelled()
        if getattr(self, 'fail', False): raise RuntimeError('model failure')
        return self.executed


def load_sampler():
    package = types.ModuleType('_internal_guard_tests')
    package.__path__ = [str(ROOT)]
    parent = types.ModuleType(package.__name__ + '.donut_grounding_schedule')
    parent.DonutSampler = ParentSampler
    folder_paths = types.ModuleType('folder_paths')
    folder_paths.get_filename_list = lambda folder: ['same.safetensors', 'ignored.bin']
    folder_paths.get_full_path_or_raise = lambda folder, name: '/models/' + name
    overrides = {package.__name__:package, parent.__name__:parent, 'folder_paths':folder_paths,
                 package.__name__ + '.donut_txtfusion_guard':guard}
    context = patch.dict(sys.modules, overrides)
    context.start()
    spec = importlib.util.spec_from_file_location(package.__name__ + '.donut_txtfusion_guard_sampler', ROOT/'donut_txtfusion_guard_sampler.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, context


class SamplerTests(unittest.TestCase):
    def setUp(self):
        self.module, context = load_sampler()
        self.addCleanup(context.stop)
        self.sampler = self.module.DonutSampler()
        self.model, self.patched = object(), object()
        self.run = guard.GuardRun({'reference':object()}, 'file-digest')
        self.run.forward_calls = 2; self.run.contribution_calls = 4
    def call(self, **extra):
        options = dict(nag_enabled=True, nag_alpha=.45,
                       txtfusion_internal_guard=True, txtfusion_reference_checkpoint='same.safetensors')
        options.update(extra)
        return self.sampler.sample(self.model, **options)
    def test_off_is_exact_parent_dispatch_without_install(self):
        with patch.object(self.module, 'install_guard', side_effect=AssertionError('guard called')):
            result = self.call(txtfusion_internal_guard=False)
        self.assertIs(result[0], self.model)
        self.assertEqual(self.sampler.received, (self.model,123,8,{'nag_enabled':True,'nag_alpha':.45}))
        self.assertIsNone(self.module._REQUEST.get())
    def test_on_installs_only_at_first_pass_and_keeps_sampling_values(self):
        with patch.object(self.module, 'install_guard', return_value=(self.patched, self.run)) as install:
            result = self.call()
        install.assert_called_once_with(self.model, '/models/same.safetensors')
        self.assertIs(result[0], self.patched)
        self.assertEqual(result[1], (123,8,{'nag_enabled':True,'nag_alpha':.45}))
        self.assertEqual(self.run.references, {})
        self.assertIsNone(self.module._REQUEST.get())
    def test_cancel_cleans_context_and_references_without_touching_model(self):
        self.sampler.cancel = True
        with patch.object(self.module, 'install_guard', return_value=(self.patched, self.run)):
            with self.assertRaises(Cancelled): self.call()
        self.assertEqual(self.run.references, {})
        self.assertIsNone(self.module._REQUEST.get())
    def test_model_exception_also_cleans(self):
        self.sampler.fail = True
        with patch.object(self.module, 'install_guard', return_value=(self.patched, self.run)):
            with self.assertRaisesRegex(RuntimeError, 'model failure'): self.call()
        self.assertEqual(self.run.references, {})
        self.assertIsNone(self.module._REQUEST.get())
    def test_logging_exception_cannot_leak_references(self):
        with patch.object(self.module, 'install_guard', return_value=(self.patched, self.run)), \
             patch.object(self.module.LOGGER, 'info', side_effect=RuntimeError('logger failure')):
            with self.assertRaisesRegex(RuntimeError, 'logger'): self.call()
        self.assertEqual(self.run.references, {})
        self.assertIsNone(self.module._REQUEST.get())
    def test_unused_requested_guard_is_not_reported_as_success(self):
        self.run.contribution_calls = 0
        with patch.object(self.module, 'install_guard', return_value=(self.patched, self.run)):
            with self.assertRaisesRegex(RuntimeError, 'never executed'): self.call()
        self.assertEqual(self.run.references, {})
    def test_unsupported_modes_fail_without_running(self):
        for settings in [dict(mode='advanced'), dict(edit_mode=True), dict(sda_enabled=True),
                         dict(nag_enabled=False), dict(nag_alpha=0), dict(nag_phi=0)]:
            with self.subTest(settings=settings), self.assertRaises(ValueError): self.call(**settings)
        self.assertIsNone(self.module._REQUEST.get())
    def test_reference_filename_must_come_from_model_catalog(self):
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            self.call(txtfusion_reference_checkpoint='../../other.safetensors')
    def test_optional_schema_appends_without_changing_existing_fields(self):
        old = ParentSampler.INPUT_TYPES()
        new = self.sampler.INPUT_TYPES()
        self.assertEqual(new['required'], old['required'])
        self.assertEqual(list(new['optional']), [*old['optional'], 'txtfusion_internal_guard', 'txtfusion_reference_checkpoint'])
        self.assertIs(new['optional']['txtfusion_internal_guard'][1]['default'], False)
        self.assertEqual(new['optional']['txtfusion_reference_checkpoint'][0], ['None','same.safetensors'])
        self.assertEqual(ParentSampler.INPUT_TYPES(), old)
    def test_new_widget_values_are_not_forwarded_as_nag_options(self):
        with patch.object(self.module, 'install_guard', return_value=(self.patched, self.run)):
            self.call()
        self.assertNotIn('txtfusion_internal_guard', self.sampler.received[3])
        self.assertNotIn('txtfusion_reference_checkpoint', self.sampler.received[3])
    def test_unrelated_or_upscale_execution_has_no_active_request(self):
        self.sampler.run_simple(self.model, 1, 2)
        self.assertIs(self.sampler.executed[0], self.model)
        self.assertIsNone(self.module._REQUEST.get())
    def test_no_active_adapters_passes_parent_without_reference(self):
        with patch.object(self.module, 'install_guard', return_value=(self.model,None)) as install:
            result = self.call(txtfusion_reference_checkpoint='None')
        install.assert_called_once_with(self.model, None)
        self.assertIs(result[0], self.model)


if __name__ == '__main__': unittest.main()

"""CPU contract/regression tests; no model download or GPU required.

Torch tensors and files are real. ComfyUI lifecycle interfaces are test doubles;
these tests do not claim full installed-ComfyUI or quantized GPU parity.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

import torch

ROOT = Path(__file__).resolve().parent
TARGET = "diffusion_model.blocks.0.weight"
SIGMAS = torch.tensor([1., .9567, .9, .82, .7, .5, .3, .1, 0.])


class Keyframe:
    def __init__(self, strength, start_percent, guarantee_steps):
        self.strength = strength
        self.start_percent = start_percent


class Keyframes:
    def __init__(self):
        self.keyframes = []
        self._current_index = 0
        self._current_strength = None
        self._current_keyframe = None

    def add(self, frame):
        self.keyframes.append(frame)
        self._current_keyframe = self.keyframes[0]

    @property
    def strength(self):
        return self._current_keyframe.strength if self._current_keyframe else 1.

    def reset(self):
        self._current_strength = None
        self._current_index = 0
        self._current_keyframe = self.keyframes[0]


class Hook:
    def __init__(self, strength_model, strength_clip):
        self._strength_model = strength_model
        self._strength_clip = strength_clip


class Group:
    def __init__(self):
        self.hooks = []

    def add(self, hook):
        self.hooks.append(hook)

    @staticmethod
    def combine_all_hooks(groups):
        result = Group()
        for group in groups:
            result.hooks.extend(group.hooks)
        return result


class ToyAdapter:
    def __init__(self, amount=0.5):
        # LoRA rank-one factors; actual linear arithmetic in the test manager.
        self.loaded_keys = {"some.lora_up.weight", "target.lora_up.weight"}
        self.weights = (torch.tensor([[amount]]), torch.ones(1, 1), 1., None, None, None)


class TrackingManager:
    fail_inject = False
    instances = []

    def __init__(self):
        self.adapters, self.hooks = {}, []
        self.__class__.instances.append(self)

    def add_adapter(self, key, adapter, strength):
        self.adapters[key] = (adapter, strength)

    def create_injections(self, root):
        for key, (adapter, strength) in self.adapters.items():
            module = root
            for part in key[:-7].split('.'):
                module = module[int(part)] if part.isdigit() else getattr(module, part)
            self.hooks.append([module, adapter, strength, None])

        def inject(_):
            for record in self.hooks:
                module, adapter, strength, _ = record
                previous = module.forward
                record[3] = previous
                def forward(x, _previous=previous, _adapter=adapter, _strength=strength):
                    up, down, alpha, *_ = _adapter.weights
                    return _previous(x) + torch.nn.functional.linear(
                        torch.nn.functional.linear(x, down), up) * (_strength * alpha / down.shape[0])
                module.forward = forward
                if self.fail_inject:
                    raise RuntimeError('partial injection failure')

        def eject(_):
            for module, _, _, previous in self.hooks:
                if previous is not None:
                    module.forward = previous
        return [types.SimpleNamespace(inject=inject, eject=eject)]

    def get_hook_count(self):
        return len(self.hooks)


class Model:
    def __init__(self):
        self.model = torch.nn.Module()
        d = torch.nn.Module()
        d.txtlayers, d.txtdim = 12, 2560
        d.txtfusion = torch.nn.Identity()
        d.blocks = torch.nn.ModuleList([torch.nn.Linear(1, 1, bias=False)])
        d.blocks[0].weight.data.fill_(2.)
        self.model.diffusion_model = d
        self.model_options = {'donut_lora_execution_mode': 'Comfy patches'}
        self.injections, self.wrappers = {}, {}

    def clone(self):
        result = Model()
        result.model = self.model  # Comfy clones share the base model, not its weights.
        result.model_options = deepcopy(self.model_options)
        result.injections = dict(self.injections)
        result.wrappers = dict(self.wrappers)
        return result

    def add_wrapper_with_key(self, kind, key, function):
        self.wrappers[(kind, key)] = function


class BaseSampler:
    calls = []

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model': ('MODEL',)}, 'optional': {'mode': (['simple', 'advanced', 'multi_model'],)}}

    def sample(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs['latent_image'], 'base sampler info'


def parameters(**updates):
    result = dict(seed=42, steps=8, cfg_start=1., cfg_halfway=1., cfg_end=1.,
                  halfway_step=4, sampler_name='euler', scheduler='simple',
                  positive=[[torch.ones(1, 2, 3), {}]], negative=[[torch.zeros(1, 2, 3), {}]],
                  latent_image={'samples': torch.ones(1, 1)}, denoise=1., turbo_mode=True)
    result.update(updates)
    return result


class SDATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        package = types.ModuleType('_sda_test_package')
        package.__path__ = [str(ROOT)]
        comfy = types.ModuleType('comfy')
        samplers = types.ModuleType('comfy.samplers')
        samplers.k_diffusion_sampling = types.SimpleNamespace()
        comfy.samplers = samplers
        hooks = types.ModuleType('comfy.hooks')
        hooks.HookKeyframeGroup, hooks.HookKeyframe = Keyframes, Keyframe
        hooks.WeightHook, hooks.HookGroup = Hook, Group
        utils = types.ModuleType('comfy.utils')
        utils.load_torch_file = Mock(return_value={'target.lora_up.weight': torch.ones(1, 1)})
        lora = types.ModuleType('comfy.lora')
        lora.model_lora_keys_unet = Mock(return_value={})
        lora.load_lora = Mock(return_value={TARGET: ToyAdapter()})
        convert = types.ModuleType('comfy.lora_convert')
        convert.convert_lora = lambda x: x
        ext = types.ModuleType('comfy.patcher_extension')
        ext.WrappersMP = types.SimpleNamespace(APPLY_MODEL='apply_model', SAMPLER_SAMPLE='sampler_sample')
        weight = types.ModuleType('comfy.weight_adapter')
        weight.BypassInjectionManager = TrackingManager
        paths = types.ModuleType('folder_paths')
        paths.get_full_path = Mock(return_value=None)
        for name, module in dict(hooks=hooks, utils=utils, lora=lora, lora_convert=convert,
                                 patcher_extension=ext, weight_adapter=weight).items():
            setattr(comfy, name, module)
        base = types.ModuleType(package.__name__ + '.DonutKSamplerCFGLinear')
        base.DonutSampler = BaseSampler
        policy = types.ModuleType(package.__name__ + '.donut_lora_execution')
        policy.resolve_execution_mode = lambda m: m.model_options['donut_lora_execution_mode']
        policy.publish_execution_mode = lambda m, mode: m.model_options.update(donut_lora_execution_mode=mode)
        safe = types.ModuleType(package.__name__ + '.DonutSafeApplyLoRAStack')
        safe._partition_bypass_targets = Mock(side_effect=lambda root, keys, components: (components, {}, {}))
        modules = {m.__name__: m for m in (package, comfy, samplers, hooks, utils, lora, convert, ext, weight, paths, base, policy, safe)}
        cls.modules = patch.dict(sys.modules, modules)
        cls.modules.start()
        cls.addClassCleanup(cls.modules.stop)
        def load(name):
            spec = importlib.util.spec_from_file_location(package.__name__ + '.' + name, ROOT / (name + '.py'))
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            cls.addClassCleanup(sys.modules.pop, spec.name, None)
            spec.loader.exec_module(module)
            setattr(package, name, module)
            return module
        cls.sampler_compat = load('donut_sda_sampler')
        cls.schedule = load('donut_sda_schedule')
        cls.kernels = samplers.k_diffusion_sampling
        cls.sda = load('donut_krea2_sda')
        cls.paths, cls.utils, cls.lora, cls.safe = paths, utils, lora, safe

    def setUp(self):
        BaseSampler.calls.clear()
        TrackingManager.instances.clear()
        TrackingManager.fail_inject = False
        self.utils.load_torch_file.reset_mock()
        self.lora.load_lora.return_value = {TARGET: ToyAdapter()}

    def test_reference_api_graph_and_euler_split_are_consistent(self):
        graph = json.loads((ROOT / "workflows/examples/krea2_sda_reference_api.json").read_text())
        for node in graph.values():
            for value in node["inputs"].values():
                if isinstance(value, list):
                    self.assertIn(value[0], graph)
                    self.assertIsInstance(value[1], int)
        baseline, native = graph["7"]["inputs"], graph["8"]["inputs"]
        self.assertFalse(baseline["sda_enabled"])
        self.assertTrue(native["sda_enabled"])
        self.assertEqual({k: v for k, v in baseline.items() if k != "sda_enabled"},
                         {k: v for k, v in native.items() if k != "sda_enabled"})
        first, last = graph["10"]["inputs"], graph["11"]["inputs"]
        self.assertEqual(first["noise_seed"], native["seed"])
        self.assertEqual(first["noise_seed"], last["noise_seed"])
        self.assertEqual((first["start_at_step"], first["end_at_step"],
                          last["start_at_step"], last["end_at_step"]), (0, 2, 2, 8))
        self.assertEqual(first["add_noise"], "enable")
        self.assertEqual(last["add_noise"], "disable")
        self.assertEqual(last["latent_image"], ["10", 0])
        self.assertEqual(graph["9"]["inputs"]["lora_name"], self.sda.SDA_LORA_NAME)

    def test_controls_append_without_changing_existing_schema(self):
        optional = self.sda.DonutSampler.INPUT_TYPES()['optional']
        self.assertEqual(list(optional), ['mode', 'sda_enabled', 'sda_strength'])
        self.assertFalse(optional['sda_enabled'][1]['default'])

    def test_disabled_is_exact_passthrough_without_import_or_file_io(self):
        model = object()
        for enabled, strength in [(False, 1.), (True, 0.)]:
            with self.subTest(enabled=enabled), patch.object(self.sda, '_schedule_module', side_effect=AssertionError):
                args = parameters(steps=27, turbo_mode=False, sampler_name='heun')
                actual = self.sda.DonutSampler().sample(model, sda_enabled=enabled, sda_strength=strength, **args)
                self.assertIs(actual[0], args['latent_image'])
                self.assertIs(BaseSampler.calls[-1]['model'], model)
                self.assertEqual(BaseSampler.calls[-1]['steps'], 27)
                self.assertEqual(BaseSampler.calls[-1]['sampler_name'], 'heun')

    def test_invalid_strength_and_recipes_fail_before_file_load(self):
        for strength in [float('nan'), float('inf'), -1., 2.1]:
            with self.subTest(strength=strength), self.assertRaises(ValueError):
                self.sda.DonutSampler().sample(Model(), sda_enabled=True, sda_strength=strength, **parameters())
        cases = [dict(turbo_mode=False), dict(steps=6), dict(denoise=.99), dict(denoise=float('nan')),
                 dict(edit_mode=True), dict(mode='multi_model'), dict(sampler_name='heun'),
                 dict(sampler_name='dpm_adaptive'), dict(sampler_name='dpmpp_sde'), dict(cfg_start=2),
                 dict(mode='advanced', start_at_step=1), dict(mode='advanced', end_at_step=7),
                 dict(mode='advanced', add_noise='disable'), dict(mode='advanced', return_with_leftover_noise='enable')]
        with patch.object(self.sda, '_sda_path', side_effect=AssertionError('file load should not occur')):
            for case in cases:
                with self.subTest(case=case), self.assertRaises(ValueError):
                    self.sda.DonutSampler().sample(Model(), sda_enabled=True, **parameters(**case))

    def test_simple_ignores_dormant_advanced_controls(self):
        self.sda._validate_sda_sampling(parameters(mode='simple', edit_mode=False, start_at_step=10, end_at_step=11,
                                                  return_with_leftover_noise='enable', add_noise='disable'),
                                        self.schedule.SDA_SAMPLERS)

    def test_sigma_gate_is_schedule_based_not_25_percent_or_call_count(self):
        actual = [self.schedule.sda_active(s, SIGMAS) for s in SIGMAS[:-1]]
        self.assertEqual(actual, [True, True] + [False] * 6)
        # Repeated evaluations and NAG/CFG branches cannot consume the gate.
        for _ in range(5):
            self.assertTrue(self.schedule.sda_active(SIGMAS[1], SIGMAS))
        self.assertFalse(self.schedule.sda_active(SIGMAS[2], SIGMAS))
        nonlinear = torch.tensor([1., .4, .01, .009, .008, .007, .006, .005, 0.])
        self.assertTrue(self.schedule.sda_active(.4, nonlinear))
        self.assertFalse(self.schedule.sda_active(.01, nonlinear))

    def test_invalid_runtime_schedules_and_mixed_sigmas_rejected(self):
        for sigmas in [None, SIGMAS[:-1], SIGMAS.repeat(2, 1), torch.zeros(9), SIGMAS.flip(0),
                       SIGMAS.clone().index_fill(0, torch.tensor([4]), float('nan'))]:
            with self.subTest(sigmas=sigmas), self.assertRaises(ValueError):
                self.schedule.sda_active(1., sigmas)
        with self.assertRaises(ValueError):
            self.schedule.sda_active(torch.tensor([1., .1]), SIGMAS)

    def test_native_hook_turns_off_at_third_sigma_and_resets(self):
        frames = self.schedule._SDAKeyframes()
        strengths, changes = [], []
        for sigma in SIGMAS[:-1]:
            changes.append(frames.prepare_current_keyframe(float(sigma), {'sample_sigmas': SIGMAS}))
            strengths.append(frames.strength)
        self.assertEqual(strengths, [1., 1.] + [0.] * 6)
        self.assertEqual(changes, [False, False, True] + [False] * 5)
        frames.reset()
        self.assertEqual(frames.strength, 1.)
        self.assertIsInstance(frames.clone(), self.schedule._SDAKeyframes)

    def test_regular_hooks_preserve_upstream_hooks_metadata_and_input(self):
        model = Model()
        upstream = Group()
        upstream.add('style hook')
        pos, neg = [[torch.ones(1, 1), {'hooks': upstream, 'keep': 'yes'}]], [[torch.zeros(1, 1), {'hooks': upstream}]]
        patched, out_pos, out_neg = self.schedule.prepare_sda(model, pos, neg, {TARGET: ToyAdapter()}, .75, 'Comfy patches')
        self.assertIs(patched.model, model.model)
        self.assertEqual(upstream.hooks, ['style hook'])
        self.assertIs(out_pos[0][0], pos[0][0])
        self.assertEqual(out_pos[0][1]['keep'], 'yes')
        self.assertIs(out_pos[0][1]['hooks'], out_neg[0][1]['hooks'])
        self.assertEqual(out_pos[0][1]['hooks'].hooks[0], 'style hook')
        hook = out_pos[0][1]['hooks'].hooks[1]
        self.assertEqual(hook._strength_model, .75)
        self.assertEqual(hook._strength_clip, 0.)
        self.assertFalse(model.wrappers)

    def test_bypass_scopes_only_sda_and_restores_existing_lora_forward(self):
        model = Model()
        layer = model.model.diffusion_model.blocks[0]
        base = layer.forward
        layer.forward = lambda x: base(x) + .25 * x  # existing ordinary LoRA
        upstream = layer.forward
        adapter = ToyAdapter()
        original_weights = adapter.weights
        wrapped = self.schedule._ScopedSDABypass({TARGET: adapter}, 1.)
        class Executor:
            class_obj = model.model
            def __call__(self, x, *args, **kwargs):
                return layer(x)
        for index, sigma in enumerate(SIGMAS[:-1]):
            output = wrapped(Executor(), torch.ones(1, 1), sigma, transformer_options={'sample_sigmas': SIGMAS})
            self.assertEqual(output.item(), 2.75 if index < 2 else 2.25)
            self.assertIs(layer.forward, upstream)
        self.assertIs(adapter.weights, original_weights)
        self.assertEqual(len(TrackingManager.instances), 2, 'no SDA adapters created after cutoff')
        self.assertTrue(all(instance.adapters[TARGET][0] is not adapter for instance in TrackingManager.instances))

    def test_bypass_cleans_up_model_failure_and_partial_injection(self):
        for partial in [False, True]:
            with self.subTest(partial=partial):
                model = Model()
                layer = model.model.diffusion_model.blocks[0]
                before = layer.forward
                TrackingManager.fail_inject = partial
                class Executor:
                    class_obj = model.model
                    def __call__(self, *args, **kwargs):
                        raise RuntimeError('denoiser interrupted')
                with self.assertRaises(RuntimeError):
                    self.schedule._ScopedSDABypass({TARGET: ToyAdapter()}, 1.)(
                        Executor(), torch.ones(1, 1), SIGMAS[0], transformer_options={'sample_sigmas': SIGMAS})
                self.assertEqual(layer.forward, before)

    def test_bypass_cannot_silently_fall_back_to_always_on_weights(self):
        with patch.object(self.safe, '_partition_bypass_targets', return_value=({}, {TARGET: []}, {TARGET: ('unsupported',)})):
            with self.assertRaisesRegex(ValueError, 'No always-on fallback'):
                self.schedule.prepare_sda(Model(), [], [], {TARGET: ToyAdapter()}, 1., 'Experimental bypass')

    def test_mapping_requires_full_coverage(self):
        model = Model()
        raw = {'some.lora_up.weight': torch.ones(1, 1)}
        self.assertEqual(list(self.schedule.map_sda_weights(model, raw)), [TARGET])
        for mapped in [{}, {'missing.weight': ToyAdapter()}, {TARGET: ToyAdapter(), 'extra.weight': ToyAdapter()}]:
            with self.subTest(mapped=mapped), patch.object(self.lora, 'load_lora', return_value=mapped):
                with self.assertRaises(ValueError):
                    self.schedule.map_sda_weights(model, raw)

    def test_missing_file_error_explains_download_modes(self):
        with patch.object(self.paths, 'get_full_path', return_value=None):
            with self.assertRaisesRegex(FileNotFoundError, 'never downloads'):
                self.sda._sda_path()

    def test_file_size_and_checksum_are_verified(self):
        good = b'checked SDA fixture'
        with tempfile.TemporaryDirectory() as tmp:
            filename = Path(tmp) / 'sda.safetensors'
            filename.write_bytes(good)
            with patch.object(self.sda, 'SDA_FILE_SIZE', len(good)), \
                 patch.object(self.sda, 'SDA_SHA256', hashlib.sha256(good).hexdigest()):
                self.sda._load_verified_lora(str(filename))
                self.utils.load_torch_file.assert_called_once_with(str(filename), safe_load=True)
                filename.write_bytes(b'x' * len(good))
                with self.assertRaisesRegex(ValueError, 'SHA-256'):
                    self.sda._load_verified_lora(str(filename))
                filename.write_bytes(b'x')
                with self.assertRaisesRegex(ValueError, 'file size'):
                    self.sda._load_verified_lora(str(filename))

    def test_node_runs_once_in_original_mode_with_cache_and_same_seed(self):
        good = b'checked SDA fixture'
        with tempfile.TemporaryDirectory() as tmp:
            filename = Path(tmp) / 'sda.safetensors'
            filename.write_bytes(good)
            with patch.object(self.paths, 'get_full_path', return_value=str(filename)), \
                 patch.object(self.sda, 'SDA_FILE_SIZE', len(good)), \
                 patch.object(self.sda, 'SDA_SHA256', hashlib.sha256(good).hexdigest()):
                for execution in ['Comfy patches', 'Experimental bypass']:
                    sampler = self.sda.DonutSampler()
                    model = Model()
                    model.model_options['donut_lora_execution_mode'] = execution
                    dormant = object()
                    for mode in ['simple', 'advanced']:
                        before = len(BaseSampler.calls)
                        params = parameters(mode=mode, sampler_name='er_sde', scheduler='bong_tangent', model_2=dormant)
                        _, info = sampler.sample(model, sda_enabled=True, **params)
                        self.assertEqual(len(BaseSampler.calls), before + 1)
                        call = BaseSampler.calls[-1]
                        self.assertEqual(call['mode'], mode)
                        self.assertEqual(call['seed'], 42)
                        self.assertEqual(call['sampler_name'], 'er_sde')
                        self.assertEqual(call['scheduler'], 'bong_tangent')
                        self.assertIs(call['latent_image'], params['latent_image'])
                        self.assertIs(call['model_2'], dormant)
                        self.assertIn('single run', info)
                        self.assertFalse(model.wrappers)
                self.assertEqual(self.utils.load_torch_file.call_count, 2, 'one CPU load per sampler, not per seed/mode')

    def test_schedule_guard_passes_same_objects_to_one_stateful_run(self):
        gate = self.schedule
        calls, active, old_states = [], [], []
        sentinel_noise, sentinel_latent = torch.ones(1), torch.zeros(1)
        def sample_er_sde():
            pass
        self.kernels.sample_er_sde = sample_er_sde
        class Executor:
            class_obj = types.SimpleNamespace(sampler_function=sample_er_sde, extra_options={})
            def __call__(self, *args):
                calls.append(args)
                state = 0.0
                # A history-bearing toy recurrence: detect restarts at the SDA boundary.
                for sigma in args[1][:-1]:
                    old_states.append(state)
                    enabled = gate.sda_active(sigma, args[1])
                    active.append(enabled)
                    state = state * .5 + (2. if enabled else 1.)
                return state
        args = (object(), SIGMAS, {'seed': 42}, object(), sentinel_noise, sentinel_latent, None, False)
        result = gate._sampling_guard(Executor(), *args)
        self.assertEqual(len(calls), 1)
        for actual, expected in zip(calls[0], args):
            self.assertIs(actual, expected)
        self.assertEqual(active, [True, True] + [False] * 6)
        self.assertEqual(old_states[2], 3., 'history crosses the SDA boundary')
        self.assertEqual(result, 2.015625)

    def test_runtime_guard_rejects_custom_solver_or_churn(self):
        delegate = Mock()
        delegate.class_obj = types.SimpleNamespace(sampler_function=lambda: None, extra_options={})
        with self.assertRaises(ValueError):
            self.schedule._sampling_guard(delegate, None, SIGMAS, {}, None, torch.ones(1))
        delegate.assert_not_called()
        def sample_euler():
            pass
        self.kernels.sample_euler = sample_euler
        delegate.class_obj = types.SimpleNamespace(sampler_function=sample_euler, extra_options={'s_churn': 1.})
        with self.assertRaisesRegex(ValueError, 'churn'):
            self.schedule._sampling_guard(delegate, None, SIGMAS, {}, None, torch.ones(1))

    def test_model_guards(self):
        with self.assertRaisesRegex(ValueError, 'Krea2'):
            self.sda._validate_model(object())
        model = Model()
        model.model.diffusion_model._orig_mod = object()
        with self.assertRaisesRegex(ValueError, 'compile'):
            self.sda._validate_model(model)
        model = Model()
        model.injections['donut_krea2_model_merge_bypass'] = []
        with self.assertRaisesRegex(ValueError, 'module-swap'):
            self.sda._validate_model(model)


if __name__ == '__main__':
    unittest.main()

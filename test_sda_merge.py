"""Real Donut hard-swap forwards with small CPU layers and scheduled SDA.

Loads the real merge builder/injection, source-plan resolver, SDA node and SDA
wrappers. ComfyUI patcher/adapter interfaces are doubles, not GPU lifecycle or
quantization validation. No adapter/checkpoint download is needed.
"""
from contextlib import ExitStack, contextmanager, redirect_stdout
from copy import copy, deepcopy
import importlib.util
import io
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
SOURCE = "donut_krea2_model_merge_sources"
INJECTION = "donut_krea2_model_merge_bypass"
KEYS = ("diffusion_model.txtfusion.projector.weight",
        "diffusion_model.blocks.0.weight", "diffusion_model.blocks.1.weight")
SIGMAS = torch.tensor([1., .97, .9, .82, .7, .5, .3, .1, 0.])


class Adapter:
    def __init__(self, amount=.2):
        self.weights = (torch.full((2, 1), amount), torch.ones(1, 3), 1., None, None, None)
        self.loaded_keys = {"up", "down"}

    def delta(self):
        up, down, alpha, *_ = self.weights
        return up @ down * (alpha / down.shape[0])

    def bypass_forward(self, original, x, *args, **kwargs):
        return original(x, *args, **kwargs) + F.linear(x, self.delta()) * self.multiplier


class Injection:
    def __init__(self, inject, eject):
        self.inject, self.eject = inject, eject


class Manager:
    roots = []
    fail_on_root = None
    fail_eject_root = None
    missing_hook_root = None

    def __init__(self):
        self.adapters, self.hooks = {}, []

    def add_adapter(self, key, adapter, strength=1.):
        self.adapters[key.removesuffix(".weight")] = (adapter, strength)

    def create_injections(self, root):
        self.roots.append(root)
        self.hooks = [(root.get_submodule(path), a, s) for path, (a, s) in self.adapters.items()]
        originals = []
        def inject(_):
            for layer, adapter, strength in self.hooks:
                original = layer.forward
                originals.append((layer, original))
                adapter.multiplier = strength
                layer.forward = lambda x, *a, _p=adapter, _f=original, **k: _p.bypass_forward(_f, x, *a, **k)
                if root is self.fail_on_root:
                    raise RuntimeError("source partial injection failure")
        def eject(_):
            for layer, original in reversed(originals):
                layer.forward = original
            originals.clear()
            if root is self.fail_eject_root:
                raise RuntimeError("source eject failure after restoration")
        self._root = root
        return [Injection(inject, eject)]

    def get_hook_count(self):
        return len(self.hooks) - (self._root is self.missing_hook_root)


class Patcher:
    """Shared roots, ordered patches, sources loaded before swap injection."""
    def __init__(self, value=2.):
        self.model = torch.nn.Module()
        d = torch.nn.Module()
        d.txtlayers, d.txtdim = 12, 2560
        d.txtfusion = torch.nn.Module()
        d.txtfusion.projector = torch.nn.Linear(3, 2)
        d.blocks = torch.nn.ModuleList([torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)])
        self.model.diffusion_model = d
        with torch.no_grad():
            for layer in (d.txtfusion.projector, *d.blocks):
                layer.weight.fill_(value)
                layer.bias.fill_(value / 10)
        self.raw = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.patches, self.injections, self.attachments, self.additional_models = {}, {}, {}, {}
        self.model_options, self.wrappers = {}, {}
        self.load_device = torch.device("cpu")
        self.is_injected = False

    def clone(self):
        n = copy(self)
        n.patches = {k: list(v) for k, v in self.patches.items()}
        n.injections = {k: v.copy() for k, v in self.injections.items()}
        n.attachments = dict(self.attachments)
        n.additional_models = {k: [m.clone() for m in v] for k, v in self.additional_models.items()}
        n.model_options = deepcopy(self.model_options)
        n.wrappers = dict(self.wrappers)
        n.is_injected = False
        return n

    def get_key_patches(self, prefix):
        return {k: self.effective(k) for k in self.raw if k.startswith(prefix)}

    def effective(self, key):
        value = self.raw[key].clone()
        for strength, delta, base_strength in self.patches.get(key, []):
            value = value * base_strength + (delta.delta() if isinstance(delta, Adapter) else delta) * strength
        return value

    def add_patches(self, values, strength_patch=1., strength_model=1.):
        for key, value in values.items():
            self.patches.setdefault(key, []).append((strength_patch, value, strength_model))
        return list(values)

    def set_injections(self, key, value): self.injections[key] = value
    def set_attachments(self, key, value): self.attachments[key] = value
    def set_additional_models(self, key, value): self.additional_models[key] = value
    def get_additional_models_with_key(self, key): return self.additional_models.get(key, [])
    def add_wrapper_with_key(self, kind, key, value): self.wrappers[(kind, key)] = value

    @contextmanager
    def activate(self):
        with ExitStack() as cleanup:
            for models in self.additional_models.values():
                for model in models:
                    cleanup.enter_context(model.activate())
            snapshot = {k: v.clone() for k, v in self.model.state_dict().items()}
            previous = getattr(self.model, "current_patcher", None)
            def restore():
                self.is_injected = False
                self.model.current_patcher = previous
                with torch.no_grad():
                    for key, value in self.model.state_dict().items():
                        value.copy_(snapshot[key])
            cleanup.callback(restore)
            with torch.no_grad():
                for key, value in self.model.state_dict().items():
                    value.copy_(self.effective(key))
            for injections in self.injections.values():
                for injection in injections:
                    cleanup.callback(injection.eject, self)
                    injection.inject(self)
                    self.is_injected = True
            self.model.current_patcher = self
            yield


class BaseSampler:
    calls = []
    @classmethod
    def INPUT_TYPES(cls): return {"required": {}, "optional": {}}
    def sample(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs["latent_image"], "one run"


class SDAMergeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        name = "_donut_sda_merge_testpkg"
        package = types.ModuleType(name)
        package.__path__ = [str(ROOT)]
        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        modules = {name: package, "comfy": comfy}
        for leaf in ("hooks", "lora", "lora_convert", "patcher_extension", "weight_adapter", "utils"):
            module = types.ModuleType("comfy." + leaf)
            setattr(comfy, leaf, module)
            modules[module.__name__] = module
        class Keyframes:
            def __init__(self): self.keyframes = []
            def add(self, frame): self.keyframes.append(frame)
        class HookGroup:
            def __init__(self): self.hooks = []
            def add(self, hook): self.hooks.append(hook)
            @classmethod
            def combine_all_hooks(cls, groups):
                result = cls()
                result.hooks = [hook for group in groups for hook in group.hooks]
                return result
        comfy.hooks.HookKeyframeGroup = Keyframes
        comfy.hooks.HookKeyframe = lambda **kw: types.SimpleNamespace(**kw)
        comfy.hooks.WeightHook = lambda **kw: types.SimpleNamespace(**kw)
        comfy.hooks.HookGroup = HookGroup
        comfy.patcher_extension.PatcherInjection = Injection
        comfy.patcher_extension.WrappersMP = types.SimpleNamespace(APPLY_MODEL="apply", SAMPLER_SAMPLE="sample")
        comfy.weight_adapter.BypassInjectionManager = Manager
        comfy.weight_adapter.WeightAdapterBase = Adapter
        modules["folder_paths"] = types.ModuleType("folder_paths")
        base = types.ModuleType(name + ".DonutKSamplerCFGLinear")
        base.DonutSampler = BaseSampler
        modules[base.__name__] = base
        safe = types.ModuleType(name + ".DonutSafeApplyLoRAStack")
        safe._partition_bypass_targets = Mock(side_effect=lambda root, keys, components: (components, {}, {}))
        modules[safe.__name__] = safe
        cls.patched_modules = patch.dict(sys.modules, modules)
        cls.patched_modules.start()
        cls.addClassCleanup(cls.patched_modules.stop)
        def load(leaf):
            full = name + "." + leaf
            spec = importlib.util.spec_from_file_location(full, ROOT / (leaf + ".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[full] = module
            cls.addClassCleanup(sys.modules.pop, full, None)
            spec.loader.exec_module(module)
            setattr(package, leaf, module)
            return module
        cls.merge = load("DonutModelMergeKrea2")
        cls.routing = load("donut_sda_merge")
        cls.schedule = load("donut_sda_schedule")
        cls.sda = load("donut_krea2_sda")
        cls.safe = safe

    def setUp(self):
        Manager.roots.clear()
        Manager.fail_on_root = Manager.fail_eject_root = Manager.missing_hook_root = None
        self.safe._partition_bypass_targets.reset_mock()
        BaseSampler.calls.clear()
        self.x = torch.tensor([[.3, -.7, 1.1], [.4, .2, -.1]])

    def merged(self, ratios=None):
        with redirect_stdout(io.StringIO()):
            return self.merge.DonutModelMergeKrea2().merge(
                Patcher(2.), Patcher(5.), "Experimental bypass",
                **(ratios or {"first.": 1., "txtfusion.": 0., "blocks.1.": .35}))[0]

    def prepared(self, model, selected=KEYS, strength=1.):
        patches = {key: Adapter(.2 + i * .1) for i, key in enumerate(selected)}
        return self.schedule.prepare_sda(model, [], [], patches, strength, "Experimental bypass")[0], patches

    def outputs(self, model, sigma=None, fail=False):
        root = model.model
        def run(x, *a, **k):
            values = {key: root.get_submodule(key[:-7])(x) for key in KEYS}
            if fail:
                raise RuntimeError("denoiser interrupted")
            return values
        if sigma is None:
            return run(self.x)
        executor = Mock(side_effect=run)
        executor.class_obj = root
        wrapper = model.wrappers[("apply", self.schedule.SDA_WRAPPER_KEY)]
        return wrapper(executor, self.x, sigma, transformer_options={"sample_sigmas": SIGMAS})

    def test_real_merge_routes_falsey_plan_and_accepts_model(self):
        model = self.merged()
        self.assertFalse(bool(model.injections[INJECTION]))
        self.assertGreater(len(model.injections[INJECTION]), 0)
        self.sda._validate_model(model)
        targets = self.routing.SDAMergeTargets.build(model, dict.fromkeys(KEYS))
        self.assertEqual(set(targets.source), {KEYS[0]})
        self.assertEqual(set(targets.primary), set(KEYS[1:]))

    def test_real_forward_delta_and_exact_cutoff_for_mixed_full_and_body_swaps(self):
        for ratios in ({"first.": 1., "txtfusion.": 0., "blocks.1.": .35},
                       {"first.": 0.}, {"first.": 1., "blocks.": 0.}):
            for strength in (.3, 1., 1.5):
                with self.subTest(ratios=ratios, strength=strength):
                    model, patches = self.prepared(self.merged(ratios), strength=strength)
                    with model.activate():
                        baseline = self.outputs(model)
                        for index, sigma in enumerate(SIGMAS[:-1]):
                            # Repeat an evaluation: NAG/CFG calls must not consume a step.
                            for _ in range(2):
                                values = self.outputs(model, sigma)
                                for key in KEYS:
                                    delta = F.linear(self.x, patches[key].delta()) * strength if index < 2 else 0.
                                    torch.testing.assert_close(values[key], baseline[key] + delta)
                                # SDA is scoped, so a call outside the wrapper is clean.
                                for key, value in self.outputs(model).items():
                                    torch.testing.assert_close(value, baseline[key], rtol=0, atol=0)

    def test_only_non_swapped_sda_targets_do_not_require_source_adapters(self):
        model, _ = self.prepared(self.merged(), selected=KEYS[1:])
        Manager.roots.clear()
        with model.activate():
            Manager.roots.clear()  # exclude the real merge manager
            self.outputs(model, SIGMAS[0])
            self.assertEqual(Manager.roots, [model.model])

    def test_source_only_sda_does_not_wrap_unused_primary_weights(self):
        model, _ = self.prepared(self.merged(), selected=KEYS[:1])
        source = model.additional_models[SOURCE][0]
        with model.activate():
            Manager.roots.clear()
            self.outputs(model, SIGMAS[0])
            self.assertEqual(Manager.roots, [source.model])

    def test_runtime_uses_sampling_clones_retained_source_not_loader_root(self):
        prepared, patches = self.prepared(self.merged())
        old_source = prepared.additional_models[SOURCE][0]
        clone = prepared.clone()
        clone.additional_models[SOURCE] = [Patcher(9.)]
        with clone.activate():
            Manager.roots.clear()
            baseline = self.outputs(clone)
            output = self.outputs(clone, SIGMAS[0])
            torch.testing.assert_close(output[KEYS[0]], baseline[KEYS[0]] + F.linear(self.x, patches[KEYS[0]].delta()))
            self.assertNotIn(old_source.model, Manager.roots)
            self.assertIn(clone.additional_models[SOURCE][0].model, Manager.roots)
        self.assertIs(prepared.additional_models[SOURCE][0], old_source)

    def test_upstream_regular_and_forward_adapters_survive_on_both_models(self):
        merged = self.merged()
        source = merged.additional_models[SOURCE][0]
        # Ordinary patches on the actually executed roots.
        source.add_patches({KEYS[0]: Adapter(.11)}, .7)
        merged.add_patches({KEYS[1]: Adapter(.13)}, .8)
        for owner, key in ((source, KEYS[0]), (merged, KEYS[1])):
            manager = Manager()
            manager.add_adapter(key, Adapter(.15), .6)
            owner.set_injections("ordinary_lora", manager.create_injections(owner.model))
        model, patches = self.prepared(merged)
        source = model.additional_models[SOURCE][0]
        before_injections = (dict(model.injections), dict(source.injections))
        before_attachments = dict(model.attachments)
        with model.activate():
            baseline = self.outputs(model)
            for sigma in (SIGMAS[0], SIGMAS[2]):
                output = self.outputs(model, sigma)
                for key in KEYS:
                    delta = F.linear(self.x, patches[key].delta()) if sigma > SIGMAS[2] else 0.
                    torch.testing.assert_close(output[key], baseline[key] + delta)
        self.assertEqual(before_injections, (model.injections, source.injections))
        self.assertEqual(before_attachments, model.attachments)

    def test_sda_off_and_zero_keep_merged_model_without_loading_adapter(self):
        model = self.merged()
        for enabled, strength in ((False, 1.), (True, 0.)):
            with patch.object(self.sda, "_schedule_module", side_effect=AssertionError):
                args = self.params()
                self.sda.DonutSampler().sample(model, sda_enabled=enabled, sda_strength=strength, **args)
                self.assertIs(BaseSampler.calls[-1]["model"], model)

    def params(self):
        return dict(seed=42, steps=8, cfg_start=1., cfg_halfway=1., cfg_end=1., halfway_step=4,
                    sampler_name="bleh_preset_0", scheduler="beta", positive=[], negative=[],
                    latent_image={"samples": self.x}, denoise=1., turbo_mode=True)

    def test_node_keeps_v4_bleh_beta_and_one_call_with_merge(self):
        model = self.merged({"first.": 1., "txtfusion.": 0.})
        patches = {key: Adapter() for key in KEYS}
        with patch.object(self.sda, "_sda_path", return_value="fixture"), \
             patch.object(self.sda, "_file_identity", return_value=("fixture",)), \
             patch.object(self.sda, "_load_verified_lora", return_value={}), \
             patch.object(self.schedule, "map_sda_weights", return_value=patches):
            _, info = self.sda.DonutSampler().sample(model, sda_enabled=True, **self.params())
        self.assertEqual(len(BaseSampler.calls), 1)
        actual = BaseSampler.calls[0]
        self.assertEqual((actual["sampler_name"], actual["scheduler"], actual["seed"]), ("bleh_preset_0", "beta", 42))
        self.assertEqual(actual["model"].attachments, model.attachments)
        self.assertEqual(actual["model"].model_options["donut_lora_execution_mode"], "Experimental bypass")
        self.assertIn("ON 1-2 / OFF 3-8", info)

    def test_failure_in_source_injection_or_denoiser_restores_both_roots(self):
        for phase in ("injection", "denoiser", "eject"):
            with self.subTest(phase=phase):
                model, _ = self.prepared(self.merged())
                source = model.additional_models[SOURCE][0]
                with model.activate():
                    before = [(layer, layer.forward) for root in (model.model, source.model)
                              for layer in root.modules() if isinstance(layer, torch.nn.Linear)]
                    baseline = self.outputs(model)
                    Manager.fail_on_root = source.model if phase == "injection" else None
                    Manager.fail_eject_root = source.model if phase == "eject" else None
                    try:
                        with self.assertRaises(RuntimeError):
                            self.outputs(model, SIGMAS[0], fail=phase == "denoiser")
                    finally:
                        Manager.fail_on_root = Manager.fail_eject_root = None
                    for layer, forward in before:
                        self.assertEqual(layer.forward, forward)
                    for key, value in self.outputs(model).items():
                        torch.testing.assert_close(value, baseline[key], rtol=0, atol=0)

    def test_missing_source_hook_fails_before_any_sda_injection(self):
        model, _ = self.prepared(self.merged())
        source = model.additional_models[SOURCE][0]
        with model.activate():
            Manager.missing_hook_root = source.model
            try:
                with self.assertRaisesRegex(RuntimeError, "retained model2"):
                    self.outputs(model, SIGMAS[0])
            finally:
                Manager.missing_hook_root = None

    def test_preflight_checks_source_not_unused_primary_target(self):
        model = self.merged()
        source = model.additional_models[SOURCE][0]
        def partition(root, keys, components):
            if root is source.model and KEYS[0] in components:
                return {}, components, {KEYS[0]: ("source pre-quantization scale",)}
            return components, {}, {}
        with patch.object(self.safe, "_partition_bypass_targets", side_effect=partition):
            with self.assertRaisesRegex(ValueError, "retained model2.*No always-on fallback"):
                self.prepared(model)
        self.assertFalse(model.wrappers)

    def test_later_model_and_source_weights_are_byte_unchanged(self):
        model, _ = self.prepared(self.merged())
        snapshots = [{k: v.clone() for k, v in root.state_dict().items()}
                     for root in (model.model, model.additional_models[SOURCE][0].model)]
        with model.activate():
            for sigma in SIGMAS[:-1]: self.outputs(model, sigma)
        for root, expected in zip((model.model, model.additional_models[SOURCE][0].model), snapshots):
            for key, value in root.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)

    def test_single_model_bypass_still_works(self):
        model, patches = self.prepared(Patcher())
        with model.activate():
            baseline = self.outputs(model)
            for key, value in self.outputs(model, SIGMAS[0]).items():
                torch.testing.assert_close(value, baseline[key] + F.linear(self.x, patches[key].delta()))

    def test_metadata_only_or_missing_source_plans_fail(self):
        for corrupt in ("plans", "source", "injection", "empty_injection", "ratio"):
            model = self.merged()
            if corrupt == "plans": model.attachments.clear()
            elif corrupt == "source": model.additional_models.clear()
            elif corrupt == "injection": model.injections.clear()
            elif corrupt == "empty_injection": model.injections[INJECTION] = []
            else:
                key = next(iter(model.attachments))
                path, weight, _ = model.attachments[key][0]
                model.attachments[key] = [(path, weight, .5)]
            with self.subTest(corrupt=corrupt), self.assertRaises(ValueError):
                self.sda._validate_model(model)

    def test_source_compilation_and_wrong_architecture_are_rejected(self):
        for attribute, value in (("_orig_mod", object()), ("txtdim", 99)):
            model = self.merged()
            setattr(model.additional_models[SOURCE][0].model.diffusion_model, attribute, value)
            with self.assertRaises(ValueError): self.sda._validate_model(model)

    def test_active_patcher_and_unchanged_plan_required_at_runtime(self):
        for corrupt in ("no_patcher", "not_injected", "plan"):
            model, _ = self.prepared(self.merged())
            with model.activate():
                if corrupt == "no_patcher": model.model.current_patcher = None
                elif corrupt == "not_injected": model.is_injected = False
                else:
                    key = next(iter(model.attachments))
                    model.attachments[key] = [(KEYS[1][:-7], KEYS[1], 0.)]
                with self.subTest(corrupt=corrupt), self.assertRaises(ValueError):
                    self.outputs(model, SIGMAS[0])

    def test_missing_and_incompatible_live_source_targets_rejected(self):
        for replacement in (torch.nn.Identity(), torch.nn.Linear(4, 2)):
            model = self.merged()
            model.additional_models[SOURCE][0].model.diffusion_model.txtfusion.projector = replacement
            with self.assertRaises(ValueError): self.prepared(model)

    def test_native_single_model_conditioning_hooks_still_preserve_metadata(self):
        model = Patcher()
        tensor = torch.ones(1, 1, 1)
        positive = [[tensor, {"keep": "original"}]]
        weights = {KEYS[1]: Adapter()}
        out, pos, neg = self.schedule.prepare_sda(model, positive, [], weights, .7, "Comfy patches")
        self.assertIs(pos[0][0], tensor)
        self.assertEqual(pos[0][1]["keep"], "original")
        self.assertNotIn("hooks", positive[0][1])
        self.assertIs(pos[0][1]["hooks"].hooks[0].weights, weights)
        self.assertIs(out.model, model.model)
        self.assertFalse(model.wrappers)

    def test_native_primary_only_hook_is_allowed_when_other_layers_are_swapped(self):
        model = self.merged()
        weights = {KEYS[1]: Adapter()}
        out, pos, _ = self.schedule.prepare_sda(
            model, [[self.x, {}]], [], weights, 1., "Comfy patches")
        self.assertIs(pos[0][1]["hooks"].hooks[0].weights, weights)
        self.assertEqual(out.attachments, model.attachments)
        self.assertIn(INJECTION, out.injections)

    def test_aliasing_and_malformed_injection_are_not_silently_accepted(self):
        model = self.merged()
        source = model.additional_models[SOURCE][0]
        source.model.diffusion_model.txtfusion.projector = model.model.diffusion_model.txtfusion.projector
        with self.assertRaisesRegex(ValueError, "aliases"):
            self.prepared(model)
        model = self.merged()
        model.injections[INJECTION] = None
        with self.assertRaisesRegex(ValueError, "runtime merge injection"):
            self.sda._validate_model(model)

    def test_no_silent_native_weight_hook_on_unused_swapped_layer(self):
        model = self.merged()
        with self.assertRaisesRegex(ValueError, "inherits Experimental bypass"):
            self.schedule.prepare_sda(model, [], [], {KEYS[0]: Adapter()}, 1., "Comfy patches")


if __name__ == "__main__":
    unittest.main()

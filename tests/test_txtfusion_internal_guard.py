"""CPU execution tests with small torch components and Comfy interface doubles.

These exercise real tensor math, safetensors file references and wrapper calls.
They are NOT checkpoint/GPU image-quality or full-Comfy lifecycle validation.
"""
import copy
import importlib.util
import logging
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import tempfile

import torch
from torch import nn
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import donut_txtfusion_guard as guard


class Linear(nn.Linear):
    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight.to(x), None if self.bias is None else self.bias.to(x))


class Norm(nn.Module):
    def __init__(self, n, device=None):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(n, device=device))
    def forward(self, x):
        return torch.nn.functional.rms_norm(x.float(), (x.shape[-1],), 1 + self.scale.float(), eps=1e-5).to(x)


class QKNorm(nn.Module):
    def __init__(self, n, device=None):
        super().__init__()
        self.qnorm, self.knorm = Norm(n, device), Norm(n, device)
    def forward(self, q, k):
        return self.qnorm(q), self.knorm(k)


class Attention(nn.Module):
    def __init__(self, dim, heads=1, kvheads=1, bias=False, device=None, operations=None):
        super().__init__()
        self.heads, self.kvheads = heads, kvheads
        linear = operations.Linear if operations else Linear
        for name in ('wq', 'wk', 'wv', 'gate', 'wo'):
            setattr(self, name, linear(dim, dim, bias=bias, device=device))
        self.qknorm = QKNorm(dim, device)
    def forward(self, x, mask=None, transformer_options=None):
        q, k = self.qknorm(self.wq(x), self.wk(x))
        a = torch.softmax(q @ k.transpose(-1, -2) / x.shape[-1] ** .5, dim=-1)
        return self.wo((a @ self.wv(x)) * torch.sigmoid(self.gate(x)))


class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate, self.up, self.down = Linear(dim, dim * 2, bias=False), Linear(dim, dim * 2, bias=False), Linear(dim * 2, dim, bias=False)
    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


class TextFusionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.prenorm, self.postnorm = Norm(dim), Norm(dim)
        self.attn, self.mlp = Attention(dim), SwiGLU(dim)
    def forward(self, x, mask=None, transformer_options=None):
        x = x + self.attn(self.prenorm(x), mask=mask, transformer_options=transformer_options)
        return x + self.mlp(self.postnorm(x))


class TextFusionTransformer(nn.Module):
    def __init__(self, width=4, taps=3):
        super().__init__()
        self.layerwise_blocks = nn.ModuleList([TextFusionBlock(width), TextFusionBlock(width)])
        self.projector = Linear(taps, 1, bias=False)
        self.refiner_blocks = nn.ModuleList([TextFusionBlock(width), TextFusionBlock(width)])
    def forward(self, x, mask=None, transformer_options=None):
        b, l, n, d = x.shape
        x = x.reshape(b*l, n, d)
        for block in self.layerwise_blocks:
            x = block(x.contiguous(), transformer_options=transformer_options)
        x = self.projector(x.reshape(b, l, n, d).permute(0, 1, 3, 2)).squeeze(-1)
        for block in self.refiner_blocks:
            x = block(x, mask=mask, transformer_options=transformer_options)
        return x


def comfy_modules():
    names = ['comfy', 'comfy.ops', 'comfy.patcher_extension', 'comfy.ldm', 'comfy.ldm.krea2', 'comfy.ldm.krea2.model']
    modules = {n: types.ModuleType(n) for n in names}
    for name, module in modules.items():
        module.__path__ = []
        if '.' in name:
            parent, child = name.rsplit('.', 1)
            setattr(modules[parent], child, module)
    modules['comfy.ops'].manual_cast = types.SimpleNamespace(Linear=Linear)
    modules['comfy.patcher_extension'].WrappersMP = types.SimpleNamespace(DIFFUSION_MODEL='diffusion_model')
    core = modules['comfy.ldm.krea2.model']
    for cls in [Attention, SwiGLU, TextFusionBlock, TextFusionTransformer]:
        setattr(core, cls.__name__, cls)
    return modules


class Patcher:
    def __init__(self, fusion):
        self.model = types.SimpleNamespace(txtfusion=fusion)
        self.patches = {}
        self.injections = {}
        self.attachments = {}
        self.model_options = {'transformer_options': {}}
        self.wrappers = {'diffusion_model': {guard.NAG_KEY: [self.nag]}}
    @staticmethod
    def nag(executor, x, transformer_options=None):
        if transformer_options and transformer_options.get('inactive'):
            return executor(x)
        return executor.class_obj.txtfusion(x, transformer_options=transformer_options)
    def clone(self):
        result = copy.copy(self)
        result.wrappers = {k: {n: list(v) for n, v in g.items()} for k, g in self.wrappers.items()}
        result.model_options = copy.deepcopy(self.model_options)
        return result
    def get_model_object(self, name):
        assert name == 'diffusion_model'
        return self.model
    def get_attachment(self, key):
        return self.attachments.get(key)


class Executor:
    def __init__(self, model):
        self.class_obj = model
    def __call__(self, value):
        return value


class InternalGuardTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.file = Path(self.tmp.name) / 'base.safetensors'
        self.fusion = TextFusionTransformer()
        self.state = {f'txtfusion.{k}': v.detach().clone() for k, v in self.fusion.state_dict().items()}
        save_file(self.state, str(self.file))
        self.model = Patcher(self.fusion)
        self.modules = patch.dict(sys.modules, comfy_modules())
        self.modules.start(); self.addCleanup(self.modules.stop)
        self.x = torch.randn(2, 5, 3, 4)
    def adapt(self):
        with torch.no_grad():
            self.fusion.layerwise_blocks[0].attn.wo.weight.mul_(2)
            self.fusion.refiner_blocks[1].mlp.down.weight.mul_(3)
        for name in ['layerwise_blocks.0.attn.wo.weight', 'refiner_blocks.1.mlp.down.weight']:
            self.model.patches[guard.PREFIX + name] = [(1., object(), 1., None, None)]
    def test_reference_is_file_derived_even_after_live_adapters_materialize(self):
        self.adapt()
        installed, run = guard.install_guard(self.model, self.file)
        ref = run.references['layerwise_blocks.0.attn'].wo.weight
        expected = self.state['txtfusion.layerwise_blocks.0.attn.wo.weight']
        torch.testing.assert_close(ref, expected)
        self.assertFalse(torch.equal(ref, self.fusion.layerwise_blocks[0].attn.wo.weight))
        with torch.no_grad(): self.fusion.layerwise_blocks[0].attn.wo.weight.zero_()
        torch.testing.assert_close(ref, expected)
        self.assertNotEqual(ref.data_ptr(), expected.data_ptr())
    def test_guard_changes_internal_residual_not_final_scalar(self):
        self.adapt()
        baseline = self.fusion(self.x)
        installed, run = guard.install_guard(self.model, self.file)
        wrapped = installed.wrappers['diffusion_model'][guard.NAG_KEY][0]
        out = wrapped(Executor(installed.model), self.x)
        self.assertGreater(run.contribution_calls, 0)
        self.assertFalse(torch.allclose(out, baseline))
        # Effect survives a final RMSNorm: not just overall output scaling.
        rms = lambda x: torch.nn.functional.rms_norm(x, (4,), eps=1e-5)
        self.assertFalse(torch.allclose(rms(out), rms(baseline), atol=1e-5))
        self.assertEqual(set(run.reports), {'layerwise_blocks.0.attn', 'refiner_blocks.1.mlp'})
    def test_no_adapters_returns_identical_patcher_without_reference_io(self):
        with patch.object(guard, 'load_reference_states', side_effect=AssertionError('IO')):
            installed, run = guard.install_guard(self.model, None)
        self.assertIs(installed, self.model); self.assertIsNone(run)
    def test_zero_strength_returns_original_path(self):
        self.model.patches[guard.PREFIX + 'layerwise_blocks.0.attn.wo.weight'] = [(0., object(), 1., None, None)]
        self.assertIs(guard.install_guard(self.model, None)[0], self.model)
    def test_native_order_without_references_matches_exactly(self):
        run = guard.GuardRun({}, 'test')
        actual = guard.GuardedFusion(self.fusion, run)(self.x)
        self.assertTrue(torch.equal(actual, self.fusion(self.x)))
    def test_mask_and_options_forwarding_keeps_unguarded_behavior(self):
        options = {'sigmas': torch.tensor([.3])}
        run = guard.GuardRun({}, 'test')
        torch.testing.assert_close(guard.GuardedFusion(self.fusion, run)(self.x, transformer_options=options),
                                   self.fusion(self.x, transformer_options=options))
    def test_batch_independence_including_flattened_layerwise_tokens(self):
        self.adapt()
        _, run = guard.install_guard(self.model, self.file)
        fn = guard.GuardedFusion(self.fusion, run)
        first = fn(self.x[:1])
        combined = fn(torch.cat([self.x[:1], self.x[1:] * 100]))
        torch.testing.assert_close(first, combined[:1])
    def test_original_wrappers_weights_and_forward_are_not_mutated(self):
        self.adapt()
        self.model.wrappers['diffusion_model'] = {'before':[object()], guard.NAG_KEY:[Patcher.nag], 'after':[object()]}
        before = {k:v.clone() for k,v in self.fusion.state_dict().items()}
        installed, run = guard.install_guard(self.model, self.file)
        self.assertEqual(list(installed.wrappers['diffusion_model']), ['before', guard.NAG_KEY, 'after'])
        self.assertIs(self.model.wrappers['diffusion_model'][guard.NAG_KEY][0], Patcher.nag)
        installed.wrappers['diffusion_model'][guard.NAG_KEY][0](Executor(installed.model), self.x)
        self.assertNotIn('forward', self.fusion.__dict__)
        for k,v in self.fusion.state_dict().items(): self.assertTrue(torch.equal(v, before[k]))
        run.close(); self.assertEqual(run.references, {})
    def test_inactive_nag_delegates_original_executor(self):
        self.adapt()
        installed, run = guard.install_guard(self.model, self.file)
        out = installed.wrappers['diffusion_model'][guard.NAG_KEY][0](Executor(installed.model), self.x, {'inactive':True})
        self.assertIs(out, self.x); self.assertEqual(run.forward_calls, 0)
    def test_does_not_stack_with_previous_midpoint_experiment(self):
        self.model.wrappers['diffusion_model'][guard.EXPERIMENT_KEY] = [object()]
        with self.assertRaisesRegex(ValueError, 'standard non-edit'): guard.install_guard(self.model, self.file)
    def test_no_reference_is_an_error_not_late_snapshot(self):
        self.adapt()
        with self.assertRaisesRegex(ValueError, 'Select the same checkpoint'): guard.install_guard(self.model, None)
    def test_missing_or_wrong_checkpoint_tensor_fails(self):
        self.adapt()
        del self.state['txtfusion.layerwise_blocks.0.attn.wo.weight']
        save_file(self.state, str(self.file))
        with self.assertRaisesRegex(ValueError, 'does not match'): guard.install_guard(self.model, self.file)
    def test_quantized_reference_is_explicitly_unsupported(self):
        self.adapt()
        self.state['txtfusion.layerwise_blocks.0.attn.wo.weight'] = self.state['txtfusion.layerwise_blocks.0.attn.wo.weight'].to(torch.float8_e4m3fn)
        save_file(self.state, str(self.file))
        with self.assertRaisesRegex(ValueError, 'Unsupported'): guard.install_guard(self.model, self.file)
    def test_unsupported_merge_injection_fails(self):
        self.model.injections['model2_swap'] = []
        with self.assertRaisesRegex(ValueError, 'merges'): guard.install_guard(self.model, self.file)
    def test_reference_component_finite_and_cpu(self):
        self.adapt()
        _, run = guard.install_guard(self.model, self.file)
        for component in run.references.values():
            for parameter in component.parameters():
                self.assertEqual(parameter.device.type, 'cpu')
                self.assertFalse(parameter.requires_grad)
        self.assertEqual(len(run.digest), 64)
    def test_recorded_bypass_adapters_are_detected_without_conversion(self):
        key = guard.PREFIX + 'refiner_blocks.0.mlp.down.weight'
        self.model.injections['donut_bypass_lora'] = [object()]
        self.model.attachments[guard.BYPASS_KEY] = {key:[(object(), .7)]}
        self.assertEqual(guard.affected_components(self.model), {'refiner_blocks.0.mlp'})
        installed, _ = guard.install_guard(self.model, self.file)
        self.assertIs(installed.injections, self.model.injections)
        self.assertEqual(self.model.patches, {})
    def test_unrecorded_bypass_rejected(self):
        self.model.injections['donut_bypass_lora'] = [object()]
        with self.assertRaisesRegex(RuntimeError, 'recorded'): guard.affected_components(self.model)
    def test_custom_block_forward_rejected(self):
        self.adapt()
        self.fusion.layerwise_blocks[0].forward = lambda x: x
        with self.assertRaisesRegex(ValueError, 'Custom'): guard.install_guard(self.model, self.file)
    def test_gain_cap_and_zero_degenerate_inputs(self):
        reference, patched = torch.ones(2,3,4), torch.ones(2,3,4) * 100
        adjusted, _, _, gains, clipped = guard.match_contribution(reference, patched, batch_size=2)
        self.assertTrue(torch.equal(gains, torch.full_like(gains, .25)))
        self.assertEqual(clipped, 2)
        zero = torch.zeros_like(reference)
        self.assertTrue(torch.equal(guard.match_contribution(zero, patched, batch_size=2)[0], patched))
    def test_nonfinite_contribution_is_not_hidden(self):
        with self.assertRaisesRegex(RuntimeError, 'Nonfinite'):
            guard.match_contribution(torch.ones(1,3), torch.full((1,3), float('nan')), batch_size=1)
    def test_input_and_projector_are_unchanged(self):
        self.adapt()
        source = self.x.clone(); projector = self.fusion.projector.weight.detach().clone()
        installed, run = guard.install_guard(self.model, self.file)
        guard.GuardedFusion(self.fusion, run)(self.x)
        self.assertTrue(torch.equal(self.x, source))
        self.assertTrue(torch.equal(self.fusion.projector.weight, projector))
    def test_baseline_file_prefixes_are_unambiguous(self):
        self.adapt()
        prefixed = {'diffusion_model.' + k:v for k,v in self.state.items()}
        save_file(prefixed, str(self.file))
        guard.install_guard(self.model, self.file)
        prefixed.update({k:v.clone() for k,v in self.state.items()}); save_file(prefixed, str(self.file))
        with self.assertRaisesRegex(ValueError, 'ambiguous'): guard.install_guard(self.model, self.file)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()

"""Model-guard CPU contracts; real tensor/FP8/INT8 math, Comfy API doubles.

Not a substitute for GPU kernels or a complete Comfy/V5 Run-button execution.
"""
import copy
import importlib.util
import logging
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import weakref
import gc

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import donut_txtfusion_model_guard as guard


def identity(x, **kwargs): return x


class Linear(nn.Linear):
    def forward(self, x):
        return nn.functional.linear(x, self.weight.to(x), None if self.bias is None else self.bias.to(x))


class FP8Linear(Linear):
    def __init__(self, a, b, bias=False):
        super().__init__(a, b, bias=bias)
        self.register_buffer('weight_scale', torch.tensor(.01))
        self.weight = nn.Parameter((self.weight.detach() / self.weight_scale).to(torch.float8_e4m3fn), requires_grad=False)
    def forward(self, x):
        return nn.functional.linear(x, self.weight.float().to(x) * self.weight_scale.to(x), None)


class INT8Linear(Linear):
    def __init__(self, a, b, bias=False):
        super().__init__(a, b, bias=bias)
        self.weight = nn.Parameter(torch.quantize_per_tensor(self.weight.detach(), .01, 3, torch.qint8), requires_grad=False)
    def forward(self, x):
        return nn.functional.linear(x, self.weight.dequantize().to(x), None)


class Norm(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.scale = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        return nn.functional.rms_norm(x.float(), (x.shape[-1],), 1 + self.scale.float(), 1e-5).to(x)


class Attention(nn.Module):
    def __init__(self, width, linear=Linear):
        super().__init__()
        for name in ('wq','wk','wv','gate','wo'): setattr(self, name, linear(width, width, bias=False))
        self.qnorm, self.knorm = Norm(width), Norm(width)
    def forward(self, x, mask=None, transformer_options=None):
        q, k = self.qnorm(self.wq(x)), self.knorm(self.wk(x))
        a = q @ k.transpose(-1,-2) / x.shape[-1]**.5
        if mask is not None: a = a + mask
        return self.wo((a.softmax(-1) @ self.wv(x)) * torch.sigmoid(self.gate(x)))


class MLP(nn.Module):
    def __init__(self, width, linear=Linear):
        super().__init__(); self.gate = linear(width,width*2,bias=False)
        self.up = linear(width,width*2,bias=False); self.down = linear(width*2,width,bias=False)
    def forward(self, x): return self.down(nn.functional.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, width, linear=Linear):
        super().__init__(); self.prenorm, self.postnorm = Norm(width), Norm(width)
        self.attn, self.mlp = Attention(width,linear), MLP(width,linear)
    def forward(self, x, mask=None, transformer_options=None):
        x = x + self.attn(self.prenorm(x), mask=mask, transformer_options=transformer_options)
        return x + self.mlp(self.postnorm(x))


class Fusion(nn.Module):
    def __init__(self, linear=Linear):
        super().__init__()
        self.layerwise_blocks = nn.ModuleList([Block(4,linear),Block(4,linear)])
        self.projector = linear(3,1,bias=False)
        self.refiner_blocks = nn.ModuleList([Block(4,linear),Block(4,linear)])
    def forward(self, x, mask=None, transformer_options=None):
        b,l,n,d = x.shape
        x = x.reshape(b*l,n,d)
        for block in self.layerwise_blocks: x = block(x.contiguous(), transformer_options=transformer_options)
        x = self.projector(x.reshape(b,l,n,d).permute(0,1,3,2)).squeeze(-1)
        for block in self.refiner_blocks: x = block(x, mask=mask, transformer_options=transformer_options)
        return x


class Root(nn.Module):
    def __init__(self, fusion):
        super().__init__(); self.diffusion_model = nn.Module(); self.diffusion_model.txtfusion = fusion
        self.current_patcher = None
    def get_dtype(self): return torch.float32


def nested_copy(d):
    if isinstance(d, dict): return {k:nested_copy(v) for k,v in d.items()}
    if isinstance(d, list): return list(d)
    return d


class Patcher:
    def __init__(self, model, load_device=torch.device('cpu'), offload_device=torch.device('cpu')):
        self.model, self.load_device, self.offload_device = model, load_device, offload_device
        self.callbacks = {}; self.wrappers = {}; self.attachments = {}; self.additional_models = {}
        self.model_options = {'transformer_options':{}}; self.patches = {}; self.backup = {}
        self.injections = {}; self.merge_info = None
    def clone(self):
        out = copy.copy(self)
        for field in ('callbacks','wrappers','attachments','additional_models','model_options','patches','injections'):
            setattr(out, field, nested_copy(getattr(self, field)))
        return out
    def get_model_object(self, path): return self.model.get_submodule(path)
    def model_dtype(self): return self.model.get_dtype()
    def get_attachment(self, key): return self.attachments.get(key)
    def set_attachments(self, key, value): self.attachments[key] = value
    def remove_attachments(self, key): self.attachments.pop(key, None)
    def set_additional_models(self, key, value): self.additional_models[key] = value
    def get_additional_models_with_key(self, key): return self.additional_models.get(key, [])
    def remove_additional_models(self, key): self.additional_models.pop(key, None)
    def add_callback_with_key(self, kind, key, callback): self.callbacks.setdefault(kind, {}).setdefault(key, []).append(callback)
    def remove_callbacks_with_key(self, kind, key): self.callbacks.get(kind, {}).pop(key, None)
    def add_wrapper_with_key(self, kind, key, value): self.wrappers.setdefault(kind, {}).setdefault(key, []).append(value)
    def remove_wrappers_with_key(self, kind, key): self.wrappers.get(kind, {}).pop(key, None)
    def pre_run(self):
        self.model.current_patcher = self
        for values in list(self.callbacks.get('pre',{}).values()):
            for callback in list(values): callback(self)
    def cleanup(self):
        self.model.current_patcher = None
        for values in list(self.callbacks.get('cleanup',{}).values()):
            for callback in list(values): callback(self)


def recipes(patcher, prefix=None):
    result = {}
    for key, value in list(patcher.model.named_parameters()) + list(patcher.model.named_buffers()):
        if key.startswith(guard.PREFIX + '.'):
            result[key] = [(patcher.backup.get(key, value), identity), *patcher.patches.get(key, ())]
    return result


def merge_info(patcher): return patcher.merge_info


def calculate(patches, weight, key, **kwargs):
    # Minimal core merge recipe evaluator, also used to independently compute
    # expected baselines. LoRA payloads are not accepted in a reference recipe.
    for strength, payload, model_scale, offset, function in patches:
        target = weight if offset is None else weight.narrow(*offset)
        target.mul_(model_scale)
        if not isinstance(payload,list): raise AssertionError('adapter survived reference filtering')
        source = calculate(payload[1:], payload[0][1](payload[0][0].float().clone()), key)
        target.add_(source * strength if function is None else function(source * strength))
    return weight


def modules():
    names = ['comfy','comfy.model_patcher','comfy.patcher_extension','comfy.lora','comfy.float','comfy.utils',
             'DonutModelMergeKrea2','donut_krea2_merge_serialization']
    result = {name:types.ModuleType(name) for name in names}
    for name, module in result.items():
        if '.' in name:
            parent, child = name.rsplit('.',1); setattr(result[parent], child, module)
    result['comfy.model_patcher'].ModelPatcher = Patcher
    result['comfy.patcher_extension'].CallbacksMP = types.SimpleNamespace(ON_PRE_RUN='pre',ON_CLEANUP='cleanup',ON_DETACH='detach')
    result['comfy.patcher_extension'].WrappersMP = types.SimpleNamespace(OUTER_SAMPLE='outer')
    result['comfy.lora'].calculate_weight = calculate
    result['comfy.float'].stochastic_rounding = lambda x, dtype, **kwargs: x.to(dtype)
    result['comfy.utils'].string_to_seed = lambda key: 42
    result['DonutModelMergeKrea2']._get_merge_key_patches = recipes
    result['donut_krea2_merge_serialization'].get_krea2_merge_bypass_info = merge_info
    return result


class Cancelled(BaseException): pass


class ModelGuardTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(14); torch.set_num_threads(1)
        ctx = patch.dict(sys.modules, modules()); ctx.start(); self.addCleanup(ctx.stop)
        self.fusion = Fusion(); self.model = Patcher(Root(self.fusion)); self.x = torch.randn(2,5,3,4)*3
    def install(self, model=None): return guard.attach_model_guard(model or self.model)
    def evaluate(self, model, x=None, *, fn=None):
        def execute():
            model.pre_run()
            try:
                return (fn or (lambda: model.model.diffusion_model.txtfusion(self.x if x is None else x,
                      transformer_options=model.model_options['transformer_options'])))()
            finally: model.cleanup()
        outer = model.wrappers.get('outer',{}).get(guard.KEY)
        return outer[0](execute) if outer else execute()
    def adapt(self, model, name='layerwise_blocks.0.attn.wo.weight', magnitude=2.0):
        key = guard.PREFIX + '.' + name; path, attr = key.rsplit('.',1)
        m = model.get_model_object(path); value = getattr(m,attr)
        model.backup[key] = value.detach().clone()
        model.patches[key] = [(1.0,types.SimpleNamespace(weights=('delta',)),1.0,None,None)]
        with torch.no_grad(): value.mul_(magnitude)
    def assert_no_hooks(self, fusion=None):
        for module in (fusion or self.fusion).modules():
            self.assertFalse(module._forward_hooks); self.assertFalse(module._forward_pre_hooks)
    def test_no_adapters_on_is_exact_output_parity(self):
        expected = self.fusion(self.x)
        guarded = self.install()
        self.assertTrue(torch.equal(self.evaluate(guarded),expected)); self.assert_no_hooks()
    def test_native_materialized_lora_uses_backup_not_live_snapshot(self):
        original = self.fusion.layerwise_blocks[0].attn.wo.weight.detach().clone()
        self.adapt(self.model)
        guarded = self.install()
        reference = guarded.get_additional_models_with_key(guard.KEY)[0].model.txtfusion
        self.assertTrue(torch.equal(reference.layerwise_blocks[0].attn.wo.weight,original))
        self.assertNotEqual(reference.layerwise_blocks[0].attn.wo.weight.data_ptr(),original.data_ptr())
        before = self.fusion(self.x); output = self.evaluate(guarded)
        self.assertFalse(torch.allclose(output,before))
    def test_reference_remains_pristine_after_later_native_patch(self):
        guarded = self.install()
        reference = guarded.get_additional_models_with_key(guard.KEY)[0].model.txtfusion
        weights = reference.layerwise_blocks[0].attn.wo.weight.detach().clone()
        self.adapt(guarded)
        self.assertTrue(torch.equal(reference.layerwise_blocks[0].attn.wo.weight,weights))
        self.assertFalse(torch.allclose(self.evaluate(guarded),self.fusion(self.x)))
    def test_guard_effect_survives_final_rmsnorm(self):
        guarded = self.install(); self.adapt(guarded)
        off = self.fusion(self.x); on = self.evaluate(guarded)
        norm = lambda x: nn.functional.rms_norm(x,(4,),eps=1e-5)
        self.assertFalse(torch.allclose(norm(off),norm(on),atol=1e-6))
    def test_nag_alpha_zero_and_no_nag_have_real_contribution_calls(self):
        guarded = self.install(); self.adapt(guarded)
        def upstream_wrapper(alpha):
            # This is the previously broken alpha=0 -> plain executor route.
            plain = lambda: self.fusion(self.x, transformer_options=guarded.model_options['transformer_options'])
            return plain() if alpha == 0 else plain() + plain()
        for alpha in (0.0,.45):
            self.evaluate(guarded,fn=lambda:upstream_wrapper(alpha))
            report = guarded.get_attachment(guard.KEY).last_report
            self.assertEqual(report['calls'],1 if alpha == 0 else 2)
            self.assertEqual(report['contributions'],9 if alpha == 0 else 18)
    def test_positive_negative_unequal_lengths_are_independent(self):
        guarded = self.install(); self.adapt(guarded)
        short = self.x[:,:2]
        def both():
            return self.fusion(self.x), self.fusion(short)
        pair = self.evaluate(guarded,fn=both)
        torch.testing.assert_close(pair[0],self.evaluate(guarded,self.x))
        torch.testing.assert_close(pair[1],self.evaluate(guarded,short))
    def test_rebalance_input_is_unchanged(self):
        guarded = self.install(); self.adapt(guarded)
        saved = self.x.clone(); self.evaluate(guarded)
        self.assertTrue(torch.equal(self.x,saved))
    def test_projector_only_lora_is_guarded_too(self):
        guarded = self.install(); self.adapt(guarded,'projector.weight')
        self.evaluate(guarded)
        report = guarded.get_attachment(guard.KEY).last_report['stats']
        self.assertIn('1:projector',report)
        self.assertAlmostEqual(report['1:projector']['gain'][0],.5,places=6)
    def test_multiple_component_adapters_are_not_only_uncensorfix_targets(self):
        guarded = self.install()
        self.adapt(guarded,'refiner_blocks.0.mlp.down.weight',3)
        self.adapt(guarded,'layerwise_blocks.1.attn.wo.weight',2)
        self.evaluate(guarded)
        report=guarded.get_attachment(guard.KEY).last_report['stats']
        self.assertAlmostEqual(report['1:refiner_blocks.0.mlp']['gain'][0],1/3,places=5)
        self.assertAlmostEqual(report['1:layerwise_blocks.1.attn']['gain'][0],.5,places=5)
    def test_directional_adapter_effect_is_not_removed(self):
        guarded = self.install()
        before = self.fusion(self.x)
        with torch.no_grad():
            w=self.fusion.refiner_blocks[1].mlp.down.weight
            w[0].add_(.8)
        result=self.evaluate(guarded)
        self.assertFalse(torch.allclose(before,result))
    def test_batch_items_do_not_share_statistics(self):
        guarded=self.install();self.adapt(guarded)
        alone=self.evaluate(guarded,self.x[:1])
        mixed=self.evaluate(guarded,torch.cat([self.x[:1],self.x[1:]*100]))
        torch.testing.assert_close(alone,mixed[:1],rtol=1e-5,atol=1e-5)
    def test_bypass_adapter_hook_does_not_contaminate_reference(self):
        leaf=self.fusion.refiner_blocks[1].mlp.down
        original=leaf.forward; leaf.forward=lambda x:original(x)*3
        self.model.injections['donut_bypass_lora']=[object()]
        guarded=self.install()
        reference=guarded.get_additional_models_with_key(guard.KEY)[0].model.txtfusion
        self.assertNotIn('forward',reference.refiner_blocks[1].mlp.down.__dict__)
        self.evaluate(guarded)
        self.assertAlmostEqual(guarded.get_attachment(guard.KEY).last_report['stats']['1:refiner_blocks.1.mlp']['gain'][0],1/3,places=5)
    def test_fp8_checkpoint_preserves_scaled_weight_format(self):
        self.fusion=Fusion(FP8Linear); self.model=Patcher(Root(self.fusion))
        guarded=self.install(); reference=guarded.get_additional_models_with_key(guard.KEY)[0].model.txtfusion
        self.assertEqual(reference.layerwise_blocks[0].attn.wq.weight.dtype,torch.float8_e4m3fn)
        self.assertTrue(torch.equal(reference.layerwise_blocks[0].attn.wq.weight_scale,self.fusion.layerwise_blocks[0].attn.wq.weight_scale))
        leaf=self.fusion.refiner_blocks[1].mlp.down;orig=leaf.forward;leaf.forward=lambda x:orig(x)*2
        self.evaluate(guarded)
        self.assertAlmostEqual(guarded.get_attachment(guard.KEY).last_report['stats']['1:refiner_blocks.1.mlp']['gain'][0],.5,places=5)
    def test_real_cpu_int8_checkpoint_keeps_quantization_metadata(self):
        self.fusion=Fusion(INT8Linear); self.model=Patcher(Root(self.fusion))
        guarded=self.install(); reference=guarded.get_additional_models_with_key(guard.KEY)[0].model.txtfusion
        q=reference.layerwise_blocks[0].attn.wq.weight
        self.assertEqual(q.dtype,torch.qint8);self.assertEqual(q.q_zero_point(),3);self.assertAlmostEqual(q.q_scale(),.01)
        self.assertTrue(torch.equal(self.evaluate(guarded),self.fusion(self.x)))
    def test_runtime_vbar_and_adapter_caches_not_copied(self):
        leaf=self.fusion.layerwise_blocks[0].attn.wq
        leaf._v=object();leaf._v_signature=object();leaf._pin_state={'weights':'live'}
        leaf.weight_function=[lambda x:x*100];leaf.weight_lowvram_function=object()
        ref=guard.capture_reference(self.model)
        copied=ref.layerwise_blocks[0].attn.wq
        self.assertFalse(hasattr(copied,'_v'));self.assertFalse(hasattr(copied,'_pin_state'))
        self.assertEqual(copied.weight_function,[]);self.assertFalse(hasattr(copied,'weight_lowvram_function'))
    def test_source_per_linear_merge_in_same_component_is_supported(self):
        source=Patcher(Root(copy.deepcopy(self.fusion)))
        leaf='layerwise_blocks.0.attn.wq';path=guard.PREFIX+'.'+leaf
        with torch.no_grad():source.get_model_object(path).weight.mul_(3)
        self.model.injections[guard._MERGE_KEY]=[object()]
        self.model.merge_info=(source,((path,path+'.weight',0.),),())
        reference=guard.capture_reference(self.model)
        self.assertTrue(torch.equal(reference.get_submodule(leaf).weight,source.get_model_object(path).weight))
        self.assertTrue(torch.equal(reference.layerwise_blocks[0].attn.wo.weight,self.fusion.layerwise_blocks[0].attn.wo.weight))
    def test_native_partial_merge_preserved_but_source_lora_removed(self):
        key=guard.PREFIX+'.layerwise_blocks.0.attn.wo.weight'
        base=self.fusion.layerwise_blocks[0].attn.wo.weight.detach().clone()
        other=base*4
        source_recipe=[(other,identity),(1.,types.SimpleNamespace(weights=()),1.,None,None)]
        self.model.patches[key]=[(.25,source_recipe,.75,None,None),(1.,types.SimpleNamespace(weights=()),1.,None,None)]
        reference=guard.capture_reference(self.model)
        torch.testing.assert_close(reference.layerwise_blocks[0].attn.wo.weight,base*.75+other*.25)
    def test_v5_model2_quantized_txtfusion_plus_bypass_runs(self):
        source=Patcher(Root(Fusion(FP8Linear)))
        plans=[]
        for name, module in source.model.diffusion_model.txtfusion.named_modules():
            if isinstance(module,Linear):
                path=guard.PREFIX+'.'+name;plans.append((path,path+'.weight',0.))
                target=self.model.get_model_object(path)
                target.forward=lambda x,source_module=module:source_module(x)
        self.model.injections[guard._MERGE_KEY]=[object()]
        self.model.merge_info=(source,tuple(plans),())
        guarded=self.install()
        leaf=source.model.diffusion_model.txtfusion.layerwise_blocks[0].attn.wo
        orig=leaf.forward;leaf.forward=lambda x:orig(x)*2
        result=self.evaluate(guarded)
        self.assertTrue(torch.isfinite(result).all())
        self.assertAlmostEqual(guarded.get_attachment(guard.KEY).last_report['stats']['1:layerwise_blocks.0.attn']['gain'][0],.5,places=5)
    def test_reference_does_not_alias_either_model_after_merge(self):
        source=Patcher(Root(copy.deepcopy(self.fusion)))
        path=guard.PREFIX+'.projector';self.model.merge_info=(source,((path,path+'.weight',0.),),())
        self.model.injections[guard._MERGE_KEY]=[object()]
        reference=guard.capture_reference(self.model)
        before=reference.projector.weight.detach().clone()
        with torch.no_grad():source.model.diffusion_model.txtfusion.projector.weight.zero_()
        self.assertTrue(torch.equal(reference.projector.weight,before))
    def test_later_sda_like_dynamic_adapters_are_guarded_each_call(self):
        guarded=self.install();leaf=self.fusion.refiner_blocks[1].mlp.down;orig=leaf.forward
        for strength in (1.5,3.):
            leaf.forward=lambda x,s=strength:orig(x)*s
            self.evaluate(guarded)
            gain=guarded.get_attachment(guard.KEY).last_report['stats']['1:refiner_blocks.1.mlp']['gain'][0]
            self.assertAlmostEqual(gain,1/strength,places=5)
    def test_same_model_guard_propagates_to_upscale_and_detailer_clones(self):
        guarded=self.install();self.adapt(guarded)
        for length in (5,11,3):
            clone=guarded.clone();self.evaluate(clone,torch.randn(1,length,3,4))
            self.assertEqual(clone.get_attachment(guard.KEY).last_report['contributions'],9)
            self.assert_no_hooks()
    def test_cancellation_and_exception_remove_all_hooks(self):
        guarded=self.install();self.adapt(guarded)
        for error in (Cancelled(),RuntimeError('model fail')):
            def fail():
                self.fusion(self.x)
                raise error
            with self.assertRaises(type(error)):self.evaluate(guarded,fn=fail)
            self.assert_no_hooks();self.assertIsNone(guard._CURRENT.get())
    def test_failure_inside_txtfusion_unwinds_batch_context(self):
        guarded=self.install();leaf=self.fusion.projector;orig=leaf.forward
        def fail(x):raise Cancelled()
        leaf.forward=fail
        with self.assertRaises(Cancelled):self.evaluate(guarded)
        self.assert_no_hooks();self.assertIsNone(guard._CURRENT.get())
        leaf.forward=orig;self.evaluate(guarded);self.assert_no_hooks()
    def test_cleanup_callback_error_still_has_outer_finally(self):
        guarded=self.install()
        def fail(p):raise RuntimeError('third-party cleanup')
        guarded.callbacks['cleanup']={'bad':[fail],**guarded.callbacks['cleanup']}
        with self.assertRaisesRegex(RuntimeError,'third-party'):self.evaluate(guarded)
        self.assert_no_hooks();self.assertIsNone(guard._CURRENT.get())
    def test_logger_failure_does_not_leak_hooks(self):
        guarded=self.install()
        with patch.object(guard.LOGGER,'info',side_effect=RuntimeError('logger')):
            with self.assertRaisesRegex(RuntimeError,'logger'):self.evaluate(guarded)
        self.assert_no_hooks()
    def test_repeated_enable_is_not_double_install(self):
        guarded=self.install();self.assertIs(guard.attach_model_guard(guarded),guarded)
        self.evaluate(guarded)
        self.assertEqual(guarded.get_attachment(guard.KEY).last_report['contributions'],9)
    def test_disable_preserves_other_callbacks_and_wrappers(self):
        sentinel=lambda *a:None
        self.model.callbacks['pre']={'other':[sentinel]};self.model.wrappers['outer']={'other':[sentinel]}
        guarded=self.install();off=guard.remove_model_guard(guarded)
        self.assertIs(off.callbacks['pre']['other'][0],sentinel)
        self.assertIs(off.wrappers['outer']['other'][0],sentinel)
        self.assertNotIn(guard.KEY,off.additional_models)
        self.assertNotIn(guard.KEY,off.model_options['transformer_options'])
        self.assertIsNotNone(guarded.get_attachment(guard.KEY))
    def test_off_does_not_clone_unrelated_model_or_do_io(self):
        with patch.object(guard,'capture_reference',side_effect=AssertionError('capture')):
            self.assertIs(guard.remove_model_guard(self.model),self.model)
    def test_guard_does_not_register_injection_or_change_lora_execution_mode(self):
        self.model.model_options['donut_lora_execution_mode']='Experimental bypass'
        guarded=self.install()
        self.assertEqual(guarded.injections,{})
        self.assertEqual(guarded.model_options['donut_lora_execution_mode'],'Experimental bypass')
    def test_hook_cleanup_on_patcher_garbage_collection(self):
        guarded=self.install();guarded.pre_run();root=guarded.model
        root.current_patcher=None
        ref=weakref.ref(guarded);del guarded;gc.collect()
        self.assertIsNone(ref());self.assert_no_hooks()
    def test_guarded_and_unguarded_clones_share_weights_not_behavior(self):
        guarded=self.install();self.adapt(guarded);off=guard.remove_model_guard(guarded)
        expected=self.fusion(self.x)
        self.evaluate(guarded)
        self.assertTrue(torch.equal(self.evaluate(off),expected))
    def test_unobserved_component_is_an_error_not_silent_success(self):
        guarded=self.install()
        block=self.fusion.layerwise_blocks[0]
        def bypass_hooks(x, mask=None, transformer_options=None):
            x=x+block.attn.forward(block.prenorm(x),mask=mask,transformer_options=transformer_options)
            return x+block.mlp(block.postnorm(x))
        block.forward=bypass_hooks
        with self.assertRaisesRegex(RuntimeError,'did not observe'):
            self.evaluate(guarded)
        self.assert_no_hooks()
    def test_nested_unguarded_clone_does_not_inherit_guard(self):
        guarded=self.install();self.adapt(guarded);off=guard.remove_model_guard(guarded)
        expected=self.fusion(self.x)
        def nested():
            first=self.fusion(self.x)
            inner=self.evaluate(off)
            last=self.fusion(self.x)
            return first,inner,last
        first,inner,last=self.evaluate(guarded,fn=nested)
        self.assertTrue(torch.equal(inner,expected))
        torch.testing.assert_close(first,last)
        self.assert_no_hooks()
    def test_gain_clamp_and_zero_energy(self):
        ref=torch.ones(2,3);out=ref*100
        adjusted,_,_,gain=guard.match_rms(ref,out,2)
        self.assertTrue(torch.equal(gain,torch.full_like(gain,.25)))
        self.assertIs(guard.match_rms(torch.zeros_like(ref),out,2)[0],out)
    def test_nonfinite_is_not_hidden(self):
        with self.assertRaisesRegex(RuntimeError,'Nonfinite'):
            guard.match_rms(torch.ones(1,3),torch.full((1,3),float('nan')),1)
    def test_unknown_reference_provenance_fails_not_silent_base_selection(self):
        self.model.injections['unknown_swap']=[object()]
        with self.assertRaisesRegex(ValueError,'unrecorded'):self.install()
    def test_reference_no_live_forward_callable_is_retained(self):
        self.fusion.projector.forward=lambda x:x.sum(-1,keepdim=True)*100
        ref=guard.capture_reference(self.model)
        for m in ref.modules():self.assertNotIn('forward',m.__dict__)
    def test_mask_is_shared_with_checkpoint_component(self):
        guarded=self.install();self.adapt(guarded)
        mask=torch.zeros(5,5);mask[:,-1]=-1e4
        result=self.evaluate(guarded,fn=lambda:self.fusion(self.x,mask=mask))
        self.assertTrue(torch.isfinite(result).all())
    def test_plain_compute_dtypes(self):
        for dtype in (torch.float32,torch.float16,torch.bfloat16):
            fusion=Fusion().to(dtype);model=Patcher(Root(fusion));guarded=self.install(model)
            x=self.x.to(dtype)
            expected=fusion(x)
            self.assertTrue(torch.equal(self.evaluate(guarded,x),expected))



class KitchenMetadataTests(unittest.TestCase):
    def test_quantized_scale_metadata_is_not_shared_by_clone(self):
        from dataclasses import dataclass
        @dataclass
        class Params:
            scale: torch.Tensor
            block_scale: torch.Tensor
            orig_dtype: object = torch.float32
        class QuantizedTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, data, params):
                out = torch.Tensor._make_subclass(cls, data, False)
                out._qdata, out._params = data, params
                return out
            def _copy_with(self, qdata=None, params=None):
                return QuantizedTensor(self._qdata if qdata is None else qdata,
                                       self._params if params is None else params)
        source = QuantizedTensor(torch.ones(2,2),Params(torch.ones(1),torch.ones(2)))
        ref = guard._cpu_copy(source)
        source._qdata.zero_();source._params.scale.fill_(7);source._params.block_scale.fill_(9)
        self.assertTrue(torch.equal(ref._qdata,torch.ones(2,2)))
        self.assertTrue(torch.equal(ref._params.scale,torch.ones(1)))
        self.assertTrue(torch.equal(ref._params.block_scale,torch.ones(2)))

if __name__=='__main__':unittest.main()

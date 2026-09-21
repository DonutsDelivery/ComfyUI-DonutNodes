"""Saved sampler-control compatibility, now independent of NAG/mode.

Comfy sampler/Fusion parents are interface doubles. Tensor/lifecycle tests are
in test_txtfusion_model_guard.py; these check dispatch and input preservation.
"""
from pathlib import Path
import importlib.util
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import donut_txtfusion_model_guard as guard


class ParentSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'seed':('INT',{}),'steps':('INT',{})},
                'optional': {'old_flag':('BOOLEAN',{'default':False}), 'grounding_end_px':('INT',{'default':1088})}}
    def sample(self, model, seed=123, steps=8, mode='simple', edit_mode=False,
               sda_enabled=False, model_2=None, model_3=None, **nag_options):
        self.received=(model,seed,steps,mode,edit_mode,sda_enabled,model_2,model_3,nag_options)
        return self.received


class ParentFusion:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'model':('MODEL',),'compatibility_preset':(['Off','Custom','Rebalance','Balanced'],{})},
                'optional':{'nag_match_taps':('BOOLEAN',{'default':True}),
                            'nag_batch_txtfusion':('BOOLEAN',{'default':False})}}
    def apply(self, model, conditioning_in_1=None, **kwargs):
        self.received=(model,conditioning_in_1,kwargs)
        return (model,conditioning_in_1,None,None,None,'original diagnostics')


def load_module(filename):
    name='_model_guard_node_tests'
    package=types.ModuleType(name);package.__path__=[str(ROOT)]
    parent=types.ModuleType(name+'.donut_grounding_schedule');parent.DonutSampler=ParentSampler
    fusion=types.ModuleType(name+'.donut_krea2_fusion_experiments');fusion.DonutKrea2FusionControl=ParentFusion
    folder=types.ModuleType('folder_paths');folder.get_filename_list=lambda key:['same.safetensors','not.bin']
    overrides={name:package,parent.__name__:parent,fusion.__name__:fusion,
               name+'.donut_txtfusion_model_guard':guard,'folder_paths':folder}
    ctx=patch.dict(sys.modules,overrides);ctx.start()
    spec=importlib.util.spec_from_file_location(name+'.'+filename,ROOT/(filename+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,ctx


class SamplerTests(unittest.TestCase):
    def setUp(self):
        self.module,ctx=load_module('donut_txtfusion_guard_sampler');self.addCleanup(ctx.stop)
        self.sampler=self.module.DonutSampler();self.model=object();self.patched=object()
    def test_off_is_exact_parent_dispatch_without_install(self):
        with patch.object(self.module,'attach_model_guard',side_effect=AssertionError('guard')):
            result=self.sampler.sample(self.model,nag_enabled=False,nag_alpha=0.)
        self.assertIs(result[0],self.model);self.assertEqual(result[-1],{'nag_enabled':False,'nag_alpha':0.})
    def test_alpha_zero_is_forwarded_unchanged_and_guard_attached(self):
        with patch.object(self.module,'attach_model_guard',return_value=self.patched) as install:
            result=self.sampler.sample(self.model,nag_enabled=True,nag_alpha=0.,txtfusion_internal_guard=True)
        install.assert_called_once_with(self.model)
        self.assertIs(result[0],self.patched);self.assertEqual(result[-1]['nag_alpha'],0.)
    def test_nag_disabled_is_supported_without_inventing_guidance(self):
        with patch.object(self.module,'attach_model_guard',return_value=self.patched):
            result=self.sampler.sample(self.model,nag_enabled=False,nag_alpha=.45,txtfusion_internal_guard=True)
        self.assertFalse(result[-1]['nag_enabled']);self.assertEqual(result[-1]['nag_alpha'],.45)
    def test_edit_sda_advanced_phi_zero_are_not_rejected(self):
        for settings in ({'mode':'advanced'},{'edit_mode':True},{'sda_enabled':True},{'nag_phi':0.}):
            with self.subTest(settings=settings),patch.object(self.module,'attach_model_guard',return_value=self.patched):
                result=self.sampler.sample(self.model,txtfusion_internal_guard=True,**settings)
                self.assertIs(result[0],self.patched)
    def test_multimodel_arguments_are_each_guarded(self):
        other,third=object(),object()
        with patch.object(self.module,'attach_model_guard',side_effect=lambda m:('guarded',m)) as install:
            result=self.sampler.sample(self.model,mode='multi_model',model_2=other,model_3=third,txtfusion_internal_guard=True)
        self.assertEqual(install.call_count,3);self.assertEqual(result[6],('guarded',other));self.assertEqual(result[7],('guarded',third))
    def test_sampler_values_and_extra_options_preserved(self):
        with patch.object(self.module,'attach_model_guard',return_value=self.patched):
            result=self.sampler.sample(self.model,987,8,nag_alpha=.45,scheduler='beta',sampler_name='bleh_preset_0',txtfusion_internal_guard=True)
        self.assertEqual(result[1:3],(987,8));self.assertEqual(result[-1],{'nag_alpha':.45,'scheduler':'beta','sampler_name':'bleh_preset_0'})
    def test_reference_field_is_explicitly_deprecated_not_silently_loaded(self):
        with patch.object(self.module,'attach_model_guard',return_value=self.patched),self.assertLogs(self.module.LOGGER,level='WARNING') as logs:
            result=self.sampler.sample(self.model,txtfusion_internal_guard=True,txtfusion_reference_checkpoint='same.safetensors')
        self.assertIn('not used',logs.output[0]);self.assertNotIn('txtfusion_reference_checkpoint',result[-1])
    def test_schema_positions_preserved(self):
        old=ParentSampler.INPUT_TYPES();new=self.sampler.INPUT_TYPES()
        self.assertEqual(old['required'],new['required'])
        self.assertEqual(list(new['optional']),[*old['optional'],'txtfusion_internal_guard','txtfusion_reference_checkpoint'])
        self.assertIs(new['optional']['txtfusion_internal_guard'][1]['default'],False)
    def test_false_does_not_remove_upstream_model_guard(self):
        with patch.object(self.module,'attach_model_guard',side_effect=AssertionError('attach')):
            result=self.sampler.sample(self.patched,txtfusion_internal_guard=False)
        self.assertIs(result[0],self.patched)
    def test_boolean_input_validated(self):
        with self.assertRaisesRegex(ValueError,'boolean'):
            self.sampler.sample(self.model,txtfusion_internal_guard='false')


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.module,ctx=load_module('donut_txtfusion_model_guard_node');self.addCleanup(ctx.stop)
        self.node=self.module.DonutKrea2FusionControl();self.model=object();self.patched=object()
    def test_general_switch_appended_after_all_old_widgets(self):
        old=ParentFusion.INPUT_TYPES();new=self.node.INPUT_TYPES()
        self.assertEqual(old['required'],new['required'])
        self.assertEqual(list(new['optional']),[*old['optional'],'txtfusion_rms_guard'])
        self.assertIs(new['optional']['txtfusion_rms_guard'][1]['default'],False)
    def test_reference_prepared_before_uncensorfix_parent_execution(self):
        order=[]
        with patch.object(self.module,'attach_model_guard',side_effect=lambda m:order.append('capture') or self.patched),\
             patch.object(ParentFusion,'apply',side_effect=lambda *a,**kw:order.append('uncensorfix') or (kw['model'],None,None,None,None,'diag')):
            self.node.apply(model=self.model,txtfusion_rms_guard=True,uncensorfix_controls='Fusion + UncensorFix weights')
        self.assertEqual(order,['capture','uncensorfix'])
    def test_rebalance_and_conditioning_inputs_not_rewritten(self):
        cond=object()
        with patch.object(self.module,'attach_model_guard',return_value=self.patched):
            result=self.node.apply(model=self.model,conditioning_in_1=cond,compatibility_preset='Rebalance',
                                   tap_normalization='none',nag_match_taps=True,txtfusion_rms_guard=True)
        self.assertIs(result[1],cond);self.assertEqual(self.node.received[2]['compatibility_preset'],'Rebalance')
        self.assertEqual(self.node.received[2]['tap_normalization'],'none')
        self.assertNotIn('txtfusion_rms_guard',self.node.received[2])
    def test_guard_is_independent_of_fusion_preset_off(self):
        with patch.object(self.module,'attach_model_guard',return_value=self.patched):
            self.node.apply(model=self.model,compatibility_preset='Off',txtfusion_rms_guard=True)
        self.assertIs(self.node.received[0],self.patched)
    def test_off_preserves_exact_parent_diagnostics(self):
        with patch.object(self.module,'attach_model_guard',side_effect=AssertionError('capture')):
            result=self.node.apply(model=self.model,compatibility_preset='Balanced',txtfusion_rms_guard=False)
        self.assertIs(result[0],self.model);self.assertEqual(result[-1],'original diagnostics')
    def test_positional_model_preserved(self):
        cond=object()
        with patch.object(self.module,'attach_model_guard',return_value=self.patched):
            result=self.node.apply(self.model,cond,txtfusion_rms_guard=True)
        self.assertIs(result[0],self.patched);self.assertIs(result[1],cond)
    def test_standalone_node_uses_identical_model_guard(self):
        with patch.object(self.module,'attach_model_guard',return_value=self.patched) as install:
            result=self.module.DonutTxtfusionRMSGuard().apply(self.model,True)
        install.assert_called_once_with(self.model);self.assertIs(result[0],self.patched)


if __name__=='__main__':unittest.main()

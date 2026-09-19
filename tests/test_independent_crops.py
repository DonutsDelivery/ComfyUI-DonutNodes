"""CPU geometry and bridge contracts. Original Comfy/model APIs are stubbed."""
import copy
import importlib.util
import json
from pathlib import Path
import random
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
PKG='_donut_crop_test'
package=types.ModuleType(PKG);package.__path__=[str(ROOT)];sys.modules[PKG]=package

def load(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/file)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

g=load(PKG+'.donut_reference_geometry','donut_reference_geometry.py')
def document(bounds=(0,0,1,1),size=(100,200),image='B',aspect='Free'):
    return json.dumps(dict(version=1,image=image,source_size=list(size),bounds=list(bounds),aspect=aspect))

class GeometryTests(unittest.TestCase):
    def test_empty_crop_is_whole_source(self):self.assertEqual(g.crop_pixels('','B',(100,200)),(0,0,100,200))
    def test_independent_box_is_in_source_pixels(self):self.assertEqual(g.crop_pixels(document((.1,.2,.9,.8)),'B',(100,200)),(10,40,90,160))
    def test_rounding_is_half_up(self):self.assertEqual(g.crop_pixels(document((.025,.0125,.975,.9875)),'B',(100,200)),(3,3,98,198))
    def test_stale_image_rejected(self):
        with self.assertRaisesRegex(ValueError,'different image'):g.crop_pixels(document(),'A',(100,200))
    def test_changed_dimensions_rejected(self):
        with self.assertRaises(ValueError):g.crop_pixels(document(),'B',(101,200))
    def test_invalid_document_rejected(self):
        for d in ('[0,0,1,1]','garbage','null','{"version":2}',document((0,0,0,1)),document((0,0,1,1),aspect='invalid')):
            with self.subTest(d=d),self.assertRaises(ValueError):g.crop_pixels(d,'B',(100,200))
    def test_invalid_coordinates_rejected(self):
        for b in ((False,0,1,1),(0,0,float('nan'),1),(0,0,float('inf'),1),(-1,0,1,1),(0,0,2,1),(0,0,.001,.001)):
            with self.subTest(b=b),self.assertRaises(ValueError):g.crop_pixels(document(b),'B',(100,200))
    def test_fit_portrait_on_landscape(self):
        f=g.fit_geometry((100,200),(200,120));self.assertEqual(f.content,(60,120));self.assertEqual(f.offset,(70,0))
    def test_fit_landscape_on_portrait(self):
        f=g.fit_geometry((200,100),(120,200));self.assertEqual(f.content,(120,60));self.assertEqual(f.offset,(0,70))
    def test_full_height_subject_retained(self):
        a=np.zeros((200,100,3),dtype=np.uint8);a[:12,:,0]=255;a[-12:,:,2]=255
        image,fit=g.crop_fit_image(Image.fromarray(a),(0,0,100,200),(200,120))
        self.assertGreater(image[0,0,100,0].item(),.95);self.assertGreater(image[0,-1,100,2].item(),.95)
        self.assertTrue(torch.allclose(image[0,:,0],torch.full((120,3),128/255)))
    def test_tensor_fit_retains_input_and_batch(self):
        x=torch.rand(2,200,100,3);original=x.clone();out=g.fit_tensor(x,200,120)
        self.assertEqual(tuple(out.shape),(2,120,200,3));self.assertTrue(torch.equal(x,original));self.assertTrue(torch.all(out[:,:,:70]==.5))
    def test_same_canvas_fit_has_no_resample(self):
        x=torch.rand(1,64,96,3);self.assertTrue(torch.equal(g.fit_tensor(x,96,64),x))
    def test_fit_geometry_rounding_bound(self):
        rng=random.Random(42)
        for _ in range(100):
            source=(rng.randint(20,4000),rng.randint(20,4000));canvas=(rng.randint(16,2000),rng.randint(16,2000));f=g.fit_geometry(source,canvas)
            scale=min(canvas[0]/source[0],canvas[1]/source[1])
            for i in (0,1):
                self.assertLessEqual(abs(f.content[i]-source[i]*scale),1)
                self.assertGreaterEqual(f.offset[i],0);self.assertLessEqual(f.content[i]+f.offset[i],canvas[i])
    def test_mask_padding_matches_image(self):
        f=g.fit_geometry((100,200),(200,120));mask=g.fit_mask(torch.ones(1,120,60),f)
        self.assertTrue(torch.all(mask[:,:,:70]==0));self.assertTrue(torch.all(mask[:,:,70:130]==1));self.assertTrue(torch.all(mask[:,:,130:]==0))
    def test_feather_never_adds_unselected_pixels(self):
        f=g.fit_geometry((100,200),(200,120));hard=g.fit_mask(torch.ones(1,120,60),f);soft=g.fit_mask(torch.ones(1,120,60),f,8)
        self.assertTrue(torch.all(soft<=hard));self.assertTrue(torch.any((soft>0)&(soft<1)))
    def test_invalid_mask_dimensions_rejected(self):
        with self.assertRaises(ValueError):g.fit_mask(torch.ones(1,30,30),g.fit_geometry((100,200),(200,120)))
    def test_legacy_or_disabled_does_not_parse_geometry(self):
        for values in ({},{'geometry_mode':g.LEGACY,'enabled':True},{'geometry_mode':g.INDEPENDENT,'enabled':False}):self.assertIsNone(g.output_dimensions(values))
    def test_follow_A_uses_crop_ratio(self):
        dims=g.output_dimensions({'geometry_mode':g.INDEPENDENT,'enabled':True,'megapixels':1,'multiple':32},(300,200),(0,0,100,200))
        self.assertEqual(dims,(736,1440))
    def test_independent_canvas_leaves_crop_unchanged(self):
        crop=(0,0,100,200);before=tuple(crop)
        dims=g.output_dimensions({'geometry_mode':g.INDEPENDENT,'enabled':True,'output_canvas':'Independent output','aspect_ratio':'16:9 Wide','megapixels':1},(100,200),crop)
        self.assertGreater(dims[0],dims[1]);self.assertEqual(crop,before)
    def test_custom_independent_dimensions(self):
        v={'geometry_mode':g.INDEPENDENT,'enabled':True,'output_canvas':'Independent output','resolution_mode':'Custom','width':1536,'height':864,'multiple':32}
        self.assertEqual(g.output_dimensions(v,(100,200),(0,0,100,200)),(1536,864))
    def test_crop_only_uses_cropped_native_size_without_cutting(self):
        v={'geometry_mode':g.INDEPENDENT,'enabled':True,'resolution_mode':'Reference A · crop only','multiple':32}
        self.assertEqual(g.output_dimensions(v,(200,200),(0,0,135,170)),(128,160))
    def test_invalid_budget_and_modes(self):
        for kwargs in ({'megapixels':float('nan')},{'output_canvas':'unknown'},{'multiple':7},{'geometry_mode':'unknown'}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):g.output_dimensions({'geometry_mode':g.INDEPENDENT,'enabled':True,**kwargs},(100,200),(0,0,100,200))

class Model:
    def __init__(self):self.model_options={'fusion':'keep'}
    def clone(self):m=Model();m.model_options=copy.deepcopy(self.model_options);return m

class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.env=patch.dict(sys.modules);self.env.start();self.addCleanup(self.env.stop)
        self.opens=[];self.delegations=[];self.masks=[]
        self.images={'A':Image.new('RGB',(120,80),'red'),'B':Image.new('RGB',(60,160),'blue')}
        self.original_mask=torch.zeros(1,80,120);self.original_mask[:,20:60,30:90]=1
        test=self
        original=types.ModuleType(PKG+'.DonutEditStudio')
        class Original:
            @classmethod
            def INPUT_TYPES(cls):return {'required':{'enabled':('BOOLEAN',{'default':False})},'optional':{}}
            def prepare(self,enabled,image_a,image_b,use_reference_b,prompt,resolution_mode,aspect_ratio,megapixels,width,height,multiple,grounding_px,lora_name,lora_strength,crop_a_x=.5,crop_a_y=.5,crop_b_x=.5,crop_b_y=.5,model=None,text_seed=0,inpaint_enabled=False,mask_data='',mask_feather=8,grounding_schedule='constant',grounding_start_px=512,grounding_end_px=1088):
                test.delegations.append(dict(enabled=enabled,inpaint_enabled=inpaint_enabled,width=width,height=height))
                if not enabled:return (None,None,False,width,height,grounding_px,None,prompt,None,grounding_schedule,grounding_start_px,grounding_end_px)
                return (torch.ones(1,height,width,3),torch.ones(1,height,width,3) if use_reference_b else None,True,width,height,grounding_px,model.clone(),prompt,None,grounding_schedule,grounding_start_px,grounding_end_px)
        class Subject(Original):
            def prepare(self,*args,**kwargs):
                test.delegations.append('subject')
                return super().prepare(*args,**{k:v for k,v in kwargs.items() if not k.startswith('mask_b')})
        class Reference:
            @classmethod
            def INPUT_TYPES(cls):return {'required':{},'optional':{}}
            @classmethod
            def IS_CHANGED(cls,*args):return 'reference-fingerprint'
            def prepare(self,enabled=False,image_a='',image_b='',use_reference_b=False,edit_active=False):return ('legacy-a','legacy-b',enabled and not edit_active)
        original.DonutEditStudio=Original;original.DonutReferenceStudio=Reference
        def open_image(name):test.opens.append(name);return test.images[name].copy()
        original._open_reference=open_image
        def rasterize(data,name,source_size,box,size,feather):
            test.masks.append((name,source_size,box,size,feather));x1,y1,x2,y2=box
            return F.interpolate(test.original_mask[:,None,y1:y2,x1:x2],size=size[::-1],mode='bilinear',align_corners=False)[:,0]
        original.rasterize_mask=rasterize
        subject=types.ModuleType(PKG+'.donut_reference_mask');subject.DonutSubjectMaskStudio=Subject
        subject.MODEL_NAME='birefnet.safetensors';subject.MAX_PIXELS=32_000_000;subject.BACKGROUNDS={'Neutral gray':.5,'White':1,'Black':0}
        self.record={'version':1,'image':'B','mask':'donutmask:'+'a'*64};self.bmask=torch.ones(1,160,60)
        subject.auto_mask=lambda name,source,model:(self.record,self.bmask)
        def load_mask(record,name,source):
            if record!=self.record or name!='B' or source.size!=(60,160):raise ValueError('stale mask')
            return self.bmask
        subject.load_mask=load_mask;subject.store_mask=lambda name,source,mask:self.record
        subject.composite=lambda source,mask,grow,feather,background:source.copy()
        for m in (original,subject):sys.modules[m.__name__]=m;setattr(package,m.__name__.rsplit('.',1)[-1],m)
        self.mod=load(PKG+'.donut_crop_studio','donut_crop_studio.py')
        self.kw=dict(enabled=True,image_a='A',image_b='B',use_reference_b=True,prompt='prompt',resolution_mode='Custom',aspect_ratio='16:9 Wide',megapixels=1,width=192,height=128,multiple='32',grounding_px=768,lora_name='None',lora_strength=0,model=Model())
    def run_new(self,**kwargs):return self.mod.DonutCropEditStudio().prepare(**{**self.kw,'geometry_mode':g.INDEPENDENT,'output_canvas':'Independent output',**kwargs})
    def test_legacy_delegates_unchanged(self):
        out=self.mod.DonutCropEditStudio().prepare(**self.kw)
        self.assertEqual(len(out),12);self.assertEqual(self.opens,[]);self.assertEqual(self.delegations[0],'subject');self.assertNotIn(g.FIT_KEY,out[6].model_options)
    def test_off_does_not_open_or_validate_crops(self):
        out=self.run_new(enabled=False,crop_data_a='broken');self.assertEqual(len(out),12);self.assertFalse(out[2]);self.assertEqual(self.opens,[])
    def test_A_and_B_keep_separate_shapes_inside_canvas(self):
        out=self.run_new()['result'];self.assertEqual(tuple(out[0].shape),(1,128,192,3));self.assertEqual(tuple(out[1].shape),(1,128,192,3))
        self.assertTrue(torch.all(out[1][0,:,96,2]>.99));self.assertTrue(torch.allclose(out[1][0,:,0],torch.full((128,3),128/255)))
    def test_metadata_clone_keeps_fusion_and_does_not_mutate_model(self):
        result=self.run_new()['result'];self.assertTrue(result[6].model_options[g.FIT_KEY]);self.assertEqual(result[6].model_options['fusion'],'keep');self.assertNotIn(g.FIT_KEY,self.kw['model'].model_options)
    def test_follow_A_changes_canvas_not_B_crop(self):
        out=self.run_new(output_canvas='Follow A crop',crop_data_a=document((.25,0,.75,1),(120,80),'A'))['result']
        self.assertLess(out[3],out[4]);self.assertEqual(out[1].shape[1:3],out[0].shape[1:3])
    def test_inpaint_uses_same_transform_and_source_mask(self):
        out=self.run_new(crop_data_a=document((.25,.25,.75,.75),(120,80),'A'),inpaint_enabled=True,mask_data='selection',mask_feather=0)['result']
        self.assertFalse(self.delegations[-1]['inpaint_enabled']);self.assertEqual(self.masks[-1],('A',(120,80),(30,20,90,60),(192,128),0))
        self.assertTrue(torch.equal(out[8]['image'],out[0]));self.assertTrue(torch.all(out[8]['mask']==1))
    def test_padded_A_is_protected_outside_content(self):
        out=self.run_new(crop_data_a=document((.375,0,.625,1),(120,80),'A'),inpaint_enabled=True,mask_data='selection',mask_feather=8)['result']
        mask=out[8]['mask'];self.assertTrue(torch.all(mask[:,:,:72]==0));self.assertTrue(torch.all(mask[:,:,120:]==0));self.assertTrue(torch.any(mask>0))
    def test_auto_B_mask_reuses_original_mask_pipeline(self):
        out=self.run_new(mask_b_mode='Auto subject');self.assertEqual(out['ui']['donut_subject_mask'],[self.record])
    def test_saved_B_mask_is_still_source_aligned(self):
        out=self.run_new(mask_b_mode='Saved mask',mask_b_data=json.dumps(self.record),crop_data_b=document((0,.25,1,.75),(60,160),'B'));self.assertEqual(out['ui']['donut_subject_mask'],[self.record])
    def test_stale_saved_B_mask_is_error(self):
        with self.assertRaises(ValueError):self.run_new(mask_b_mode='Saved mask',mask_b_data='{}')
    def test_external_mask_missing_does_not_fallback(self):
        with self.assertRaisesRegex(ValueError,'MASK'):self.run_new(mask_b_mode='External mask')
    def test_unused_B_crop_is_not_parsed(self):
        out=self.run_new(use_reference_b=False,crop_data_b='bad')['result'];self.assertIsNone(out[1])
    def test_unknown_geometry_mode_is_error(self):
        with self.assertRaises(ValueError):self.run_new(geometry_mode='bad')
    def test_reference_guidance_crops_at_native_sizes(self):
        out=self.mod.DonutCropReferenceStudio().prepare(True,'A','B',True,False,g.INDEPENDENT,document((.25,0,.75,1),(120,80),'A'),document((0,.25,1,.75),(60,160),'B'))
        self.assertEqual(tuple(out[0].shape),(1,80,60,3));self.assertEqual(tuple(out[1].shape),(1,80,60,3))
    def test_reference_guidance_paused_needs_no_valid_crop(self):
        out=self.mod.DonutCropReferenceStudio().prepare(True,'missing','missing',True,True,g.INDEPENDENT,'bad','bad');self.assertFalse(out[2]);self.assertEqual(self.opens,[])
    def test_new_inputs_append_and_legacy_default(self):
        schema=self.mod.DonutCropEditStudio.INPUT_TYPES()['optional'];self.assertEqual(list(schema)[-4:],['geometry_mode','crop_data_a','crop_data_b','output_canvas']);self.assertEqual(schema['geometry_mode'][1]['default'],g.LEGACY)

if __name__=='__main__':unittest.main()

"""Real Comfy Krea2/ModelPatcher CPU smoke tests, no checkpoint download.

Run inside a ComfyUI environment with its root on PYTHONPATH (and CPU mode when
no CUDA is available). Skipped, NOT counted as passed, outside that environment.
These do not replace a GPU workflow/PNG test.
"""
import importlib.util
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import donut_txtfusion_model_guard as guard

try:
    HAS_COMFY = importlib.util.find_spec('comfy') is not None
except (ValueError,ModuleNotFoundError):
    HAS_COMFY = False


@unittest.skipUnless(HAS_COMFY,'Real ComfyUI is not installed in this test environment')
class RealComfyTests(unittest.TestCase):
    def test_real_native_component_and_patcher_without_nag(self):
        import comfy.ops
        from comfy.ldm.krea2.model import TextFusionTransformer
        from comfy.model_patcher import ModelPatcher
        from comfy.weight_adapter.lora import LoRAAdapter
        torch.manual_seed(73)
        fusion=TextFusionTransformer(12,16,4,4,device='cpu',dtype=torch.float32,operations=comfy.ops.manual_cast)
        with torch.no_grad():
            for name,parameter in fusion.named_parameters():
                parameter.zero_() if name.endswith('scale') else parameter.normal_(0,.06)
        class Root(nn.Module):
            def __init__(self):
                super().__init__();self.diffusion_model=nn.Module();self.diffusion_model.txtfusion=fusion
                self.current_patcher=None
            def get_dtype(self):return torch.float32
            def memory_required(self,shape):return 0
        model=ModelPatcher(Root(),load_device=torch.device('cpu'),offload_device=torch.device('cpu'))
        key='diffusion_model.txtfusion.layerwise_blocks.0.attn.wo.weight'
        original=fusion.layerwise_blocks[0].attn.wo.weight.detach().clone()
        adapter=LoRAAdapter(set(),(torch.randn(16,2)*.15,torch.randn(2,16)*.15,2.,None,None,None))
        model.add_patches({key:adapter},1.)
        model.patch_weight_to_device(key,device_to=torch.device('cpu'))
        patched=guard.attach_model_guard(model)
        reference=patched.get_additional_models_with_key(guard.KEY)[0].model.txtfusion
        torch.testing.assert_close(reference.layerwise_blocks[0].attn.wo.weight,original,rtol=0,atol=0)
        value=torch.randn(1,5,12,16)
        off=fusion(value)
        patched.pre_run()
        try:
            on=fusion(value,transformer_options=patched.model_options['transformer_options'])
        finally:
            patched.cleanup()
        self.assertFalse(torch.allclose(off,on))
        self.assertEqual(patched.get_attachment(guard.KEY).last_report['contributions'],9)
        self.assertFalse(fusion._forward_hooks)
        for block in [*fusion.layerwise_blocks,*fusion.refiner_blocks]:
            self.assertFalse(block.attn._forward_hooks);self.assertFalse(block.mlp._forward_hooks)

    def test_real_kitchen_quantized_tensor_has_independent_scale_metadata(self):
        from comfy.quant_ops import QuantizedTensor,TensorCoreFP8Layout
        if not hasattr(QuantizedTensor,'from_float'):
            self.skipTest('Comfy kitchen quantized operations unavailable')
        value=QuantizedTensor.from_float(torch.randn(64,64),TensorCoreFP8Layout.__name__,scale='recalculate')
        copied=guard._cpu_copy(value)
        self.assertNotEqual(copied._qdata.data_ptr(),value._qdata.data_ptr())
        self.assertNotEqual(copied._params.scale.data_ptr(),value._params.scale.data_ptr())
        torch.testing.assert_close(copied.dequantize(),value.dequantize(),rtol=0,atol=0)


if __name__=='__main__':unittest.main()

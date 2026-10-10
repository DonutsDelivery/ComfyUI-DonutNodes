import ast,copy,gc,hashlib,json,logging,math,tempfile,types,unittest,weakref
from pathlib import Path
from typing import Optional,Union
import torch
import torch.nn as nn
REPO=Path('/home/user/Programs/ComfyUI/custom_nodes/donutnodes')
CORE=Path('/tmp/donut-cache-audit-core')
def extract(path,names,ns):
 tree=ast.parse(path.read_text()); tree.body=[x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name in names]
 exec(compile(tree,str(path),'exec'),ns)
class Base:
 def bypass_forward(self,original,x,*a,**kw): return original(x,*a,**kw)
class Train(Base):pass
class Adapter(Base):
 def __init__(self): self.weights=(torch.tensor([[2.]]),)
 def bypass_forward(self,original,x,*a,**kw):return original(x,*a,**kw)+nn.functional.linear(x,self.weights[0])*self.multiplier
class Injection:
 def __init__(self,inject,eject):self.inject=inject;self.eject=eject
comfy=types.SimpleNamespace(model_management=types.SimpleNamespace(get_torch_device=lambda:torch.device('cpu')))
ns=dict(logging=logging,Optional=Optional,Union=Union,torch=torch,nn=nn,comfy=comfy,WeightAdapterBase=Base,WeightAdapterTrainBase=Train,BypassAdapter=Union[Base,Train],PatcherInjection=Injection)
extract(CORE/'comfy/weight_adapter/bypass.py',{'get_module_type_info','BypassForwardHook','BypassInjectionManager'},ns)
ns.update(copy=copy,weakref=weakref,_CompositeBypassAdapter=type('Composite',(),{}),_LOKR_ADAPTER_BASE=None)
extract(REPO/'DonutSafeApplyLoRAStack.py',{'_copy_runtime_adapter','_trace_lokr_calls','_eject_runtime_bypass','_make_rebinding_bypass_injections'},ns)
class Owner:
 def __init__(self,root):self.model=root;self.model_options={}
class Transition(unittest.TestCase):
 def setup_plan(self,strength):
  root=nn.Module();root.proj=nn.Linear(1,1,bias=False);root.proj.weight.data.fill_(1)
  manager=ns['BypassInjectionManager']();manager.add_adapter('proj.weight',Adapter(),strength)
  plan=ns['_make_rebinding_bypass_injections'](manager,root)[0]
  return root,manager,plan
 def output(self,root):return root.proj(torch.ones(1,1)).item()
 def test_strength_changes_and_off(self):
  root,_,plan=self.setup_plan(.25);first=Owner(root)
  plan.inject(first);self.assertEqual(self.output(root),1.5)
  plan.eject(first);self.assertEqual(self.output(root),1)
  for strength in [1.,-.5,2.]:
   manager=ns['BypassInjectionManager']();manager.add_adapter('proj.weight',Adapter(),strength)
   new=ns['_make_rebinding_bypass_injections'](manager,root)[0];owner=Owner(root)
   new.inject(owner);self.assertEqual(self.output(root),1+2*strength)
   new.eject(owner);self.assertEqual(self.output(root),1)
 def test_clone_rebinding_and_late_ejection(self):
  root,_,plan=self.setup_plan(1);a,b=Owner(root),Owner(root)
  plan.inject(a);plan.inject(b);plan.eject(a)
  self.assertEqual(self.output(root),3)
  plan.eject(b);self.assertEqual(self.output(root),1)
 def test_abandoned_clone_finalizer_restores_forward(self):
  root,_,plan=self.setup_plan(1);owner=Owner(root)
  plan.inject(owner);self.assertEqual(self.output(root),3)
  del owner;gc.collect();self.assertEqual(self.output(root),1)
 def test_runtime_does_not_mutate_canonical_adapter(self):
  root,manager,plan=self.setup_plan(1);owner=Owner(root);canonical=manager.adapters['proj'][0]
  original=canonical.weights
  plan.inject(owner);plan.eject(owner)
  self.assertIs(canonical.weights,original)
class FileInvalidation(unittest.TestCase):
 def test_edit_studio_signature_does_not_track_lora_file(self):
  with tempfile.TemporaryDirectory() as directory:
   p=Path(directory)/'edit.safetensors';p.write_bytes(b'old')
   env=dict(hashlib=hashlib,directory_fingerprint=lambda:(),_reference_path=lambda _:Path(directory)/'none')
   tree=ast.parse((REPO/'DonutEditStudio.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DonutEditStudio')
   method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='IS_CHANGED');method.decorator_list=[]
   exec(compile(ast.Module(body=[method],type_ignores=[]),'DonutEditStudio.py','exec'),env)
   signature=lambda:env['IS_CHANGED'](None,enabled=True,lora_name=str(p),lora_strength=1)
   old=signature();p.write_bytes(b'new and changed')
   self.assertEqual(signature(),old,'Documents missing disk-file invalidation; this is not a passing repaired contract')
if __name__=='__main__':unittest.main()

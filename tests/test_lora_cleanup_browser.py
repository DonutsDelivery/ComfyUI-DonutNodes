"""Real Chromium DOM/layout tests with a SMALL Comfy node adapter and mocked HTTP.

Not a full ComfyUI renderer, live CivitAI, or GPU test. The shipped toolbar,
metadata renderer, native-editor logic, event handlers and ResizeObserver run
unchanged. The adapter validates combo construction and serialization contracts.
Run: python -m unittest discover -s tests -p test_lora_cleanup_browser.py -v
Requires Playwright and Chromium in the developer/test environment only.
"""
import json
import os
from pathlib import Path
import shutil
import unittest

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

ROOT = Path(__file__).resolve().parents[1]
PAGE = r'''<!doctype html><html><head><meta charset="utf-8"><style>
body {background:#202020;color:#ddd;font:12px sans-serif;margin:20px;}
h1 {font-size:15px;font-weight:500;} .node {width:420px;background:#353535;padding:8px 0;border-radius:7px;}
.control {box-sizing:border-box;display:flex;justify-content:space-between;align-items:center;gap:12px;
 margin:0 15px;padding:2px 10px;background:#292929;border:1px solid #595959;border-radius:11px;height:21px;}
.control span {overflow:hidden;white-space:nowrap;text-overflow:ellipsis;}
.control span:first-child {color:#aaa;flex-shrink:0;} .control span:last-child {text-align:right;}
.control button {width:100%;border:0;background:transparent;color:#ddd;cursor:pointer;}
.hint {max-width:420px;line-height:1.5;color:#aaa;margin-top:14px;} a {color:#91c9f4;}
</style></head><body><h1>LoRA UI · browser test harness</h1><div id="mount" class="node"></div>
<div class="hint">Actual toolbar and metadata DOM; native ComfyUI controls represented by a test adapter. Preview and server responses are fixtures.</div>
<script type="module">
import { installNativeLoras, createLoraService } from '/web/donut_native_lora.js';
window.requests = [];
const api = {
 apiURL: value => value.startsWith('/donut/loras/preview')
  ? 'data:image/svg+xml,' + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="92" height="112"><rect width="92" height="112" fill="#647384"/><text x="46" y="53" text-anchor="middle" font-family="sans-serif" font-size="12" fill="white">Preview</text><text x="46" y="70" text-anchor="middle" font-family="sans-serif" font-size="10" fill="white">fixture</text></svg>')
  : '/api' + value,
 async fetchApi(url) {
  requests.push(url);
  let data;
  if (url === '/models/loras') data = Array.from({length:102}, (_,i)=>`styles/portrait-${i}.safetensors`);
  else if (url.startsWith('/object_info')) data = {DonutLoRAStack:{input:{required:{
   lora_name_1:[['None']],block_preset_1:[['None','KREA2-ALL:1,1,1','KREA2-FACE:0,1,0']]}}}};
  else if (url.includes('/analyze?')) data = {supported:true,tensor_count:512,components:[{name:'UNet',modules:256,
   groups:[{name:'blocks',indices:Array.from({length:28},(_,i)=>i)},{name:'txtfusion.layerwise',indices:[0,1]}]}]};
  else data = {hash:'abcdef1234',has_collage:true,civitai:{model_id:12,model_version_id:34,
   model_name:'Portrait style',version_name:'v1.0',base_model:'Krea 2',creator_username:'Example author',
   recommended_weight:.8,trained_words:['portrait style'],description: 'Example description. '.repeat(80)}};
  return {ok:true,status:200,json:async()=>data};
 }
};
const names = ['model_type','slots_json','global_block_vector','civitai_lookup','safe_stack','fusion_aware','max_fusion_boost','safe_limit','execution_mode'];
const initial = Array.from({length:3},(_,i)=>({id:`row-${i}`,enabled:i===0,lora_name:i===0?'styles/portrait-0.safetensors':'None',
 model_weight:1,clip_weight:0,block_preset:'None',block_vector:'1,1,1',inherit_block_vector:true,lora_hash:'',custom:{keep:i}}));
class TestNode {
 constructor() {
  this.widgets=[];this.inputs=[];this.properties={};this.size=[420,900];
  this.graph={beforeChange:()=>{this.undo={};this.onSerialize(this.undo);},afterChange(){}};
  const values=['KREA2',JSON.stringify(initial),'1,1,1','On','On','Use headroom',2,1,'Experimental bypass'];
  names.forEach((name,i)=>this.addWidget(name==='slots_json'?'customtext':'text',name,values[i],()=>{},{ }));
 }
 addWidget(type,name,value,callback,options={}) {
  if(type==='combo'&&!Array.isArray(options.values)) throw new Error('Combo requires initial values');
  const w={type,name,value,callback,options};this.widgets.push(w);
  return w;
 }
 addDOMWidget(name,type,element,options) {const w={name,type,element,options};this.widgets.push(w);return w;}
 removeWidget(w) {if(!this.widgets.includes(w))throw Error('Unknown widget');w.onRemove?.();w.wrapper?.remove();this.widgets.splice(this.widgets.indexOf(w),1);}
 computeSize() {return [this.size[0],this.widgets.reduce((h,w)=>h+(w.hidden?0:(w.options.getMinHeight?.()||24)+4),16)];}
 setSize(size) {this.size=size;this.layout();}
 setDirtyCanvas() {this.layout();}
 layout() {
  const mount=document.getElementById('mount');mount.style.width=this.size[0]+'px';
  for(const w of this.widgets) {
   if(!w.wrapper){w.wrapper=document.createElement('div');mount.append(w.wrapper);if(w.element)w.wrapper.append(w.element);}
   w.wrapper.style.display=w.hidden?'none':'block';
   w.wrapper.style.height=(w.options.getMinHeight?.()||24)+4+'px';
   if(w.element){w.element.style.height=w.options.getMinHeight?.()+'px';}
   else {
    const bar=document.createElement('div');bar.className='control';
    if(w.type==='button') {const button=document.createElement('button');button.textContent=w.label||w.name;
      button.onclick=()=>w.callback();bar.append(button);}
    else {const label=document.createElement('span'),value=document.createElement('span');label.textContent=w.label||w.name;
      value.textContent=w.value;bar.append(label,value);}
    w.wrapper.replaceChildren(bar);
   }
  }
 }
}
window.node=new TestNode();installNativeLoras(node,{input:{required:{slots_json:['STRING',{}]}}},
 {app:{graph:{change(){}}},api,service:createLoraService(api)});
window.rows=()=>JSON.parse(node.widgets.find(w=>w.name==='slots_json').value);
window.widget=(id,suffix)=>node.widgets.find(w=>w.name===`donut_row:${id}:${suffix}`);
window.change=(id,suffix,value)=>{const w=widget(id,suffix);w.value=value;w.callback(value);node.layout();};
window.ready=true;
</script></body></html>'''

@unittest.skipUnless(sync_playwright, 'Playwright is a development-only dependency')
class BrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pw=sync_playwright().start()
        executable=os.environ.get('CHROMIUM_EXECUTABLE') or shutil.which('chromium')
        kwargs={'headless':True,'args':['--no-sandbox']}
        if executable: kwargs['executable_path']=executable
        cls.browser=cls.pw.chromium.launch(**kwargs)

    @classmethod
    def tearDownClass(cls):
        cls.browser.close();cls.pw.stop()

    def setUp(self):
        self.page=self.browser.new_page(viewport={'width':1000,'height':1200})
        self.errors=[];self.page.on('pageerror',lambda e:self.errors.append(str(e)))
        # This test is entirely offline: no local or external HTTP access.
        # Imports are stripped ONLY to evaluate the exact shipped modules in the
        # same isolated browser page; the node adapter above is not ComfyUI.
        import re
        markup, script = PAGE.split('<script type="module">', 1)
        script = script.split('</script>', 1)[0]
        self.page.set_content(markup + '</body></html>')
        for name in ('donut_lora_ui.js', 'donut_native_lora.js'):
            source=(ROOT/'web'/name).read_text()
            source=re.sub(r'^import .*;\n', '', source, flags=re.M).replace('export function ', 'function ')
            self.page.add_script_tag(content=source)
        self.page.add_script_tag(content=re.sub(r'^import .*;\n', '', script, flags=re.M))
        self.page.wait_for_function('window.ready && document.querySelector("img")?.complete')
        self.page.wait_for_timeout(120)

    def tearDown(self):
        self.page.close();self.assertEqual(self.errors,[])

    def test_01_visible_remove_for_active_disabled_and_final_slot(self):
        before=self.page.evaluate('rows()')
        self.page.get_by_role('button',name='Remove LoRA 2 from this stack (keeps the file)',exact=True).click()
        self.assertEqual(self.page.evaluate('rows()'),[before[0],before[2]])
        self.page.get_by_role('button',name='Remove LoRA 1 from this stack (keeps the file)',exact=True).click()
        self.assertEqual(self.page.evaluate('rows()'),[before[2]])
        self.page.get_by_role('button',name='Remove LoRA 1 from this stack (keeps the file)',exact=True).click()
        self.assertEqual(self.page.evaluate('rows()'),[])
        self.page.get_by_role('button',name='+ Add LoRA',exact=True).click()
        self.assertEqual(len(self.page.evaluate('rows()')),1)
        self.assertTrue(self.page.get_by_role('button',name='Remove LoRA 1 from this stack (keeps the file)',exact=True).is_visible())

    def test_02_move_then_remove_correct_row_and_restore_snapshot(self):
        before=self.page.evaluate('rows()')
        self.page.get_by_role('button',name='Move LoRA 3 up',exact=True).click()
        self.assertEqual(self.page.evaluate('rows().map(r=>r.id)'),['row-0','row-2','row-1'])
        self.page.get_by_role('button',name='Remove LoRA 2 from this stack (keeps the file)',exact=True).click()
        self.assertEqual(self.page.evaluate('rows()'),before[:2])
        self.page.evaluate('node.onConfigure(node.undo)')
        self.assertEqual(self.page.evaluate('rows().map(r=>r.id)'),['row-0','row-2','row-1'])

    def test_03_metadata_is_not_clipped_at_320_and_420_width(self):
        for width in (320,420):
            self.page.evaluate('(w)=>node.setSize([w,node.size[1]])',width);self.page.wait_for_timeout(100)
            bounds=self.page.evaluate('''()=>{
              const root=widget('row-0','information').element,r=root.getBoundingClientRect();
              const img=root.querySelector('img').getBoundingClientRect();
              const link=[...root.querySelectorAll('a')].find(a=>a.textContent.includes('CivitAI')).getBoundingClientRect();
              return {outer:root.scrollWidth<=root.clientWidth+1,visible:[img,link].every(x=>x.top>=r.top&&x.bottom<=r.bottom&&x.left>=r.left&&x.right<=r.right),height:r.height};}''')
            self.assertTrue(bounds['outer']);self.assertTrue(bounds['visible']);self.assertLessEqual(bounds['height'],360)
            self.assertTrue(self.page.get_by_role('button',name='Remove LoRA 1 from this stack (keeps the file)',exact=True).is_visible())

    def test_04_more_details_scrolls_without_expanding_the_whole_node(self):
        info=self.page.locator('[aria-label="LoRA 1 information"]')
        info.get_by_text('More details',exact=True).click();self.page.wait_for_timeout(150)
        size=self.page.evaluate('''()=>{const r=widget('row-0','information').element;return [r.clientHeight,r.scrollHeight,r.scrollWidth,r.clientWidth];}''')
        self.assertLessEqual(size[0],360);self.assertGreater(size[1],size[0]);self.assertLessEqual(size[2],size[3]+1)
        self.page.get_by_role('button',name='Remove LoRA 1 from this stack (keeps the file)',exact=True).click()
        self.assertEqual(len(self.page.evaluate('rows()')),2)

    def test_05_collapsing_row_retains_strengths_and_remove_visibility(self):
        before=self.page.evaluate('rows()')
        self.page.evaluate("change('row-0','enabled',false)");self.page.wait_for_timeout(70)
        self.assertFalse(self.page.locator('[aria-label="LoRA 1 information"]').is_visible())
        self.assertTrue(self.page.get_by_role('button',name='Remove LoRA 1 from this stack (keeps the file)',exact=True).is_visible())
        self.page.evaluate("change('row-0','enabled',true)");self.page.wait_for_timeout(70)
        self.assertEqual(self.page.evaluate('rows()'),before)

    def test_06_dom_controls_are_not_serialized_into_workflow_or_api_inputs(self):
        out=self.page.evaluate('''()=>{const o={};node.onSerialize(o);return o;}''')
        self.assertEqual(len(out['widgets_values']),9);self.assertEqual(len(out['widgets_values_named']),9)
        self.page.evaluate('(o)=>node.onConfigure(o)',out);self.page.wait_for_timeout(50)
        self.assertEqual(self.page.evaluate('rows()'),json.loads(out['widgets_values_named']['slots_json']))
        self.assertTrue(self.page.evaluate("node.widgets.filter(w=>w.name.startsWith('donut_row:')).every(w=>w.serialize===false && w.options.serialize===false)"))

    def test_07_sections_preserve_expansion_across_async_metadata_rerenders(self):
        section=self.page.locator('[aria-label="LoRA 1 information"] details').first
        section.locator('summary').click();self.page.wait_for_timeout(50)
        self.assertTrue(section.evaluate('(e)=>e.open'))
        self.page.evaluate("change('row-0','model_weight',0.7)");self.page.wait_for_timeout(50)
        self.assertTrue(section.evaluate('(e)=>e.open'))
        self.assertIn('blocks: 0–27',section.inner_text())
        self.assertEqual(self.page.evaluate("rows()[0].clip_weight"),0)

    def test_08_layout_preview_and_resize_observer_stability(self):
        for i in range(6):
            self.page.evaluate('(w)=>node.setSize([w,node.size[1]])',320 if i%2 else 420)
            self.page.wait_for_timeout(50)
        self.page.evaluate('node.setSize([420,node.size[1]])');self.page.wait_for_timeout(120)
        target=os.environ.get('DONUT_UI_SCREENSHOT')
        if target:
            Path(target).parent.mkdir(parents=True,exist_ok=True)
            self.page.screenshot(path=target,full_page=True)
        self.assertEqual(self.page.get_by_role('button',name='Remove LoRA',exact=False).count(),3)

if __name__=='__main__': unittest.main()

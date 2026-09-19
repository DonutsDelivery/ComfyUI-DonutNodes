"""Isolated Chromium checks of the real crop editor, not full ComfyUI."""
import argparse
from pathlib import Path
import re

from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
HTML='''<!doctype html><meta charset="utf-8"><body><script type="module">
import {openReferenceCropEditor} from '/web/donut_reference_crop_editor.js';
window.saved=null;
window.start=async({name='A',value='',width=300,height=600}={})=>{
 const canvas=document.createElement('canvas');canvas.width=width;canvas.height=height;
 const ctx=canvas.getContext('2d');ctx.fillStyle='#ddd';ctx.fillRect(0,0,width,height);
 ctx.fillStyle='#e74';ctx.fillRect(width/3,0,width/3,height);
 const image=new Image();image.src=canvas.toDataURL();await image.decode();
 window.saved=null;window.closeCrop=openReferenceCropEditor({image,imageName:name,value,onApply:v=>{window.saved=JSON.parse(v);}});
};window.ready=true;
</script>'''
RUNTIME_FIXTURE = 'const app=window.app={rootGraph:{nodes:[]},registerExtension:e=>(window.extensions||=[]).push(e),canvas:{}};\nconst api={apiURL:p=>p};\nfunction graphEntries(graph){return (graph?.nodes||[]).map(node=>({node,path:[node.id]}));}\nwindow.makeControllerFixture=async()=>{\n async function image(w,h){const c=document.createElement(\'canvas\');c.width=w;c.height=h;const x=c.getContext(\'2d\');x.fillStyle=\'#bbc\';x.fillRect(0,0,w,h);const im=new Image();im.src=c.toDataURL();await im.decode();return im;}\n const a=await image(120,80),b=await image(60,160),root=document.createElement(\'div\');root.id=\'editfixture\';root.innerHTML=\'<div class="de-top">Edit Studio</div><p><span class="de-crop-key"></span> old caption</p>\';\n const slots={};for(const [key,im]of[[\'a\',a],[\'b\',b]]){const card=document.createElement(\'section\');card.dataset.key=key;const stage=document.createElement(\'div\');stage.className=\'de-stage\';stage.style.cssText=\'width:260px;height:207px;position:relative\';stage.append(im);card.append(stage);root.append(card);slots[key]={card,stage,image:im};}\n document.body.append(root);\n const values={enabled:true,geometry_mode:INDEPENDENT,image_a:\'A\',image_b:\'B\',use_reference_b:true,crop_data_a:\'\',crop_data_b:\'\',output_canvas:\'Independent output\',resolution_mode:\'Custom\',width:192,height:128,multiple:\'32\',megapixels:1,aspect_ratio:\'16:9 Wide\'};\n const node=window.editNode={id:1,widgets:Object.entries(values).map(([name,value])=>({name,value})),graph:{beforeChange(){},afterChange(){}},setDirtyCanvas(){}};\n node._donutEditStudio={root,slots,render(){}};\n const panelRoot=document.createElement(\'div\');document.body.append(panelRoot);\n const panel=window.sizePanel={id:2,properties:{donut_app_controls:{groups:[{title:\'Image size\',donut_image_size:[1],controls:[]}]}},_donutAppControls:{root:panelRoot,render(){panelRoot.innerHTML=\'<input aria-label="Output canvas">\';}}};\n app.rootGraph.nodes=[node,panel];for(const e of window.extensions)e.afterConfigureGraph?.();\n};\n'

def run(chromium):
    count=0
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=chromium,headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1100,'height':900});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        # No network navigation: execute the local modules with imports
        # inlined in an isolated page. The full Comfy module loader is not tested.
        page.set_content('<!doctype html><meta charset="utf-8"><body></body>')
        geometry=(ROOT/'web/donut_reference_crop_geometry.js').read_text()
        editor=(ROOT/'web/donut_reference_crop_editor.js').read_text()
        page.add_script_tag(content=re.sub(r'^export ', '', geometry, flags=re.M))
        editor=re.sub(r'^import .*;\n', '', editor, flags=re.M)
        page.add_script_tag(content=re.sub(r'^export ', '', editor, flags=re.M))
        fixture=HTML.split('<script type="module">',1)[1].split('</script>',1)[0]
        fixture=re.sub(r'^import .*;\n', '', fixture, flags=re.M)
        page.add_script_tag(content=fixture);page.wait_for_function('window.ready')
        page.evaluate('start()');page.get_by_label('Crop aspect ratio').select_option('1:1')
        assert '300 × 300' in page.get_by_role('status').inner_text();count+=1
        page.get_by_role('button',name='Use crop').click()
        data=page.evaluate('saved');assert data['image']=='A' and data['bounds']==[0,.25,1,.75] and data['source_size']==[300,600];count+=1
        page.evaluate('start({name:"B"})');page.get_by_role('button',name='Cancel').click();assert page.evaluate('saved') is None;count+=1
        page.evaluate('start()');page.get_by_label('Crop aspect ratio').select_option('Free')
        rect=page.locator('dialog[open] canvas').bounding_box();page.mouse.move(rect['x']+rect['width']-2,rect['y']+rect['height']-2);page.mouse.down();page.mouse.move(rect['x']+.6*rect['width'],rect['y']+.8*rect['height'],steps=5);page.mouse.up()
        page.get_by_role('button',name='Use crop').click();data=page.evaluate('saved');assert abs(data['bounds'][2]-.6)<.015 and abs(data['bounds'][3]-.8)<.015;count+=1
        page.evaluate('value=>start({value})',data);page.get_by_role('button',name='Reset to full image').click();page.get_by_role('button',name='Use crop').click();assert page.evaluate('saved.bounds')==[0,0,1,1];count+=1
        page.evaluate('start({width:1600,height:2400})');assert '1600 × 2400' in page.get_by_role('status').inner_text();page.get_by_role('button',name='Use crop').click();assert page.evaluate('saved.source_size')==[1600,2400];count+=1
        page.evaluate('start()');rect=page.locator('dialog[open] canvas').bounding_box();page.mouse.move(rect['x']+rect['width']-2,rect['y']+rect['height']-2);page.mouse.down();page.mouse.move(rect['x']+.6*rect['width'],rect['y']+.8*rect['height'],steps=4);page.mouse.up();page.get_by_role('button',name='Use crop').click()
        b=page.evaluate('saved.bounds');assert abs((b[2]-b[0])-(b[3]-b[1]))<1e-6;count+=1
        page.evaluate('start()');page.keyboard.press('Escape');page.wait_for_function('document.querySelectorAll("dialog").length===0');assert page.locator('dialog').count()==0 and page.evaluate('saved') is None;count+=1
        assert errors==[],errors;count+=1
        # Exercise the actual card adapter against a small fake Comfy graph.
        page.add_script_tag(content=RUNTIME_FIXTURE)
        adapter=(ROOT/'web/donut_reference_crops.js').read_text()
        page.add_script_tag(content=re.sub(r'^import .*;\n','',adapter,flags=re.M))
        page.evaluate('makeControllerFixture()');page.wait_for_function('editNode._donutReferenceCrops')
        card=page.locator('#editfixture [data-key="b"]')
        card.get_by_role('button',name='Crop image…').click();page.get_by_label('Crop aspect ratio').select_option('1:1');page.get_by_role('button',name='Use crop').click()
        assert page.evaluate('editNode.donutCropBox("b")')==[0,50,60,110];count+=1
        assert page.evaluate('editNode.donutCropBox("a")')==[0,0,120,80];count+=1
        saved=page.evaluate('editNode.widgets.find(w=>w.name==="crop_data_b").value')
        page.evaluate('editNode.widgets.find(w=>w.name==="width").value=640;editNode.widgets.find(w=>w.name==="height").value=1024;editNode._donutReferenceCrops.refresh()')
        assert page.evaluate('editNode.donutCropOutputSize()')==[640,1024] and page.evaluate('editNode.widgets.find(w=>w.name==="crop_data_b").value')==saved;count+=1
        page.locator('#editfixture').get_by_label('Reference crop geometry').select_option('Legacy output-linked')
        assert page.evaluate('editNode.donutCropBox("b")') is None;count+=1
        page.locator('#editfixture').get_by_label('Reference crop geometry').select_option('Independent crops')
        assert page.evaluate('editNode.widgets.find(w=>w.name==="crop_data_b").value')==saved;count+=1
        page.evaluate('editNode._donutReferenceCrops.refresh()')
        assert not page.locator('#editfixture').get_by_label('Reference crop output canvas').is_visible();count+=1
        page.evaluate('app.rootGraph.nodes=[editNode];editNode._donutReferenceCrops.refresh()')
        assert page.locator('#editfixture').get_by_label('Reference crop output canvas').is_visible();count+=1
        assert errors==[],errors;count+=1
        browser.close()
    print(f'{count} isolated Chromium checks passed.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--chromium',default='/usr/bin/chromium');args=parser.parse_args();run(args.chromium)

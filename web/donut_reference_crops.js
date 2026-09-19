import {app} from '../../scripts/app.js';
import {api} from '../../scripts/api.js';
import {graphEntries} from './donut_panel_categories_model.js';
import {LEGACY,INDEPENDENT,parseCrop,cropBox,outputDimensions,fitGeometry,subjectBounds,sizingVisibility} from './donut_reference_crop_geometry.js';
import {openReferenceCropEditor} from './donut_reference_crop_editor.js';

const el=(tag,text)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n;};
function install(node) {
    if(node._donutReferenceCrops || !node.widgets?.some(w=>w.name==='geometry_mode'))return;
    const edit=!!node._donutEditStudio;
    const owner=node._donutEditStudio || node._donutReferenceStudio;
    if(!owner?.root || !owner.slots)return;
    const root=owner.root,widgets=new Map(node.widgets.map(w=>[w.name,w]));
    const get=name=>widgets.get(name)?.value;
    const values=()=>Object.fromEntries([...widgets].map(([k,w])=>[k,w.value]));
    const imageFor=key=>owner.slots[key]?.image;
    const imageSize=key=>{const image=imageFor(key);return image?.naturalWidth?[image.naturalWidth,image.naturalHeight]:null;};
    let disposed=false,closeEditor=null,timer=null;
    const cards=[];
    function commit(changes){
        node.graph?.beforeChange?.();
        for(const [name,value] of Object.entries(changes)){
            const w=widgets.get(name);if(!w)continue;w.value=value;w.callback?.(value,app.canvas,node);
        }
        node.graph?.afterChange?.();node.setDirtyCanvas?.(true,true);refresh();owner.render?.();owner.refresh?.();
    }
    const setup=el('section');setup.className='de-section';
    const modeLabel=el('label','Reference crop geometry '),mode=el('select');
    mode.setAttribute('aria-label','Reference crop geometry');
    for(const choice of [LEGACY,INDEPENDENT]){const o=el('option',choice);o.value=choice;mode.append(o);}
    mode.onchange=()=>commit({geometry_mode:mode.value});modeLabel.append(mode);setup.append(modeLabel);
    const help=el('p','Independent crops keep each selection when output size changes. Legacy keeps the old output-linked behavior.');help.className='de-help';setup.append(help);
    root.querySelector('.de-top')?.after(setup);
    const message=el('p');message.className='de-help';message.setAttribute('role','status');setup.append(message);
    let outputSelect=null,outputFallback=null;
    if(edit){
        outputFallback=el('label','Output canvas ');outputSelect=el('select');outputSelect.setAttribute('aria-label','Reference crop output canvas');
        for(const choice of ['Follow A crop','Independent output']){const o=el('option',choice);o.value=choice;outputSelect.append(o);}
        outputSelect.onchange=()=>commit({output_canvas:outputSelect.value});outputFallback.append(outputSelect);setup.append(outputFallback);
    }
    function selectedBox(key){const size=imageSize(key);return size?cropBox(get(`crop_data_${key}`),get(`image_${key}`),size):null;}
    function effectiveSize(){
        if(!get('enabled') || get('geometry_mode')!==INDEPENDENT)return null;
        return outputDimensions(values(),imageSize('a'),selectedBox('a'),imageSize('b'),imageSize('b')?selectedBox('b'):null);
    }
    function openCrop(key){
        if(get('geometry_mode')!==INDEPENDENT)return false;
        const image=imageFor(key),name=get(`image_${key}`);if(!image?.naturalWidth||!name)return true;
        closeEditor?.();
        closeEditor=openReferenceCropEditor({image,imageName:name,value:get(`crop_data_${key}`),onApply:value=>{
            if(disposed||get(`image_${key}`)!==name)throw new Error('Reference changed. Reopen the crop editor.');
            commit({[`crop_data_${key}`]:value});
        }});return true;
    }
    if(edit){
        node.donutCropBox=key=>{if(get('geometry_mode')!==INDEPENDENT)return null;try{return selectedBox(key);}catch(error){message.textContent=error.message;return null;}};
        node.donutCropOutputSize=()=>{try{return effectiveSize();}catch(error){message.textContent=error.message;return null;}};
        node.donutCropInpaintSize=()=>{
            if(get('geometry_mode')!==INDEPENDENT)return null;
            try{const b=selectedBox('a'),size=effectiveSize();return b&&size?fitGeometry([b[2]-b[0],b[3]-b[1]],size).content:null;}
            catch(error){message.textContent=error.message;return null;}
        };
        node.donutOpenReferenceCrop=openCrop;
        node.donutApplyCropSizing=()=>{
            const hidden=sizingVisibility(values());if(!hidden)return;
            const input=label=>root.querySelector(`[aria-label="${label}"]`);
            const width=input('Custom output width'),aspect=input('Output aspect ratio'),mp=input('Output megapixels');
            const customRow=width?.closest('.de-row'),presetRow=mp?.closest('.de-row');
            if(customRow)customRow.hidden=hidden.width;
            if(presetRow)presetRow.hidden=hidden.aspect_ratio&&hidden.megapixels;
            if(aspect?.closest('.de-field'))aspect.closest('.de-field').hidden=hidden.aspect_ratio;
            if(mp?.closest('.de-field'))mp.closest('.de-field').hidden=hidden.megapixels;
        };
    }
    for(const key of ['a','b']){
        const slot=owner.slots[key];if(!slot?.card)continue;
        const section=el('section');section.className='de-section';section.style.padding='10px';
        section.append(el('strong',`${key.toUpperCase()} · Source crop`));
        const buttons=el('div');buttons.className='de-ref-actions';buttons.style.flexWrap='wrap';
        const crop=el('button','Crop image…'),reset=el('button','Reset crop'),readout=el('output');readout.className='de-help';
        crop.type=reset.type='button';crop.onclick=()=>openCrop(key);reset.onclick=()=>commit({[`crop_data_${key}`]:''});buttons.append(crop,reset);section.append(buttons,readout);
        let padding=null,subject=null,subjectBusy=false;
        if(edit&&key==='b'){
            subject=el('button','Crop to subject');subject.type='button';padding=el('input');padding.type='number';padding.value='4';padding.min='0';padding.max='50';padding.step='1';padding.style.width='60px';padding.setAttribute('aria-label','Subject crop padding percent');
            const label=el('label','Padding % ');label.append(padding);buttons.append(subject,label);
            subject.onclick=async()=>{
                const name=get('image_b'),size=imageSize('b'),maskData=get('mask_b_data'),cropData=get('crop_data_b');if(!size||subjectBusy)return;
                subjectBusy=true;subject.disabled=true;
                try{
                    const record=JSON.parse(get('mask_b_data')||'null');
                    if(record?.image!==name || !/^donutmask:[a-f0-9]{64}$/.test(record?.mask||''))throw new Error('Auto select or save a subject mask for this B first.');
                    const mask=new Image();const loaded=new Promise((resolve,reject)=>{mask.onload=resolve;mask.onerror=()=>reject(new Error('Saved subject mask is unavailable.'));});
                    mask.src=api.apiURL(`/donut/edit-studio/subject-mask/${record.mask.slice(10)}`);await loaded;
                    if(disposed||get('image_b')!==name||get('mask_b_data')!==maskData||get('crop_data_b')!==cropData)throw new Error('Reference, mask or crop changed. The old crop result was not applied.');
                    if(mask.naturalWidth!==size[0]||mask.naturalHeight!==size[1])throw new Error('Mask dimensions do not match original B.');
                    const c=document.createElement('canvas');c.width=size[0];c.height=size[1];const ctx=c.getContext('2d');ctx.drawImage(mask,0,0);
                    const bounds=subjectBounds(ctx.getImageData(0,0,...size).data,...size,Number(padding.value));
                    commit({crop_data_b:JSON.stringify({version:1,image:name,source_size:size,bounds,aspect:'Free'})});
                }catch(error){message.textContent=error.message;}finally{subjectBusy=false;if(!disposed)refresh();}
            };
        }
        let overlay=null;
        if(!edit){
            overlay=el('canvas');overlay.style.cssText='position:absolute;inset:0;width:100%;height:100%;pointer-events:none;';slot.stage.append(overlay);
        }
        slot.card.append(section);cards.push({key,slot,section,crop,reset,readout,subject,overlay,isBusy:()=>subjectBusy});
    }
    function refresh(){
        if(disposed)return;
        mode.value=get('geometry_mode')||LEGACY;
        const independent=mode.value===INDEPENDENT;
        help.textContent=independent&&edit?'Each crop is independent. Follow A uses its crop aspect and the megapixel budget (or native crop-only mode); independent output uses global Preset/Custom sizing.':'Independent crops keep each selection when output size changes. Legacy keeps the old output-linked behavior.';
        if(edit){const text=root.querySelector('.de-crop-key')?.parentElement?.lastChild;if(text?.nodeType===3)text.textContent=independent?'Red frame = source crop · click to edit':'Red frame = kept area · drag to reposition';}
        if(outputSelect)outputSelect.value=get('output_canvas')||'Follow A crop';
        if(outputFallback){
            const entries=graphEntries(app.rootGraph||app.graph),own=entries.find(e=>e.node===node);
            const hasPanel=own&&entries.some(({node:panel})=>panel.properties?.donut_app_controls?.groups?.some(group=>
                JSON.stringify(group.donut_image_size?.map(String))===JSON.stringify(own.path.map(String)))
                && !!panel._donutAppControls?.root?.querySelector('[aria-label="Output canvas"]'));
            outputFallback.hidden=!independent||!!hasPanel;
        }
        for(const item of cards){
            const {key,slot,section,crop,reset,readout,subject,overlay}=item;
            const size=imageSize(key);section.hidden=!independent;crop.disabled=reset.disabled=!size;
            if(overlay)overlay.hidden=!independent||!size;
            if(subject)subject.disabled=item.isBusy()||!size||!get('mask_b_data');
            if(!size) {readout.textContent='Load an image to crop.';continue;}
            try{
                const box=selectedBox(key);
                readout.textContent=`${box[2]-box[0]} × ${box[3]-box[1]} source pixels · independent of output size`;
                if(overlay&&independent){
                    const w=slot.stage.clientWidth||260,h=slot.stage.clientHeight||207;overlay.width=w;overlay.height=h;
                    const ctx=overlay.getContext('2d'),scale=Math.min(w/size[0],h/size[1]),ox=(w-size[0]*scale)/2,oy=(h-size[1]*scale)/2;
                    ctx.clearRect(0,0,w,h);const [x1,y1,x2,y2]=box;
                    ctx.fillStyle='#0008';ctx.fillRect(ox,oy,size[0]*scale,y1*scale);ctx.fillRect(ox,oy+y2*scale,size[0]*scale,(size[1]-y2)*scale);
                    ctx.fillRect(ox,oy+y1*scale,x1*scale,(y2-y1)*scale);ctx.fillRect(ox+x2*scale,oy+y1*scale,(size[0]-x2)*scale,(y2-y1)*scale);
                    ctx.strokeStyle='#ff7580';ctx.lineWidth=2;ctx.strokeRect(ox+x1*scale,oy+y1*scale,(x2-x1)*scale,(y2-y1)*scale);
                }
            }catch(error){readout.textContent=error.message;}
        }
        if(!edit){const help=root.querySelector(':scope > p.de-help');if(help)help.textContent=independent?'Uses each selected source crop. Describe what to borrow in the main prompt.':'Uses each full image. Describe what to borrow in the main prompt.';}
    }
    const observer=new IntersectionObserver(entries=>{clearInterval(timer);if(entries.some(e=>e.isIntersecting)){refresh();timer=setInterval(()=>{refresh();if(edit)owner.render?.();},500);}});observer.observe(root);
    const removed=node.onRemoved,added=node.onAdded,configured=node.onConfigure;
    node.onRemoved=function(){disposed=true;closeEditor?.();clearInterval(timer);observer.disconnect();return removed?.apply(this,arguments);};
    node.onAdded=function(){disposed=false;observer.observe(root);refresh();return added?.apply(this,arguments);};
    node.onConfigure=function(){const result=configured?.apply(this,arguments);refresh();return result;};
    node._donutReferenceCrops={refresh};refresh();
}
let pending=false;
function refreshAll(){
    if(pending)return;pending=true;queueMicrotask(()=>{
        pending=false;const entries=graphEntries(app.rootGraph||app.graph);
        for(const {node} of entries)install(node);
        for(const {node:panel} of entries){
            const groups=panel.properties?.donut_app_controls?.groups;if(!groups)continue;
            let changed=false;
            for(const group of groups){
                if(!group.donut_image_size)continue;
                const same=entry=>JSON.stringify(entry.path.map(String))===JSON.stringify(group.donut_image_size.map(String));
                const target=entries.find(same)?.node;
                if(!target?.widgets?.some(w=>w.name==='output_canvas'))continue;
                if(!group.controls.some(c=>c.widget==='output_canvas')){
                    group.controls.unshift({path:[...group.donut_image_size],widget:'output_canvas',title:'Output canvas'});changed=true;
                }
            }
            if(changed)panel._donutAppControls?.render();
        }
    });
}
app.registerExtension({name:'Donut.IndependentReferenceCrops',nodeCreated:refreshAll,afterConfigureGraph:refreshAll,
    beforeRegisterNodeDef(type,definition){
        if(!['DonutEditStudio','DonutReferenceStudio'].includes(definition.name))return;
        const created=type.prototype.onNodeCreated;
        type.prototype.onNodeCreated=function(){const result=created?.apply(this,arguments);queueMicrotask(()=>install(this));return result;};
    }
});

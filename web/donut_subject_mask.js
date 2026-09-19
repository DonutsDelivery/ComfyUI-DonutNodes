import {app} from '../../scripts/app.js';
import {api} from '../../scripts/api.js';
import {installEditStudio} from './donut_edit_studio.js';
import {openSubjectMaskEditor} from './donut_subject_mask_editor.js';
import {readSubjectMask, subjectMaskPrompt, acceptSubjectMaskResult} from './donut_subject_mask_state.js';

function install(node, definition) {
    if (node._donutSubjectMask || !node.widgets?.some(w => w.name === 'mask_b_mode')) return;
    if (!node._donutEditStudio) installEditStudio(node, definition);
    const studio = node._donutEditStudio;
    if (!studio?.slots?.b?.card) return;
    const widgets = new Map(node.widgets.map(widget => [widget.name, widget]));
    const get = name => widgets.get(name)?.value;
    const controls = [], cleanup = [];
    let job = null, disposed = false, closeEditor = null;
    function commit(changes) {
        node.graph?.beforeChange?.();
        for (const [name,value] of Object.entries(changes)) {
            const widget = widgets.get(name); if (!widget) continue;
            widget.value = value; widget.callback?.(value,app.canvas,node);
        }
        node.graph?.afterChange?.(); node.setDirtyCanvas?.(true,true); refresh();
    }
    function el(tag, text) {const value=document.createElement(tag); if(text !== undefined) value.textContent=text; return value;}
    const section=el('div'); section.className='de-section'; section.style.padding='10px';
    section.append(el('strong','Smart subject mask · B'));
    const message=el('p'); message.className='de-help'; message.setAttribute('role','status'); message.setAttribute('aria-live','polite');
    const field=(name,title,choices=null) => {
        const label=el('label'); label.className='de-field'; label.append(el('span',title));
        const input=el(choices ? 'select' : 'input'); input.setAttribute('aria-label',title);
        if(choices) for(const value of choices) {const option=el('option',value); option.value=value; input.append(option);}
        else {input.type='number'; input.min=String(widgets.get(name)?.options?.min ?? 0); input.max=String(widgets.get(name)?.options?.max ?? 64); input.step='1';}
        input.addEventListener('change',()=>{const value=choices ? input.value : Number(input.value); if(choices || Number.isFinite(value)) commit({[name]:value});});
        label.append(input); section.append(label); controls.push({name,input,choices}); return input;
    };
    field('mask_b_mode','Reference B mask mode',['Off','Auto subject','Saved mask','External mask']);
    const values=widgets.get('mask_b_model')?.options?.values;
    field('mask_b_model','Background removal model',typeof values === 'function' ? values() : values || ['birefnet.safetensors']);
    field('mask_b_grow','Grow / shrink · B source pixels');
    field('mask_b_feather','Feather · B source pixels');
    field('mask_b_background','Outside the subject',['Neutral gray','White','Black']);
    const actions=el('div'); actions.className='de-ref-actions'; section.append(actions);
    function button(text, callback) {const b=el('button',text); b.type='button'; b.addEventListener('click',callback); actions.append(b); return b;}
    const auto=button('Auto select subject',async()=>{
        const reference=get('image_b'); if(!reference || job) return;
        const nodeId=`donut_subject_${globalThis.crypto?.randomUUID?.() || `${Date.now()}_${Math.random()}`}`;
        const pending={nodeId,reference,promptId:null}; job=pending; refresh(); message.textContent='Subject selection queued. Only the mask is processed, not the full workflow.';
        try {
            const response=await api.queuePrompt(0,subjectMaskPrompt(reference,get('mask_b_model'),nodeId));
            if(job === pending) job.promptId=response.prompt_id;
        } catch(error) {if(!disposed && job === pending) {job=null; refresh(); message.textContent=error.message;}}
    });
    const refine=button('Refine / paint…',async()=>{
        const reference=get('image_b'), image=studio.slots.b.image;
        if(!image || !reference) return;
        const record=readSubjectMask(get('mask_b_data'),reference);
        try {
            let maskImage=null;
            if(record) {
                maskImage=new Image(); const loaded=new Promise((resolve,reject)=>{maskImage.onload=resolve; maskImage.onerror=()=>reject(new Error('Saved mask is unavailable. Auto select again.'));});
                maskImage.src=api.apiURL(`/donut/edit-studio/subject-mask/${record.mask.slice(10)}`); await loaded;
            }
            if(disposed || reference !== get('image_b')) return;
            closeEditor?.();
            closeEditor=openSubjectMaskEditor({image,maskImage,onApply:async blob=>{
                if(disposed || reference !== get('image_b')) throw new Error('Reference B changed. Close this editor and select the new subject.');
                const form=new FormData(); form.append('reference',reference); form.append('mask',blob,'subject-mask.png');
                const response=await api.fetchApi('/donut/edit-studio/subject-mask',{method:'POST',body:form});
                if(!response.ok) throw new Error(await response.text());
                const saved=readSubjectMask(await response.json(),reference);
                if(!saved) throw new Error('The server returned an invalid mask.');
                if(disposed || reference !== get('image_b')) throw new Error('Reference B changed while saving; this mask was not applied.');
                commit({mask_b_mode:'Saved mask',mask_b_data:JSON.stringify(saved)});
                message.textContent='Subject mask saved. Save the workflow to retain this selection.';
            }});
        } catch(error) {if(!disposed) message.textContent=error.message;}
    });
    const preview=el('img'); preview.alt='Reference B foreground mask: white keeps, black removes'; preview.style.cssText='display:block;width:100%;max-height:160px;object-fit:contain;background:#111;'; preview.hidden=true;
    const help=el('p','Mask preview is before grow/feather. Auto selects foreground, not a text-named object. External mask accepts a white-foreground mask from Grounding/SAM or another node. A’s edit selection is unchanged.'); help.className='de-help';
    section.append(preview,help,message); studio.slots.b.card.append(section);
    function refresh() {
        if(disposed) return;
        const reference=get('image_b'), mode=get('mask_b_mode');
        for(const {name,input,choices} of controls) {
            const value=get(name);
            if(choices && value && ![...input.options].some(option=>option.value === String(value))) {const option=el('option',value); option.value=value; input.append(option);}
            if(document.activeElement !== input) input.value=value ?? '';
        }
        auto.disabled=!!job || !reference; refine.disabled=!!job || !studio.slots.b.image || !String(reference).startsWith('donutref:');
        const record=readSubjectMask(get('mask_b_data'),reference); preview.hidden=!record;
        if(record) {const url=api.apiURL(`/donut/edit-studio/subject-mask/${record.mask.slice(10)}`); if(preview.getAttribute('src') !== url) preview.src=url;}
        if(mode === 'Saved mask' && !record) message.textContent='No matching saved mask for this B. Auto select, paint a mask, or turn masking off.';
    }
    const executed=event=>{
        if(disposed || !job || event.detail.node !== job.nodeId) return;
        const record=acceptSubjectMaskResult(job,event.detail.node,get('image_b'),event.detail.output);
        job=null;
        if(record) {commit({mask_b_mode:'Saved mask',mask_b_data:JSON.stringify(record)}); message.textContent='Subject isolated. Refine the mask as needed, then save the workflow.';}
        else {refresh(); message.textContent='Reference B changed while selection ran; the old result was not applied.';}
    };
    const failed=event=>{
        if(!job || disposed) return;
        const detail=event.detail;
        if(detail.node_id !== job.nodeId && (!job.promptId || detail.prompt_id !== job.promptId)) return;
        job=null; refresh(); message.textContent=detail.exception_message || 'Subject selection was interrupted. You can retry or paint a mask.';
    };
    for(const [type,handler] of [['executed',executed],['execution_error',failed],['execution_interrupted',failed]]) {
        api.addEventListener(type,handler); cleanup.push(()=>api.removeEventListener(type,handler));
    }
    const onExecuted=node.onExecuted;
    node.onExecuted=function(output) {
        const result=onExecuted?.apply(this,arguments);
        const record=readSubjectMask(output?.donut_subject_mask?.[0],get('image_b'));
        // Keep Auto/External mode live; only the explicit preview button selects
        // Saved mask. A normal generation supplies a preview for later refining.
        if(record && !disposed && !job && ['Auto subject','External mask'].includes(get('mask_b_mode'))) commit({mask_b_data:JSON.stringify(record)});
        return result;
    };
    let timer;
    const observer=new IntersectionObserver(entries=>{clearInterval(timer); if(entries.some(entry=>entry.isIntersecting)) {refresh(); timer=setInterval(refresh,500);}}); observer.observe(section);
    const added=node.onAdded;
    node.onAdded=function(){
        const result=added?.apply(this,arguments);
        if(disposed) {
            disposed=false;
            for(const [type,handler] of [['executed',executed],['execution_error',failed],['execution_interrupted',failed]]) api.addEventListener(type,handler);
            observer.observe(section); refresh();
        }
        return result;
    };
    const configured=node.onConfigure;
    node.onConfigure=function(){const value=configured?.apply(this,arguments); refresh(); return value;};
    const removed=node.onRemoved;
    node.onRemoved=function(){disposed=true; job=null; closeEditor?.(); clearInterval(timer); observer.disconnect(); cleanup.forEach(fn=>fn()); return removed?.apply(this,arguments);};
    node._donutSubjectMask={refresh}; refresh();
}
app.registerExtension({
    name:'Donut.EditStudioSubjectMask',
    beforeRegisterNodeDef(nodeType,definition) {
        if(definition.name !== 'DonutEditStudio') return;
        const created=nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated=function(){const result=created?.apply(this,arguments); queueMicrotask(()=>install(this,definition)); return result;};
    },
});

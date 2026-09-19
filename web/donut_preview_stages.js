// Add side-branch previews only; the generation/finishing chain is untouched.
export function addStagePreviews(root) {
    if (!['V4 Beta','V5'].includes(root?.extra?.donut_workflow?.release)) return false;
    const panels=(root.nodes||[]).filter(n=>n.type==='DonutLatestPreview');
    const definitions=root.definitions?.subgraphs || [];
    if(panels.length!==1) return false;
    const engines=(root.nodes||[]).map(node=>({node,graph:definitions.find(g=>g.id===node.type)}))
        .filter(({graph})=>graph?.nodes?.some(n=>n.type==='DonutFaceDetailer'));
    if(engines.length!==1) return false;
    const {node:engine,graph}=engines[0];
    const faces=graph.nodes.filter(n=>n.type==='DonutFaceDetailer');
    if(faces.length!==1 || !graph.links.every(l=>!Array.isArray(l))) return false;
    const face=faces[0], edge=graph.links.find(l=>l.id===face.inputs?.find(i=>i.name==='image')?.link);
    const second=graph.nodes.find(n=>n.id===edge?.origin_id);
    const maskInput=graph.inputs?.findIndex(i=>i.name==='edit_inpaint');
    if(second?.type!=='DonutInpaintComposite' || !(maskInput>=0)) return false;
    const panel=panels[0], old=panel.properties.sources || {};
    const base=Object.keys(old).find(id=>old[id]==='Base generation');
    const first=Object.keys(old).find(id=>old[id]==='First upscale');
    const final=Object.keys(old).find(id=>['Final image','SeedVR2 / final image'].includes(old[id]));
    if(!base || !first || !final || !graph.state || !graph.inputNode) return false;
    let changed=false;
    let nodeId=Math.max(graph.state?.lastNodeId||0,...graph.nodes.map(n=>Number(n.id)||0));
    let linkId=Math.max(graph.state?.lastLinkId||0,...graph.links.map(l=>Number(l.id)||0));
    function connect(source,slot,target,input,type) {
        const id=++linkId;
        graph.links.push({id,origin_id:source.id,origin_slot:slot,target_id:target.id,target_slot:input,type});
        source.outputs[slot].links ||= [];source.outputs[slot].links.push(id);target.inputs[input].link=id;
    }
    function preview(stage,title,source) {
        let node=graph.nodes.find(n=>n.properties?.donut_preview_stage===stage);
        if(node) return node;
        node={id:++nodeId,type:'PreviewImage',pos:[1400,stage==='second'?800:1100],size:[220,80],flags:{collapsed:true},order:graph.nodes.length,mode:0,
            inputs:[{name:'images',type:'IMAGE',link:null}],outputs:[],title:`Result · ${title}`,
            properties:{'Node name for S&R':'PreviewImage',donut_preview_stage:stage}};
        graph.nodes.push(node);connect(source,0,node,0,'IMAGE');changed=true;return node;
    }
    const secondPreview=preview('second','second upscale',second);
    let preserved=graph.nodes.find(n=>n.properties?.donut_preview_stage==='face_composite');
    if(!preserved) {
        preserved={id:++nodeId,type:'DonutInpaintComposite',pos:[1100,1100],size:[220,80],flags:{collapsed:true},order:graph.nodes.length,mode:0,
            inputs:[{name:'image',type:'IMAGE',link:null},{name:'inpaint',type:'DONUT_INPAINT',link:null,shape:7}],
            outputs:[{name:'image',type:'IMAGE',links:[]}],title:'Face preview · keep surroundings',
            properties:{'Node name for S&R':'DonutInpaintComposite',donut_preview_stage:'face_composite'}};
        graph.nodes.push(preserved);connect(face,0,preserved,0,'IMAGE');
        const id=++linkId;graph.links.push({id,origin_id:graph.inputNode.id,origin_slot:maskInput,target_id:preserved.id,target_slot:1,type:'DONUT_INPAINT'});
        preserved.inputs[1].link=id;graph.inputs[maskInput].linkIds ||= [];graph.inputs[maskInput].linkIds.push(id);changed=true;
    }
    const facePreview=preview('face','Face Detailer',preserved);
    graph.state.lastNodeId=nodeId;graph.state.lastLinkId=linkId;
    if(base && first && final) {
        const sources={[base]:'Base generation',[first]:'First upscale',[`${engine.id}:${secondPreview.id}`]:'Second upscale',
            [`${engine.id}:${facePreview.id}`]:'Face Detailer',[final]:'SeedVR2 / final image'};
        const order=[base,first,`${engine.id}:${secondPreview.id}`,`${engine.id}:${facePreview.id}`,final];
        if(JSON.stringify(old)!==JSON.stringify(sources) || JSON.stringify(panel.properties.source_order)!==JSON.stringify(order))changed=true;
        panel.properties.sources=sources;panel.properties.source_order=order;
    }
    return changed;
}

export function previewSourceId(detail,sources) {
    // display_node can be the enclosing subgraph, while node is the executed leaf.
    return [detail?.node,detail?.display_node].map(String).find(id=>Object.hasOwn(sources||{},id));
}

// ComfyUI can renumber nested nodes while loading a subgraph. Resolve their
// semantic tags after loading, when runtime execution paths are available.
export function rebindStageSources(properties, entries) {
    if (!properties?.sources) return;
    for (const [tag, label] of [['second', 'Second upscale'], ['face', 'Face Detailer']]) {
        const matches = entries.filter(({node}) => node.properties?.donut_preview_stage === tag);
        if (matches.length !== 1) continue;
        const old = Object.keys(properties.sources).find(id => properties.sources[id] === label);
        const current = matches[0].path.map(String).join(':');
        if (!old || old === current) continue;
        properties.sources = Object.fromEntries(Object.entries(properties.sources).map(([id,name]) => [id === old ? current : id,name]));
        properties.source_order = (properties.source_order || Object.keys(properties.sources)).map(id => id === old ? current : id);
        if (properties.preview_selection === old) properties.preview_selection = current;
        for (const key of ['stage_images','stage_prompts']) {
            if (properties[key] && Object.hasOwn(properties[key],old)) {
                properties[key][current] = properties[key][old]; delete properties[key][old];
            }
        }
    }
}

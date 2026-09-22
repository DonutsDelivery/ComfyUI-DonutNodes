// Pure serialized-workflow migration shared by the frontend and authoring CLI.
// Only the standard tagged V5 final-output path is recognized. No numeric IDs
// are assumed. Existing settings, prompts, seeds and internal graphs are kept.
export function addToneLabToV5(root) {
    const unchanged = reason => ({changed:false, reason});
    if (root?.extra?.donut_workflow?.release !== 'V5') return unchanged('not_v5');
    if (root.extra.donut_tone_lab_v1) return unchanged('already_migrated');
    const nodes = root.nodes, links = root.links;
    if (!Array.isArray(nodes) || !Array.isArray(links) || !links.every(l => Array.isArray(l) && l.length >= 6)) return unchanged('unsupported_links');
    if (nodes.some(n => n.type === 'DonutToneLab')) return unchanged('already_present');
    const unique = values => values.length === 1 ? values[0] : null;
    const latest = unique(nodes.filter(n => n.type === 'DonutLatestPreview'));
    const sources = latest?.properties?.sources || {};
    const finalKey = unique(Object.keys(sources).filter(k => ['Final image','SeedVR2 / final image'].includes(sources[k])));
    const preview = unique(nodes.filter(n => n.type === 'PreviewImage' && String(n.id) === finalKey));
    const panel = unique(nodes.filter(n => n.properties?.donut_panel_role === 'save' && Array.isArray(n.properties?.donut_app_controls?.groups)));
    const studio = unique(nodes.filter(n => n.type === 'DonutEditStudio'));
    const editSlot = studio?.outputs?.findIndex(o => o.name === 'edit_mode' && o.type === 'BOOLEAN');
    if (!preview || !panel || !(editSlot >= 0)) return unchanged('ambiguous_final_or_edit_controls');
    const inputSlot = preview.inputs?.findIndex(i => i.name === 'images' && i.type === 'IMAGE');
    const edge = inputSlot >= 0 && unique(links.filter(l => l[0] === preview.inputs[inputSlot].link));
    const engine = edge && unique(nodes.filter(n => n.id === edge[1]));
    const definition = engine && root.definitions?.subgraphs?.find(g => g.id === engine.type);
    // Final output must still come directly from the original generation engine.
    // Do not insert around an existing user-customized postprocessing chain.
    if (!edge || edge[3] !== preview.id || edge[4] !== inputSlot || edge[5] !== 'IMAGE' ||
        !definition?.nodes?.some(n => n.type === 'DonutFaceDetailer') ||
        engine.outputs?.[edge[2]]?.type !== 'IMAGE') return unchanged('custom_final_chain');
    const outbound = links.filter(l => l[1] === engine.id && l[2] === edge[2]);
    const allowed = new Set(['PreviewImage','DonutImageSave','SaveImage','ImageScale','ImageScaleBy']);
    const sourceLinks = engine.outputs[edge[2]].links || [];
    if (outbound.length !== sourceLinks.length || !outbound.every(l => {
        const target = nodes.find(n => n.id === l[3]);
        return l[5] === 'IMAGE' && sourceLinks.includes(l[0]) && allowed.has(target?.type) && target.inputs?.[l[4]]?.link === l[0];
    })) return unchanged('custom_or_inconsistent_outputs');
    // Check every precondition before touching the graph; failure is a no-op.
    const allGraphs = [root, ...(root.definitions?.subgraphs || [])];
    let nextNode = Math.max(root.last_node_id || 0, ...allGraphs.flatMap(g => (g.nodes || []).map(n => Number(n.id) || 0)));
    let nextLink = Math.max(root.last_link_id || 0, ...links.map(l => Number(l[0]) || 0));
    const id = ++nextNode, imageLink = ++nextLink, editLink = ++nextLink;
    const node = {
        id, type:'DonutToneLab', title:'Tone Lab · final learned tone',
        pos:[...(preview.pos || [0,0])], size:[320,180], flags:{collapsed:true},
        order:engine.order == null ? nodes.length : engine.order + 1, mode:0,
        inputs:[{name:'image',type:'IMAGE',link:imageLink}, {name:'edit_mode',type:'BOOLEAN',shape:7,link:editLink}],
        outputs:[{name:'image',type:'IMAGE',links:outbound.map(l => l[0])}, {name:'report',type:'STRING',links:[]}],
        properties:{'Node name for S&R':'DonutToneLab',donut_tone_lab_stage:true},
        widgets_values:[false,'None',1,false],
        widgets_values_named:{enabled:false,model_name:'None',strength:1,apply_to_edits:false},
    };
    const sourceSlot = edge[2];
    for (const link of outbound) { link[1] = id; link[2] = 0; }
    engine.outputs[sourceSlot].links = [imageLink];
    studio.outputs[editSlot].links ||= [];
    studio.outputs[editSlot].links.push(editLink);
    links.push([imageLink,engine.id,sourceSlot,id,0,'IMAGE'], [editLink,studio.id,editSlot,id,1,'BOOLEAN']);
    nodes.push(node);
    panel.properties.donut_app_controls.groups.unshift({
        title:'Tone Lab · learned auto tone', advanced:false, donut_category_fixed:true, donut_category_rank:5,
        description:'Final learned gamma + brightness after all processing. Select a Tone Lab v4 model-only JSON from models/donut_tone. Off by default. Editing is protected unless explicitly enabled.',
        controls:[['enabled','Enable learned auto tone'],['model_name','Tone Lab model'],['strength','Tone strength'],['apply_to_edits','Also grade edited images (changes preserved surroundings)']]
            .map(([widget,title]) => ({path:[id],widget,title})),
    });
    root.extra.donut_tone_lab_v1 = {node_id:id,version:1};
    root.last_node_id = nextNode; root.last_link_id = nextLink;
    // The existing final-preview ID and source-order entry are deliberately kept.
    // Both main save and secondary output now see the same corrected final image.
    return {changed:true,node_id:id};
}

// Repair the narrowly defined PR #66 serialized post-pass shape before Comfy
// imports it. This is graph compatibility, not a change to either upscaler.
export const POST_WIDGET_ORDER = [
    'seedvr2_upscale_factor', 'resampling_method', 'enabled', 'seed', 'control_after_generate',
    'seedvr2_model_name', 'seedvr2_vae_name', 'seedvr2_steps', 'seedvr2_denoise',
    'seedvr2_color_correction', 'seedvr2_vae_tile_size',
];
const fields = ['id','origin_id','origin_slot','target_id','target_slot','type'];
const edge = raw => Array.isArray(raw) ? Object.fromEntries(fields.map((key,i) => [key,raw[i]])) : raw;
const sameId = (a,b) => String(a) === String(b);
function assign(raw, values) {
    for (const [key,value] of Object.entries(values)) {
        if (Array.isArray(raw)) raw[fields.indexOf(key)] = value;
        else raw[key] = value;
    }
}
function validSettings(v) {
    return typeof v.enabled === 'boolean' && Number.isSafeInteger(v.seed) && v.seed >= 0
        && Number.isFinite(v.seedvr2_upscale_factor) && v.seedvr2_upscale_factor >= 1 && v.seedvr2_upscale_factor <= 8
        && ['lanczos','nearest','bilinear','bicubic'].includes(v.resampling_method)
        && typeof v.seedvr2_model_name === 'string' && typeof v.seedvr2_vae_name === 'string'
        && Number.isInteger(v.seedvr2_steps) && v.seedvr2_steps >= 1
        && Number.isFinite(v.seedvr2_denoise) && v.seedvr2_denoise > 0 && v.seedvr2_denoise <= 1
        && ['none','lab','wavelet','adain'].includes(v.seedvr2_color_correction)
        && Number.isInteger(v.seedvr2_vae_tile_size) && v.seedvr2_vae_tile_size >= 128;
}
export function repairSeedVR2Workflow(workflow) {
    const report = {widgetOrders:0, inpaintOrders:0, warnings:[]};
    // Only the tagged V4/V5 workflow line shipped this layout. Do not move
    // composites in unrelated user-authored post-processing graphs.
    if (!['V4 Beta','V5'].includes(workflow?.extra?.donut_workflow?.release)) return report;
    for (const graph of [workflow, ...(workflow?.definitions?.subgraphs || [])]) {
        if (!Array.isArray(graph?.nodes)) continue;
        const nodes = new Map(graph.nodes.map(node => [String(node.id),node]));
        const links = (Array.isArray(graph.links) ? graph.links : []).filter(Boolean).map(raw => ({raw, e:edge(raw)}));
        for (const post of graph.nodes.filter(node => node.type === 'DonutSeedVR2Upscale')) {
            const named = post.widgets_values_named;
            // INT widgets named seed acquire a frontend-only control widget.
            // The old ten-value exports omitted it, shifting every model and
            // sampling field. Only repair that known shape; a modern saved
            // array is authoritative over potentially stale named metadata.
            if (post.widgets_values?.length === POST_WIDGET_ORDER.length - 1 && named
                    && POST_WIDGET_ORDER.filter(key => key !== 'control_after_generate')
                        .every(key => Object.hasOwn(named,key)) && validSettings(named)) {
                const control = ['fixed','increment','decrement','randomize'].includes(named.control_after_generate)
                    ? named.control_after_generate : 'fixed';
                post.widgets_values = POST_WIDGET_ORDER.map(key => key === 'control_after_generate' ? control : named[key]);
                post.widgets_values_named = {...named, control_after_generate:control};
                report.widgetOrders++;
            }
            const inputSlot = post.inputs?.findIndex(input => input.name === 'image');
            if (inputSlot == null || inputSlot < 0) continue;
            const middle = links.find(({e}) => sameId(e.id,post.inputs[inputSlot].link));
            if (!middle || middle.e.type !== 'IMAGE' || !sameId(middle.e.target_id,post.id) || middle.e.target_slot !== inputSlot) continue;
            const composite = nodes.get(String(middle.e.origin_id));
            if (composite?.type !== 'DonutInpaintComposite') continue;
            const cSlot = composite.inputs?.findIndex(input => input.name === 'image');
            const before = cSlot >= 0 && links.find(({e}) => sameId(e.id,composite.inputs[cSlot].link));
            const cOut = links.filter(({e}) => sameId(e.origin_id,composite.id));
            const after = links.filter(({e}) => sameId(e.origin_id,post.id));
            // Do not reorder a shared/custom composite branch or guess a cable.
            // PR #66 also left its composite output backlink stale; the link
            // table and matching target inputs are the authority for this fix.
            const targetsMatch = after.every(({e}) => {
                if (sameId(e.target_id,-20)) return graph.outputs?.[e.target_slot]?.linkIds?.some(id => sameId(id,e.id));
                return sameId(nodes.get(String(e.target_id))?.inputs?.[e.target_slot]?.link,e.id);
            });
            if (!before || before.e.type !== 'IMAGE' || !sameId(before.e.target_id,composite.id)
                    || before.e.target_slot !== cSlot || middle.e.origin_slot !== 0
                    || sameId(before.e.origin_id,post.id) || sameId(before.e.origin_id,composite.id)
                    || cOut.length !== 1 || !after.length || !targetsMatch
                    || after.some(({e}) => e.type !== 'IMAGE' || e.origin_slot !== 0 || sameId(e.target_id,composite.id) || sameId(e.target_id,post.id))
                    || !post.outputs?.[0] || !composite.outputs?.[0]) {
                report.warnings.push(`SeedVR2 node ${post.id}: custom/shared inpaint wiring was left unchanged; place a preservation composite after the post-pass.`);
                continue;
            }
            // regular finishing -> SeedVR2 -> preserve A -> existing consumers.
            // Preserve every link ID (including subgraph interface linkIds).
            assign(before.raw, {target_id:post.id,target_slot:inputSlot});
            assign(middle.raw, {origin_id:post.id,origin_slot:0,target_id:composite.id,target_slot:cSlot});
            for (const {raw} of after) assign(raw, {origin_id:composite.id,origin_slot:0});
            post.inputs[inputSlot].link = before.e.id;
            composite.inputs[cSlot].link = middle.e.id;
            post.outputs[0].links = [middle.e.id];
            composite.outputs[0].links = after.map(({e}) => e.id);
            report.inpaintOrders++;
        }
    }
    return report;
}

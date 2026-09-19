// Bake the existing V4 repairs/categories plus the new crop defaults into JSON.
// Authoring only. Does not run in ComfyUI and never overwrites its input.
const fs=require('node:fs');
const {migrate}=require('./organize_v4_panels.cjs');
const EDIT_ORDER=['enabled','image_a','image_b','use_reference_b','prompt','resolution_mode','aspect_ratio','megapixels','width','height','multiple','grounding_px','lora_name','lora_strength','crop_a_x','crop_a_y','crop_b_x','crop_b_y','inpaint_enabled','mask_data','mask_feather','grounding_schedule','grounding_start_px','grounding_end_px','mask_b_mode','mask_b_model','mask_b_data','mask_b_grow','mask_b_feather','mask_b_background','geometry_mode','crop_data_a','crop_data_b','output_canvas'];
const EDIT_DEFAULTS={grounding_schedule:'constant',grounding_start_px:512,grounding_end_px:1088,mask_b_mode:'Off',mask_b_model:'birefnet.safetensors',mask_b_data:'',mask_b_grow:0,mask_b_feather:0,mask_b_background:'Neutral gray'};
function prepare(workflow){
    migrate(workflow);
    const contexts=[workflow,...(workflow.definitions?.subgraphs||[])];
    let edits=0,guides=0;
    for(const graph of contexts)for(const node of graph.nodes||[]){
        if(node.type==='DonutEditStudio'){
            const named={...EDIT_DEFAULTS,...node.widgets_values_named,
                geometry_mode:'Independent crops',crop_data_a:'',crop_data_b:'',output_canvas:'Follow A crop'};
            // Authoring this release's bundled starter, not user workflow migration.
            for(const name of EDIT_ORDER)if(!Object.hasOwn(named,name))throw new Error(`Missing named Edit Studio widget: ${name}`);
            node.widgets_values_named=named;node.widgets_values=EDIT_ORDER.map(name=>named[name]);
            node.properties||={};node.properties.donut_widget_order=[...EDIT_ORDER];edits++;
        }
        if(node.type==='DonutReferenceStudio'){
            const order=['enabled','image_a','image_b','use_reference_b','edit_active','geometry_mode','crop_data_a','crop_data_b'];
            const named={...Object.fromEntries(order.slice(0,5).map((name,i)=>[name,node.widgets_values?.[i]])),...node.widgets_values_named,
                geometry_mode:'Independent crops',crop_data_a:'',crop_data_b:''};
            for(const name of order)if(named[name]===undefined)throw new Error(`Missing Reference Guidance widget: ${name}`);
            node.widgets_values_named=named;node.widgets_values=order.map(name=>named[name]);
            node.properties||={};node.properties.donut_widget_order=order;guides++;
        }
        for(const group of node.properties?.donut_app_controls?.groups||[]){
            if(group.donut_image_size&&!group.controls?.some(c=>c.widget==='output_canvas'))
                group.controls.unshift({path:[...group.donut_image_size],widget:'output_canvas',title:'Output canvas'});
        }
    }
    if(edits!==1||guides!==1)throw new Error('Expected the bundled V4 starter with one Edit Studio and one Reference Guidance node.');
    validateLinks(workflow);
    return workflow;
}
function validateLinks(workflow){
    for(const g of [workflow,...(workflow.definitions?.subgraphs||[])]){
        const nodes=new Map((g.nodes||[]).map(n=>[String(n.id),n]));
        if(nodes.size!==(g.nodes||[]).length)throw new Error('Duplicate node IDs.');
        const edges=(g.links||[]).map(e=>Array.isArray(e)?{id:e[0],origin_id:e[1],origin_slot:e[2],target_id:e[3],target_slot:e[4]}:e);
        if(new Set(edges.map(e=>e.id)).size!==edges.length)throw new Error('Duplicate link IDs.');
        const byId=new Map(edges.map(e=>[e.id,e]));
        for(const e of edges){
            if(e.origin_id!==-10){const n=nodes.get(String(e.origin_id));if(!n?.outputs?.[e.origin_slot]?.links?.includes(e.id))throw new Error(`Missing source backlink for ${e.id}`);}
            if(e.target_id!==-20){const n=nodes.get(String(e.target_id));if(n?.inputs?.[e.target_slot]?.link!==e.id)throw new Error(`Missing target backlink for ${e.id}`);}
        }
        for(const n of nodes.values()){
            for(const [i,o]of(n.outputs||[]).entries())for(const id of o.links||[]){const e=byId.get(id);if(!e||String(e.origin_id)!==String(n.id)||e.origin_slot!==i)throw new Error(`Stale output link ${id}`);}
            for(const [i,input]of(n.inputs||[]).entries())if(input.link!=null){const e=byId.get(input.link);if(!e||String(e.target_id)!==String(n.id)||e.target_slot!==i)throw new Error(`Stale input link ${input.link}`);}
        }
    }
}
if(require.main===module){
    try{
        const [input,output,...extra]=process.argv.slice(2);
        if(!input||!output||extra.length)throw new Error('Usage: node tools/prepare_independent_crops.cjs bundled-input.json NEW-output.json');
        fs.writeFileSync(output,JSON.stringify(prepare(JSON.parse(fs.readFileSync(input,'utf8'))),null,2)+'\n',{flag:'wx'});
    }catch(e){console.error(e.message);process.exitCode=1;}
}
module.exports={prepare,validateLinks,EDIT_ORDER};

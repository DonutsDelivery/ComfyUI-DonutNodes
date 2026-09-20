const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
function load(file, names) {
    const source = fs.readFileSync(path.join(__dirname,'..','web',file),'utf8').replace(/export (function|const)/g,'$1');
    const context = vm.createContext({});
    return vm.runInContext(`${source}\n({${names.join(',')}})`,context);
}
const {organizeV4Panels,categorizeGroups,graphEntries,SIZE_FIELDS,splitV4FinishingPanels} = load('donut_panel_categories_model.js',['organizeV4Panels','categorizeGroups','graphEntries','SIZE_FIELDS','splitV4FinishingPanels']);
const {repairSeedVR2Workflow,POST_WIDGET_ORDER} = load('donut_seedvr2_workflow_repair.js',['repairSeedVR2Workflow','POST_WIDGET_ORDER']);
const plain = value => JSON.parse(JSON.stringify(value));
const same = (a,b) => assert.deepEqual(plain(a),plain(b));
const fields = (path,names) => names.map(widget => ({path,widget,title:widget}));
function panel(id,title,groups) {return {id,type:'DonutWorkflowPanel',title,properties:{donut_app_controls:{seed_path:[900,1],groups}}};}
function fixture() {
    const studio = {id:700,type:'DonutEditStudio',properties:{donut_seed_path:[900,1]},widgets_values_named:{enabled:false,resolution_mode:'Preset',aspect_ratio:'4:3 Standard',megapixels:1,width:1152,height:896,multiple:'64'}};
    const models = panel(1,'01 · Models',[
        {title:'Primary model',controls:fields([900,3],['unet_name','weight_dtype'])},
        {title:'Fusion',controls:fields([800],['tap_strength','compatibility_preset'])},
        {title:'Advanced · Fusion controls',advanced:true,controls:fields([800,9],['tap_method','projector_strength','fusion_method','uncensorfix_controls','execution_mode'])},
        {title:'AuraFlow sampling',controls:[{path:[900,4],mode:'bypass',title:'Enable AuraFlow sampling'},...fields([900,4],['shift'])]},
        {title:'Advanced · Model merge',advanced:true,visible_when:{path:[900,3],widget:'model_mode',value:'Merge two models'},controls:fields([900,3],['first.','blocks.0.','ratio_mode'])},
    ]);
    const generate = panel(2,'06 · Generate & finish',[
        {title:'Generation and finish',color:'blue',controls:[...fields([800],['turbo_mode','steps','sampler_name','scheduler','cfg_start','denoise','rescale_factor','tiled_diffusion','upscale_1_enabled','denoise_1','max_faces','denoise_2','upscale_2_enabled']),{path:[800,5],mode:'bypass',title:'Face detail',fallback_type:'DonutFaceDetailer'}]},
        {title:'Advanced · Latent batch',advanced:true,controls:fields([800,2],['batch_size'])},
        {title:'Advanced · First upscale',advanced:true,controls:fields([800,3],['resampling_method','feather','tiled_vae','nag_enabled'])},
        {title:'Advanced · Second upscale',advanced:true,controls:fields([800,4],['rescale_factor','tiled_diffusion','nag_phi'])},
        {title:'Advanced · First upscale · engine',donut_seedvr2:true,controls:fields([800,3],['upscale_engine'])},
        {title:'Advanced · First upscale · SeedVR2',advanced:true,donut_seedvr2:true,visible_when:{path:[800,3],widget:'upscale_engine',value:'SeedVR2'},controls:fields([800,3],['seedvr2_model_name','seedvr2_denoise'])},
        {title:'SeedVR2 · post upscale',controls:fields([800,8],['enabled','seedvr2_upscale_factor','resampling_method','seedvr2_model_name','seedvr2_vae_name','seedvr2_steps','seedvr2_denoise','seedvr2_color_correction','seedvr2_vae_tile_size','seedvr2_vae_overlap','seed'])},
    ]);
    const loras = panel(3,'02 · LoRAs & block weights',[
        {title:'LoRAs',loras:[900,7],controls:[]},
        {title:'LoRA settings',controls:fields([900,7],['model_type','global_block_vector','civitai_lookup','safe_stack','fusion_aware','max_fusion_boost','safe_limit','execution_mode'])},
    ]);
    const prompts = panel(4,'04 · Prompts',[
        {title:'Prompt',shared_prompt_tools:true,controls:[{path:[900,10],widget:'Text',title:'Prompt',ui:{prompt:true}}]},
        {title:'Prompt variants',prompt_sets:[800,7],prompt_source_paths:[[900,10]],controls:fields([800,7],['prompt_set_index','control_after_generate'])},
    ]);
    const guidance = panel(5,'05 · Seed & guidance',[
        {title:'Shared seed',controls:fields([900,1],['seed','fixed'])},
        {title:'Guidance',controls:fields([800],['alpha','variance_enabled'])},
        {title:'Advanced · Prompt conditioning and variance',advanced:true,controls:fields([800,7],['separator','edit_negative','variance_strength'])},
    ]);
    const save = panel(6,'08 · Save images',[
        {title:'Final save',controls:fields([70],['root','filename_delimiter','overwrite_mode','extension','quality','embed_workflow','show_previews'])},
        {title:'Secondary save',controls:fields([71],['filename_prefix'])},
    ]);
    return {nodes:[models,generate,loras,prompts,guidance,save,studio],links:[[101,1,0,2,0,'MODEL']],extra:{donut_workflow:{release:'V4 Beta'}}};
}
const groups = panel => panel.properties.donut_app_controls.groups;
const controls = graph => graphEntries(graph).flatMap(({node}) => groupsSafe(node).flatMap(group => group.controls || []));
function groupsSafe(node) {return node.properties?.donut_app_controls?.groups || [];}
const signature = control => JSON.stringify(control);
function findGroup(graph,widget,pathEnd) {return graph.nodes.flatMap(node => groupsSafe(node)).find(group => group.controls?.some(control => control.widget === widget && (pathEnd === undefined || control.path.at(-1) === pathEnd)));}

test('every existing control and all backend values/links survive regrouping',()=>{
    const graph=fixture(), links=plain(graph.links), state=plain(graph.nodes.at(-1).widgets_values_named);
    const before=controls(graph).map(signature).sort();
    organizeV4Panels(graph);
    const after=controls(graph).filter(control=>control.path[0] !== 700).map(signature).sort();
    // The NAG experiment entries are injected aliases of existing Fusion
    // Control widgets (path [800]); they are additions, not moves, so the
    // surviving set is the pre-organize set plus the two aliases.
    const aliases=[
        '{"path":[800],"widget":"nag_batch_txtfusion","title":"Batch equal-length text fusion"}',
        '{"path":[800],"widget":"nag_text_energy_compensation","title":"Text-energy compensation"}',
    ];
    same(after,[...before,...aliases].sort()); same(graph.links,links); same(graph.nodes.at(-1).widgets_values_named,state);
});
test('repeated imports/save/reload are idempotent',()=>{
    const graph=fixture(); organizeV4Panels(graph); const saved=JSON.stringify(graph);
    same(organizeV4Panels(graph),[]); assert.equal(JSON.stringify(graph),saved);
    const reloaded=JSON.parse(saved); same(organizeV4Panels(reloaded),[]);
});
test('global size is exposed in Generate; source values are not copied or reset',()=>{
    const graph=fixture(); organizeV4Panels(graph);
    const size=groups(graph.nodes[1]).find(group=>group.donut_image_size);
    same(size.controls.map(control=>control.widget),SIZE_FIELDS.map(([name])=>name));
    assert.ok(size.controls.every(control=>control.path[0] === 700));
    assert.equal(graph.nodes.at(-1).widgets_values_named.enabled,false);
});
test('multiple studios in the same family do not choose an arbitrary size owner',()=>{
    const graph=fixture(); graph.nodes.push({...plain(graph.nodes.at(-1)),id:701}); organizeV4Panels(graph);
    assert.ok(!groups(graph.nodes[1]).some(group=>group.donut_image_size));
});
test('remapped numeric/string node IDs and live widgets are supported',()=>{
    const graph=fixture(), studio=graph.nodes.at(-1); studio.id='renamed-studio';
    studio.widgets=Object.entries(studio.widgets_values_named).map(([name,value])=>({name,value})); delete studio.widgets_values_named;
    organizeV4Panels(graph); same(groups(graph.nodes[1]).find(group=>group.donut_image_size).donut_image_size,['renamed-studio']);
});
test('uncensorfix is standard and retains its original callback target',()=>{
    const graph=fixture(); organizeV4Panels(graph); const group=findGroup(graph,'uncensorfix_controls');
    assert.equal(group.title,'UncensorFix'); assert.equal(group.advanced,false); same(group.controls[0].path,[800,9]);
});
test('V5 guidance panel exposes both NAG experiment toggles from Fusion Control',()=>{
    const graph=fixture(); organizeV4Panels(graph);
    const guidance=groups(graph.nodes[4]);
    for(const widget of ['nag_text_energy_compensation','nag_batch_txtfusion']) {
        const group=guidance.find(group=>group.controls?.some(control=>control.widget===widget));
        assert.ok(group); assert.equal(group.title,'Negative attention guidance · NAG'); assert.equal(group.advanced,true);
        same(group.controls.find(control=>control.widget===widget).path,[800]);
    }
});
test('AuraFlow moves from Models to the sampling panel without duplication',()=>{
    const graph=fixture(); organizeV4Panels(graph);
    assert.ok(!groups(graph.nodes[0]).some(group=>/AuraFlow/.test(group.title)));
    assert.equal(groups(graph.nodes[1]).filter(group=>/AuraFlow/.test(group.title)).length,1);
});
test('main generation/first hires/face/second hires are separate categories',()=>{
    const graph=fixture(); organizeV4Panels(graph);
    for(const [widget,title] of [['steps','Base sampling'],['denoise','Donut hires · first upscale'],['denoise_1','Face detail'],['denoise_2','Donut hires · second upscale']]) assert.equal(findGroup(graph,widget).title,title);
});
test('legacy engine replacement is advanced; the post-pass remains standard',()=>{
    const graph=fixture(); organizeV4Panels(graph);
    assert.equal(findGroup(graph,'upscale_engine').advanced,true);
    assert.equal(findGroup(graph,'seedvr2_model_name',8).advanced,false);
    assert.equal(findGroup(graph,'seedvr2_denoise',8).advanced,true);
    same(findGroup(graph,'seedvr2_denoise',3).visible_when,{path:[800,3],widget:'upscale_engine',value:'SeedVR2'});
});
test('LoRA rows, prompt variants and shared wildcard configuration stay intact',()=>{
    const graph=fixture(), lora=plain(groups(graph.nodes[2])[0]), prompt=plain(groups(graph.nodes[3])); organizeV4Panels(graph);
    for(const source of [lora,...prompt]) {
        const found=graph.nodes.flatMap(groupsSafe).find(group=>group.title === source.title);
        const {donut_category_rank,...rest}=found; same(rest,source);
    }
});
test('prompt composition is no longer under variance',()=>{
    const graph=fixture(); organizeV4Panels(graph);
    assert.ok(groups(graph.nodes[3]).some(group=>group.controls.some(control=>control.widget === 'edit_negative')));
    assert.ok(!groups(graph.nodes[4]).some(group=>group.controls.some(control=>control.widget === 'edit_negative')));
});
test('save panel separates filenames/format/metadata while keeping secondary output',()=>{
    const graph=fixture(); organizeV4Panels(graph);
    for(const [w,title] of [['quality','Format and compression'],['root','Destination and filenames'],['embed_workflow','Metadata and previews'],['filename_prefix','Secondary save']]) assert.equal(findGroup(graph,w).title,title);
});
test('unknown controls, descriptions and visibility predicates are retained',()=>{
    const group={title:'Custom tool',description:'keep me',visible_when:{path:[1],widget:'enabled',value:true},controls:fields([42],['future_option'])};
    const result=categorizeGroups('generate',[group]); same(result[0].controls,group.controls); same(result[0].visible_when,group.visible_when); assert.equal(result[0].description,'keep me');
});
test('serialized subgraphs are resolved for image-size paths',()=>{
    const graph=fixture(), studio=graph.nodes.pop(); graph.nodes.push({id:'outer',type:'definition'}); graph.definitions={subgraphs:[{id:'definition',nodes:[studio]}]};
    organizeV4Panels(graph); same(groups(graph.nodes[1]).find(group=>group.donut_image_size).donut_image_size,['outer',700]);
});

function postFixture(asArrays=false) {
    const named={enabled:false,seed:0,seedvr2_upscale_factor:2,resampling_method:'lanczos',seedvr2_model_name:'3b.safetensors',seedvr2_vae_name:'vae.safetensors',seedvr2_steps:1,seedvr2_denoise:1,seedvr2_color_correction:'none',seedvr2_vae_tile_size:1024,seedvr2_vae_overlap:128};
    const graph={extra:{donut_workflow:{release:'V4 Beta'}},nodes:[
        {id:11,type:'Finisher',outputs:[{name:'image',type:'IMAGE',links:[90]}]},
        {id:12,type:'DonutInpaintComposite',inputs:[{name:'image',type:'IMAGE',link:90},{name:'inpaint',type:'DONUT_INPAINT',link:93}],outputs:[{name:'image',type:'IMAGE',links:[92]}]},
        {id:13,type:'DonutSeedVR2Upscale',inputs:[{name:'image',type:'IMAGE',link:91}],outputs:[{name:'image',type:'IMAGE',links:[92]}],widgets_values:[false,0,2,'lanczos','3b.safetensors','vae.safetensors',1,1,'none',512],widgets_values_named:named},
    ],inputs:[{name:'inpaint',type:'DONUT_INPAINT',linkIds:[93]}],outputs:[{name:'image',type:'IMAGE',linkIds:[92]}],links:[
        [90,11,0,12,0,'IMAGE'],[91,12,0,13,0,'IMAGE'],[92,13,0,-20,0,'IMAGE'],[93,-10,0,12,1,'DONUT_INPAINT'],
    ]};
    if(!asArrays) graph.links=graph.links.map(row=>Object.fromEntries(['id','origin_id','origin_slot','target_id','target_slot','type'].map((key,i)=>[key,row[i]])));
    return graph;
}
function validate(graph) {
    const nodes=new Map(graph.nodes.map(node=>[node.id,node]));
    for(const raw of graph.links) {
        const e=Array.isArray(raw)?Object.fromEntries(['id','origin_id','origin_slot','target_id','target_slot','type'].map((k,i)=>[k,raw[i]])):raw;
        if(e.origin_id === -10) assert.ok(graph.inputs[e.origin_slot].linkIds.includes(e.id));
        else assert.ok(nodes.get(e.origin_id).outputs[e.origin_slot].links.includes(e.id));
        if(e.target_id === -20) assert.ok(graph.outputs[e.target_slot].linkIds.includes(e.id));
        else assert.equal(nodes.get(e.target_id).inputs[e.target_slot].link,e.id);
    }
}
test('post widgets follow actual schema order, with enabled=false preserved',()=>{
    const graph=postFixture(); const named=plain(graph.nodes[2].widgets_values_named); const report=repairSeedVR2Workflow(graph);
    assert.equal(report.widgetOrders,1); same(graph.nodes[2].widgets_values,POST_WIDGET_ORDER.map(key=>key === 'control_after_generate' ? 'fixed' : named[key])); assert.equal(graph.nodes[2].widgets_values[2],false);
});
for(const arrays of [false,true]) test(`post-pass is before final inpaint preservation; reciprocal links repaired (${arrays?'arrays':'objects'})`,()=>{
    const graph=postFixture(arrays), inputs=plain(graph.inputs), outputs=plain(graph.outputs);
    assert.throws(()=>validate(graph)); const report=repairSeedVR2Workflow(graph); validate(graph);
    assert.equal(report.inpaintOrders,1); same(graph.inputs,inputs); same(graph.outputs,outputs);
    assert.equal(graph.nodes[2].inputs[0].link,90); assert.equal(graph.nodes[1].inputs[0].link,91);
    const saved=JSON.stringify(graph); same(repairSeedVR2Workflow(graph),{widgetOrders:0,inpaintOrders:0,warnings:[]}); assert.equal(JSON.stringify(graph),saved);
});
test('shared composites are reported, not silently rewired',()=>{
    const graph=postFixture(); graph.links.push({id:94,origin_id:12,origin_slot:0,target_id:77,target_slot:0,type:'IMAGE'});
    const links=plain(graph.links); const report=repairSeedVR2Workflow(graph); assert.equal(report.inpaintOrders,0); assert.equal(report.warnings.length,1); same(graph.links,links);
});
test('invalid named settings are never coerced into new defaults',()=>{
    const graph=postFixture(); graph.nodes[2].widgets_values_named.seed='bad'; const values=plain(graph.nodes[2].widgets_values);
    repairSeedVR2Workflow(graph); same(graph.nodes[2].widgets_values,values);
});
test('nested SeedVR2 repair leaves surrounding graph structure intact',()=>{
    const sub=postFixture(), root={extra:{donut_workflow:{release:'V4 Beta'}},nodes:[],links:[],definitions:{subgraphs:[sub]}}; const report=repairSeedVR2Workflow(root);
    assert.equal(report.inpaintOrders,1); validate(sub); same(root.nodes,[]); same(root.links,[]);
});

test('compatible promoted and direct controls merge into one stage category',()=>{
    const graph=fixture(); const list=groups(graph.nodes[1]);
    for(const group of list) group.color='blue';
    organizeV4Panels(graph);
    assert.equal(groups(graph.nodes[1]).filter(group=>group.title === 'Donut hires · second upscale').length,1);
    const group=groups(graph.nodes[1]).find(group=>group.title === 'Donut hires · second upscale');
    assert.equal(group.controls[0].widget,'upscale_2_enabled');
    const original=JSON.stringify(graph); organizeV4Panels(graph); assert.equal(JSON.stringify(graph),original);
});
test('workflow hook repairs post ordering before generic subgraph validation',()=>{
    const source=fs.readFileSync(path.join(__dirname,'..','web','donut_workflow.js'),'utf8')
        .replace(/^import .*;\n/gm,'').replace(/^export .*;\n/gm,'');
    const calls=[], sandbox={app:{registerExtension:extension=>sandbox.extension=extension},api:{},createLoraService:()=>({}),
        repairSeedVR2Workflow:data=>{calls.push('seed'); return repairSeedVR2Workflow(data);},
        repairEditStudioMetadata:()=>calls.push('edit'),repairStreamlinedWorkflowSafely:data=>{calls.push('generic');validate(data);}};
    vm.createContext(sandbox); vm.runInContext(source,sandbox);
    sandbox.extension.beforeConfigureGraph(postFixture()); same(calls,['seed','edit','generic']);
});

test('untagged custom post-processing workflows are not altered',()=>{
    const graph=postFixture(); delete graph.extra; const saved=JSON.stringify(graph);
    same(repairSeedVR2Workflow(graph),{widgetOrders:0,inpaintOrders:0,warnings:[]}); assert.equal(JSON.stringify(graph),saved);
});
test('panels without shared ownership metadata are not cross-wired',()=>{
    const graph=fixture(); for(const node of graph.nodes) if(node.properties?.donut_app_controls) delete node.properties.donut_app_controls.seed_path;
    organizeV4Panels(graph); assert.ok(!groups(graph.nodes[1]).some(group=>group.donut_image_size));
    assert.ok(groups(graph.nodes[0]).some(group=>group.title === 'AuraFlow sampling'));
});
test('authoring helper uses the same repairs and category definitions',()=>{
    const {migrate}=require('../tools/organize_v4_panels.cjs');
    const graph=fixture(); graph.definitions={subgraphs:[postFixture()]};
    const report=migrate(graph); assert.equal(report.inpaintOrders,1); assert.ok(report.panels>0);
    validate(graph.definitions.subgraphs[0]);
    same(migrate(graph),{widgetOrders:0,inpaintOrders:0,warnings:[],panels:0,finishingPanels:0});
    assert.throws(()=>migrate({nodes:[]}),/tagged/);
});


test('modern SeedVR2 arrays keep edited values rather than stale named metadata',()=>{
    const graph=postFixture(); repairSeedVR2Workflow(graph);
    graph.nodes[2].widgets_values[4]='increment';
    graph.nodes[2].widgets_values[5]='custom-7b.safetensors';
    const values=plain(graph.nodes[2].widgets_values);
    repairSeedVR2Workflow(graph); same(graph.nodes[2].widgets_values,values);
});
test('split finishing panels preserve every control and all execution nodes',()=>{
    const graph=fixture(); graph.extra.donut_layout={columns:[[graph.nodes[1].id]]};
    graph.extra.linearData={inputs:[[graph.nodes[1].id,'workflow_controls']]};
    organizeV4Panels(graph);
    const controls=()=>graph.nodes.flatMap(n=>n.properties?.donut_app_controls?.groups || [])
        .flatMap(g=>g.controls || []).map(c=>JSON.stringify(c)).sort();
    const before=controls(); const links=JSON.stringify(graph.links);
    const created=splitV4FinishingPanels(graph);
    assert.equal(created.length,4); same(controls(),before);
    assert.equal(JSON.stringify(graph.links),links);
    assert.equal(new Set(graph.nodes.map(n=>n.id)).size,graph.nodes.length);
    const saved=JSON.stringify(graph); same(splitV4FinishingPanels(graph),[]);
    assert.equal(JSON.stringify(graph),saved);
    assert.equal(graph.extra.donut_layout.columns.length,3);
    same(graph.extra.linearData.inputs.slice(1),created.map(n=>[n.id,'workflow_controls']));
});
test('shipped SeedVR2 values include ComfyUI frontend seed control and discover both files',()=>{
    const graph=JSON.parse(fs.readFileSync(path.join(__dirname,'../workflows/v5/DonutWF_v5.json')));
    const post=graphEntries(graph).find(({node})=>node.type==='DonutSeedVR2Upscale').node;
    // This is the actual required + optional widget order, INCLUDING the
    // control automatically inserted by ComfyUI after an INT named seed.
    const names=['seedvr2_upscale_factor','resampling_method','enabled','seed','control_after_generate',
        'seedvr2_model_name','seedvr2_vae_name','seedvr2_steps','seedvr2_denoise','seedvr2_color_correction','seedvr2_vae_tile_size','seedvr2_vae_overlap'];
    assert.equal(post.widgets_values.length,names.length);
    post.widgets=names.map((name,i)=>({name,value:post.widgets_values[i]}));
    const {modelBindings}=load('donut_model_requirements.js',['modelBindings']);
    same(modelBindings({nodes:[post]}).map(({folder,name})=>({folder,name})),[
        {folder:'diffusion_models',name:'seedvr2_3b_int8_convrot.safetensors'},
        {folder:'vae',name:'seedvr2_ema_vae_fp16.safetensors'}]);
    assert.equal(post.widgets.find(w=>w.name==='seedvr2_denoise').value,1);
    assert.equal(post.widgets.find(w=>w.name==='seedvr2_color_correction').value,'none');
    const original=JSON.stringify(graph); repairSeedVR2Workflow(graph);
    assert.equal(JSON.stringify(graph),original);
});

test('fresh-install workflow uses public Krea2 without changing personal model choices',()=>{
    const shipped=JSON.parse(fs.readFileSync(path.join(__dirname,'../workflows/v5/DonutWF_v5.json')));
    const nodes=graphEntries(shipped).map(({node})=>node);
    const loaders=nodes.filter(n=>n.type==='UNETLoader');
    assert.ok(loaders.length>=1);
    for(const node of loaders) assert.equal(node.widgets_values_named.unet_name,'krea2_turbo_bf16.safetensors');
    assert.equal(nodes.find(n=>n.type==='DonutModelMergeKrea2').widgets_values_named.model_mode,'Single model');
    loaders[0].widgets_values_named.unet_name='personal.safetensors';
    loaders[0].widgets_values[0]='personal.safetensors';
    organizeV4Panels(shipped); splitV4FinishingPanels(shipped);
    assert.equal(loaders[0].widgets_values[0],'personal.safetensors');
});

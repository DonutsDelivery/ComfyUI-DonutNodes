const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const {validateLinks}=require('../tools/prepare_independent_crops.cjs');
const source=fs.readFileSync('web/donut_preview_stages.js','utf8').replace(/^export /gm,'');
const {addStagePreviews,previewSourceId}=vm.runInNewContext(source+';({addStagePreviews,previewSourceId})');
const workflow=()=>JSON.parse(fs.readFileSync('workflows/v5/DonutWF_v5.json'));
function oldWorkflow(){
 const w=workflow(),g=w.definitions.subgraphs.find(g=>g.nodes.some(n=>n.type==='DonutFaceDetailer'));
 const removed=new Set(g.nodes.filter(n=>n.properties?.donut_preview_stage).map(n=>n.id));
 const edges=new Set(g.links.filter(l=>removed.has(l.target_id)||removed.has(l.origin_id)).map(l=>l.id));
 g.nodes=g.nodes.filter(n=>!removed.has(n.id));g.links=g.links.filter(l=>!edges.has(l.id));
 for(const n of g.nodes)for(const o of n.outputs||[])o.links=(o.links||[]).filter(id=>!edges.has(id));
 for(const i of g.inputs)i.linkIds=(i.linkIds||[]).filter(id=>!edges.has(id));
 const p=w.nodes.find(n=>n.type==='DonutLatestPreview');p.properties.sources={'912':'Base generation','913':'First upscale','914':'Final image'};delete p.properties.source_order;
 return w;
}
test('shipped stages follow the real chain with valid backlinks',()=>{
 const w=workflow();validateLinks(w);const p=w.nodes.find(n=>n.type==='DonutLatestPreview');
 assert.deepEqual(p.properties.source_order.map(id=>p.properties.sources[id]),['Base generation','First upscale','Second upscale','Face Detailer','SeedVR2 / final image']);
 const g=w.definitions.subgraphs.find(g=>g.nodes.some(n=>n.type==='DonutFaceDetailer'));
 const face=g.nodes.find(n=>n.type==='DonutFaceDetailer');
 const secondSource=g.links.find(l=>l.id===face.inputs.find(i=>i.name==='image').link).origin_id;
 const preview=g.nodes.find(n=>n.properties?.donut_preview_stage==='second');
 assert.equal(g.links.find(l=>l.id===preview.inputs[0].link).origin_id,secondSource);
 const facePreview=g.nodes.find(n=>n.properties?.donut_preview_stage==='face');
 const composite=g.nodes.find(n=>n.id===g.links.find(l=>l.id===facePreview.inputs[0].link).origin_id);
 assert.equal(composite.type,'DonutInpaintComposite');
 assert.equal(g.links.find(l=>l.id===composite.inputs[0].link).origin_id,face.id);
});
test('migration adds only preview branches and preserves all original links and settings',()=>{
 const w=oldWorkflow(),before=structuredClone(w);assert.equal(addStagePreviews(w),true);validateLinks(w);
 const g=w.definitions.subgraphs[0],old=before.definitions.subgraphs[0];
 for(const e of old.links)assert.deepEqual(g.links.find(l=>l.id===e.id),e);
 for(const n of old.nodes){const updated=g.nodes.find(x=>x.id===n.id);assert.deepEqual(updated.inputs,n.inputs);assert.deepEqual(updated.widgets_values,n.widgets_values);}
 const once=JSON.stringify(w);assert.equal(addStagePreviews(w),false);assert.equal(JSON.stringify(w),once);
});
test('unrelated or ambiguous graphs are untouched',()=>{
 const w=oldWorkflow();w.extra.donut_workflow.release='custom';const before=JSON.stringify(w);assert.equal(addStagePreviews(w),false);assert.equal(JSON.stringify(w),before);
});
test('nested executed node is matched even when display node is the enclosing graph',()=>{
 const sources={'1014:1171':'Second upscale','914':'Final'};
 assert.equal(previewSourceId({node:'1014:1171',display_node:'1014'},sources),'1014:1171');
 assert.equal(previewSourceId({node:'runtime-id',display_node:'914'},sources),'914');
 assert.equal(previewSourceId({},sources),undefined);
});
test('runtime subgraph renumbering rebinds face and second previews without losing selection',()=>{
 const {rebindStageSources}=vm.runInNewContext(source+';({rebindStageSources})');
 const properties={sources:{'912':'Base generation','1014:1171':'Second upscale','1014:1173':'Face Detailer','914':'SeedVR2 / final image'},source_order:['912','1014:1171','1014:1173','914'],preview_selection:'1014:1173'};
 rebindStageSources(properties,[{node:{properties:{donut_preview_stage:'second'}},path:[1014,1175]},{node:{properties:{donut_preview_stage:'face'}},path:[1014,1177]}]);
 assert.equal(properties.preview_selection,'1014:1177');
 assert.equal(previewSourceId({node:'1014:1177',display_node:'1014'},properties.sources),'1014:1177');
 assert.equal(properties.sources['1014:1173'],undefined);
 assert.equal(properties.source_order[1],'1014:1175');
 const before=JSON.stringify(properties);rebindStageSources(properties,[]);assert.equal(JSON.stringify(properties),before);
});

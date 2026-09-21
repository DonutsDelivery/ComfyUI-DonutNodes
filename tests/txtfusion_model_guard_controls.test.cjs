const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../web/donut_txtfusion_model_guard_controls_model.js'),'utf8').replace(/export function/g,'function');
const {addModelTxtfusionGuardControls:add}=vm.runInNewContext(source+'\n({addModelTxtfusionGuardControls})');
const plain=x=>JSON.parse(JSON.stringify(x));
function fixture(){
    const fusion={id:1118,type:'DonutKrea2FusionControl',widgets:[{name:'txtfusion_rms_guard',value:false}]};
    const panel={id:1144,title:'01 · Models',properties:{donut_panel_role:'models',donut_app_controls:{seed_path:[1138,1],groups:[
        {title:'Fusion',controls:[{path:[1014],widget:'compatibility_preset'}]},
        {title:'UncensorFix',controls:[{path:[1014,1118],widget:'uncensorfix_controls'}]},
    ]}}};
    const graph={nodes:[panel,{id:1014,type:'def',widgets:[{name:'compatibility_preset',value:'Rebalance'}]}],
        definitions:{subgraphs:[{id:'def',nodes:[fusion]}]}};
    return {graph,panel,fusion};
}
const group=p=>p.properties.donut_app_controls.groups.find(g=>g.donut_txtfusion_model_rms_guard);
test('binds actual inner Fusion Control, never outer promoted preset',()=>{
    const {graph,panel}=fixture();add(graph);
    assert.deepEqual(plain(group(panel).controls[0].path),[1014,1118]);
    assert.equal(group(panel).controls[0].widget,'txtfusion_rms_guard');
    assert.equal(group(panel).advanced,false);
});
test('model-wide description is NAG-independent and has no reference-file requirement',()=>{
    const {graph,panel}=fixture();add(graph);
    assert.match(group(panel).description,/All connected model stages/);
    assert.match(group(panel).description,/NAG on or off/);
    assert.equal(group(panel).controls.length,1);
});
test('does not copy/reset widget values or alter links',()=>{
    const {graph,panel,fusion}=fixture();fusion.widgets[0].value=true;graph.links=[[1,2,0,3,0,'MODEL']];
    const before=JSON.stringify({widgets:fusion.widgets,links:graph.links});add(graph);
    assert.equal(JSON.stringify({widgets:fusion.widgets,links:graph.links}),before);
});
test('reload is idempotent',()=>{
    const {graph}=fixture();add(graph);const saved=JSON.stringify(graph);
    assert.equal(add(graph).length,0);assert.equal(JSON.stringify(graph),saved);
    const reload=JSON.parse(saved);assert.equal(add(reload).length,0);
});
test('supports live subgraphs and string remapped ids',()=>{
    const {graph,panel,fusion}=fixture();graph.nodes[1].subgraph={_nodes:[fusion]};delete graph.definitions;
    graph.nodes[1].id='generate';fusion.id='fusion';
    const cfg=panel.properties.donut_app_controls;
    cfg.groups[0].controls[0].path=['generate'];cfg.groups[1].controls[0].path=['generate','fusion'];
    add(graph);assert.deepEqual(plain(group(panel).controls[0].path),['generate','fusion']);
});
test('does not target unrelated fusion in another family',()=>{
    const {graph,panel}=fixture();graph.nodes.push({id:77,type:'DonutKrea2FusionControl'});
    add(graph);assert.deepEqual(plain(group(panel).controls[0].path),[1014,1118]);
});
test('does not guess between two models in an ambiguous A/B panel',()=>{
    const {graph,panel}=fixture();graph.definitions.subgraphs[0].nodes.push({id:1119,type:'DonutKrea2FusionControl'});
    add(graph);assert.equal(group(panel),undefined);
});
test('only own stale group repaired, custom values untouched',()=>{
    const {graph,panel}=fixture();add(graph);group(panel).controls[0].path=['wrong'];
    panel.properties.donut_app_controls.groups.push({title:'Custom',controls:[{path:[88],widget:'anything',value:7}]});
    add(graph);assert.deepEqual(plain(group(panel).controls[0].path),[1014,1118]);
    assert.equal(panel.properties.donut_app_controls.groups.at(-1).controls[0].value,7);
});
test('custom manual guard alias is not duplicated',()=>{
    const {graph,panel}=fixture();panel.properties.donut_app_controls.groups.push({controls:[{path:[1014,1118],widget:'txtfusion_rms_guard'}]});
    add(graph);assert.equal(group(panel),undefined);
});
test('no fusion owner adds no unusable controls',()=>{
    const {graph,panel}=fixture();graph.definitions.subgraphs[0].nodes=[];add(graph);assert.equal(group(panel),undefined);
});
test('extension refresh calls existing renderer, not a second UI value store',()=>{
    const code=fs.readFileSync(path.join(__dirname,'../web/donut_txtfusion_model_guard_controls.js'),'utf8').replace(/^import.*\n/gm,'');
    const {graph,panel}=fixture();let renders=0,ext;
    panel._donutAppControls={render(){renders++}};
    const app={rootGraph:graph,registerExtension(e){ext=e}};
    vm.runInNewContext(code,{app,addModelTxtfusionGuardControls:add,queueMicrotask:fn=>fn()});
    ext.afterConfigureGraph();assert.equal(renders,1);ext.afterConfigureGraph();assert.equal(renders,1);
});

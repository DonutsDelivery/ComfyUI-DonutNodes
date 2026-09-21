const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../web/donut_txtfusion_guard_controls_model.js'), 'utf8').replace(/export (const|function)/g, '$1');
const {addTxtfusionGuardControls} = vm.runInNewContext(`${source}\n({addTxtfusionGuardControls})`);
const plain = v => JSON.parse(JSON.stringify(v));
function fixture() {
    const panel = (id, role, controls) => ({id, type:'DonutWorkflowPanel', properties:{donut_panel_role:role,
        donut_app_controls:{seed_path:[1,2], groups:[{title:'existing', controls}]}}});
    const sampler = {id:982, type:'DonutSampler', widgets:[
        {name:'txtfusion_internal_guard',value:false},
        {name:'txtfusion_reference_checkpoint',value:'None'},
    ]};
    const outer = {id:1014, type:'engine-definition', widgets:[{name:'steps',value:8}]};
    return {nodes:[panel(11,'guidance',[]), panel(12,'generate',[
        {path:[1014],widget:'steps'}, {path:[1014,982],widget:'cfg_curve'},
    ]), outer], definitions:{subgraphs:[{id:'engine-definition', nodes:[sampler]}]}};
}
function injected(graph) {
    return graph.nodes[0].properties.donut_app_controls.groups.find(g => g.donut_txtfusion_internal_guard);
}
test('V5-style promoted sampling controls bind real inner sampler widgets', () => {
    const graph = fixture(); addTxtfusionGuardControls(graph);
    const group = injected(graph);
    const sampler = graph.definitions.subgraphs[0].nodes[0];
    assert.equal(group.advanced,true);
    for (const control of group.controls) {
        assert.deepEqual(plain(control.path),[1014,982]);
        assert.ok(sampler.widgets.some(w => w.name === control.widget));
        assert.ok(!graph.nodes[2].widgets.some(w => w.name === control.widget));
    }
});
test('existing widget values links and groups are not reset', () => {
    const graph = fixture(); graph.links = [[1,2,0,3,0,'MODEL']];
    const backend = JSON.stringify(graph.definitions);
    const links = JSON.stringify(graph.links);
    addTxtfusionGuardControls(graph);
    assert.equal(JSON.stringify(graph.definitions), backend);
    assert.equal(JSON.stringify(graph.links),links);
    assert.equal(graph.nodes[0].properties.donut_app_controls.groups[0].title,'existing');
});
test('reload and repeated refresh are idempotent', () => {
    const graph=fixture(); addTxtfusionGuardControls(graph); const saved=JSON.stringify(graph);
    assert.equal(addTxtfusionGuardControls(graph).length,0);
    assert.equal(JSON.stringify(graph),saved);
    assert.equal(addTxtfusionGuardControls(JSON.parse(saved)).length,0);
});
test('live subgraphs and string IDs work', () => {
    const graph=fixture(); graph.nodes[2].subgraph=graph.definitions.subgraphs[0];
    delete graph.definitions; graph.nodes[2].subgraph.nodes[0].id='982';
    addTxtfusionGuardControls(graph); assert.ok(injected(graph));
    assert.deepEqual(plain(injected(graph).controls[0].path),[1014,'982']);
});
test('no arbitrary choice between two base samplers', () => {
    const graph=fixture(); graph.definitions.subgraphs[0].nodes.push({id:983,type:'DonutSampler'});
    assert.equal(addTxtfusionGuardControls(graph).length,0);
    assert.equal(injected(graph),undefined);
});
test('does not crosswire different families', () => {
    const graph=fixture(); graph.nodes[1].properties.donut_app_controls.seed_path=[50,60];
    assert.equal(addTxtfusionGuardControls(graph).length,0);
});
test('no owner means no invented control paths', () => {
    const graph=fixture(); graph.nodes.splice(1,1);
    assert.equal(addTxtfusionGuardControls(graph).length,0);
});
test('repairs only the tagged group when inner IDs change', () => {
    const graph=fixture(); addTxtfusionGuardControls(graph);
    graph.definitions.subgraphs[0].nodes[0].id=1999;
    graph.nodes[1].properties.donut_app_controls.groups[0].controls[1].path=[1014,1999];
    assert.equal(addTxtfusionGuardControls(graph).length,1);
    assert.deepEqual(plain(injected(graph).controls[0].path),[1014,1999]);
});
test('manual controls for the same widget/path are not duplicated', () => {
    const graph=fixture(); graph.nodes[0].properties.donut_app_controls.groups[0].controls=[
        {path:[1014,982],widget:'txtfusion_internal_guard'},
        {path:[1014,982],widget:'txtfusion_reference_checkpoint'},
    ];
    assert.equal(addTxtfusionGuardControls(graph).length,0);
});
test('extension refresh requests actual panel render after installing controls', () => {
    const text=fs.readFileSync(path.join(__dirname,'../web/donut_txtfusion_guard_controls.js'),'utf8').replace(/^import .*;\n/gm,'');
    let extension, renders=0; const queue=[]; const graph=fixture();
    graph.nodes[0]._donutAppControls={render:()=>renders++};
    vm.runInNewContext(text,{app:{rootGraph:graph,registerExtension:e=>extension=e},addTxtfusionGuardControls,
        queueMicrotask:fn=>queue.push(fn)});
    extension.afterConfigureGraph(); queue.shift()();
    assert.equal(renders,1);
    extension.afterConfigureGraph(); queue.shift()();
    assert.equal(renders,1);
});

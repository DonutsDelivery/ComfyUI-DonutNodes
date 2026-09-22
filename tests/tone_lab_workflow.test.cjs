const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname,'../web/donut_tone_lab_model.js'),'utf8').replace(/^export /gm,'');
const migrate = vm.runInNewContext(source+';addToneLabToV5');
function fixture(){
    return {last_node_id:1000,last_link_id:2000,nodes:[
        {id:10,type:'engine',order:1,outputs:[{name:'base',type:'IMAGE',links:[104]},{name:'final',type:'IMAGE',links:[101,102]}],widgets_values:['unchanged']},
        {id:20,type:'PreviewImage',pos:[100,100],inputs:[{name:'images',type:'IMAGE',link:101}],outputs:[{name:'images',type:'IMAGE',links:[103]}]},
        {id:21,type:'ImageScaleBy',inputs:[{name:'image',type:'IMAGE',link:102}],outputs:[]},
        {id:22,type:'DonutImageSave',inputs:[{name:'images',type:'IMAGE',link:103}],outputs:[],widgets_values:['my/folder',99]},
        {id:23,type:'PreviewImage',inputs:[{name:'images',type:'IMAGE',link:104}],outputs:[]},
        {id:30,type:'DonutLatestPreview',properties:{sources:{'23':'Base generation','20':'SeedVR2 / final image'},source_order:['23','20']}},
        {id:40,type:'DonutEditStudio',outputs:[{name:'edit_mode',type:'BOOLEAN',links:[]}],widgets_values:[true,'saved reference']},
        {id:50,type:'DonutWorkflowPanel',properties:{donut_panel_role:'save',donut_app_controls:{groups:[{title:'Destination',controls:[{path:[22],widget:'filename_prefix'}]}]}}},
    ],links:[[101,10,1,20,0,'IMAGE'],[102,10,1,21,0,'IMAGE'],[103,20,0,22,0,'IMAGE'],[104,10,0,23,0,'IMAGE']],
    definitions:{subgraphs:[{id:'engine',nodes:[{id:2001,type:'DonutFaceDetailer',widgets_values:['stay']}]}]},
    extra:{donut_workflow:{release:'V5'},linearData:{inputs:[[50,'workflow_controls']]}}};
}
function validate(w){
    const ns = new Map(w.nodes.map(n=>[n.id,n]));
    assert.equal(new Set(w.links.map(l=>l[0])).size,w.links.length);
    for(const [id,origin,slot,target,input] of w.links){
        assert.equal(ns.get(target).inputs[input].link,id);
        assert.ok(ns.get(origin).outputs[slot].links.includes(id));
    }
}
test('final path, both consumers, edit guard and panel bindings',()=>{
    const w=fixture(),old=structuredClone(w),r=migrate(w);assert.ok(r.changed);validate(w);
    const n=w.nodes.find(n=>n.type==='DonutToneLab');assert.equal(n.id,2002);
    assert.deepEqual(Array.from(n.widgets_values),[false,'None',1,false]);
    assert.equal(w.links.find(l=>l[0]===101)[1],n.id);assert.equal(w.links.find(l=>l[0]===102)[1],n.id);
    assert.deepEqual(w.links.find(l=>l[0]===104),old.links.find(l=>l[0]===104));
    assert.deepEqual(w.definitions,old.definitions);
    assert.deepEqual(w.nodes.find(n=>n.id===30),old.nodes.find(n=>n.id===30));
    assert.deepEqual(w.nodes.find(n=>n.id===22),old.nodes.find(n=>n.id===22));
    const g=w.nodes.find(n=>n.id===50).properties.donut_app_controls.groups[0];
    assert.deepEqual(Array.from(g.controls,c=>c.widget),['enabled','model_name','strength','apply_to_edits']);
    assert.ok(g.controls.every(c=>c.path.length===1&&c.path[0]===n.id));
    assert.equal(w.links.find(l=>l[0]===n.inputs[1].link)[1],40);
    assert.equal(w.extra.linearData.inputs[0][0],50);
});
test('save/reload is idempotent and preserves explicit choices',()=>{
    let w=fixture();migrate(w);let n=w.nodes.find(n=>n.type==='DonutToneLab');
    n.widgets_values=[true,'personal-v4.json',.65,true];n.widgets_values_named={enabled:true,model_name:'personal-v4.json',strength:.65,apply_to_edits:true};
    w=JSON.parse(JSON.stringify(w));const before=JSON.stringify(w);assert.equal(migrate(w).changed,false);assert.equal(JSON.stringify(w),before);
});
test('deleting an auto-inserted node does not re-add it',()=>{
    const w=fixture();migrate(w);w.nodes=w.nodes.filter(n=>n.type!=='DonutToneLab');assert.equal(migrate(w).reason,'already_migrated');
});
test('untagged, older, ambiguous, customized and malformed workflows are untouched',()=>{
    const changes=[w=>delete w.extra.donut_workflow,w=>w.extra.donut_workflow.release='V4 Beta',
        w=>w.nodes.push({id:51,type:'DonutEditStudio'}),w=>w.nodes.find(n=>n.id===30).properties.sources={},
        w=>w.nodes.find(n=>n.id===10).type='custom processor',w=>w.nodes.find(n=>n.id===21).type='Sampler',
        w=>w.nodes.find(n=>n.id===20).inputs[0].link=999,w=>w.links[0]={id:101},
        w=>w.nodes.find(n=>n.id===10).outputs[1].links=[],w=>w.nodes.push({id:88,type:'DonutToneLab'})];
    for(const change of changes){const w=fixture();change(w);const before=JSON.stringify(w);assert.equal(migrate(w).changed,false);assert.equal(JSON.stringify(w),before);}
});
test('numeric IDs are discovered, never hardcoded',()=>{
    const w=fixture();const map=new Map(w.nodes.map(n=>[n.id,n.id+5000]));
    for(const n of w.nodes)n.id=map.get(n.id);
    for(const l of w.links){l[1]=map.get(l[1]);l[3]=map.get(l[3]);}
    const p=w.nodes.find(n=>n.type==='DonutLatestPreview').properties;
    p.sources=Object.fromEntries(Object.entries(p.sources).map(([k,v])=>[map.get(+k),v]));
    assert.ok(migrate(w).changed);validate(w);
});
test('frontend registers the same pure migration hook',()=>{
    const text=fs.readFileSync(path.join(__dirname,'../web/donut_tone_lab.js'),'utf8').replace(/^import .*;\n/gm,'');
    let extension;vm.runInNewContext(text,{app:{registerExtension:e=>extension=e},addToneLabToV5:migrate,console:{info:()=>{}}});
    const w=fixture();extension.beforeConfigureGraph(w);assert.ok(w.nodes.some(n=>n.type==='DonutToneLab'));validate(w);
});
const packaged=path.join(__dirname,'../workflows/v5/DonutWF_v5.json');
test('shipped V5 final branch gains a disabled Tone Lab without changing internals',{skip:!fs.existsSync(packaged)},()=>{
    const w=JSON.parse(fs.readFileSync(packaged,'utf8')),old=structuredClone(w),r=migrate(w);
    assert.ok(r.changed,r.reason);validate(w);assert.deepEqual(w.definitions,old.definitions);
    const tone=w.nodes.find(n=>n.type==='DonutToneLab');assert.equal(tone.widgets_values[0],false);
    assert.equal(migrate(w).changed,false);
});

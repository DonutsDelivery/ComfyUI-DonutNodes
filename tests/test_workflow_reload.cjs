// Subgraph export/import regression tests. Fixtures are synthetic; an actual
// user PNG workflow can be supplied with DONUT_RELOAD_WORKFLOW (not committed).
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');
const clone = value => value === undefined ? undefined : JSON.parse(JSON.stringify(value));
const context = vm.createContext({console});
vm.runInContext(fs.readFileSync(path.join(__dirname,'../web/donut_workflow_repair.js'),'utf8').replace(/export function /g,'function '),context);
const repair = context.repairStreamlinedWorkflow;
const guard = context.installWorkflowSerializationGuard;
function fixture() {
    return {
        id:'root', extra:{donut_streamlining:{schema:2,required_branch:'main'}},
        nodes:[
            {id:1,type:'Source',outputs:[{name:'model',type:'MODEL',links:[1]},{name:'seed',type:'INT',links:[2]}]},
            {id:10,type:'stage',title:'My layout',pos:[300,450],size:[380,650],inputs:[
                {name:'model',type:'MODEL',link:1},{name:'seed',type:'INT',link:2,widget:{name:'seed'}}
            ],outputs:[{name:'latent',type:'LATENT',links:[3]}],widgets_values:[12,77],widgets_values_named:{steps:12,seed:77}},
            {id:2,type:'Sink',inputs:[{name:'samples',type:'LATENT',link:3}]}
        ],
        links:[[1,1,0,10,0,'MODEL'],[2,1,1,10,1,'INT'],[3,10,0,2,0,'LATENT']],
        definitions:{subgraphs:[{id:'stage',name:'User stage',inputs:[
            {id:'m',name:'model',type:'MODEL',linkIds:[101]},
            {id:'t',name:'steps',type:'INT',linkIds:[102]},
            {id:'s',name:'seed',type:'INT',linkIds:[103]}
        ],outputs:[{id:'l',name:'latent',type:'LATENT',linkIds:[104]}],nodes:[
            {id:20,type:'DonutSampler',inputs:[
                {name:'model',type:'MODEL',link:101},
                {name:'steps',type:'INT',link:102,widget:{name:'steps'}},
                {name:'seed',type:'INT',link:103,widget:{name:'seed'}}
            ],outputs:[{name:'latent',type:'LATENT',links:[104]}]}
        ],links:[
            {id:101,origin_id:-10,origin_slot:0,target_id:20,target_slot:0,type:'MODEL'},
            {id:102,origin_id:-10,origin_slot:1,target_id:20,target_slot:1,type:'INT'},
            {id:103,origin_id:-10,origin_slot:2,target_id:20,target_slot:2,type:'INT'},
            {id:104,origin_id:20,origin_slot:0,target_id:-20,target_slot:0,type:'LATENT'}
        ]}]}
    };
}
const edgeData = raw => Array.isArray(raw) ? Object.fromEntries(['id','origin_id','origin_slot','target_id','target_slot','type'].map((k,i)=>[k,raw[i]])) : raw;
function semanticLinks(workflow) {
    return [workflow,...workflow.definitions.subgraphs].flatMap(g=>g.links.map(raw=>{
        const e=edgeData(raw), source=g.nodes.find(n=>n.id===e.origin_id), target=g.nodes.find(n=>n.id===e.target_id);
        const a=e.origin_id===-10 ? g.inputs[e.origin_slot] : source.outputs[e.origin_slot];
        const b=e.target_id===-20 ? g.outputs[e.target_slot] : target.inputs[e.target_slot];
        return [g.id,e.id,e.origin_id,a.name,a.type,e.target_id,b.name,b.type];
    }));
}
test('sparse schema-2 stage ports expand without changing cable meanings or values',()=>{
    const w=fixture(), before=semanticLinks(w), old=clone(w.nodes[1]);
    assert.equal(repair(w),true);
    assert.deepEqual(clone(w.nodes[1].inputs.map(p=>p.name)),['model','steps','seed']);
    assert.equal(w.links[1][4],2); assert.deepEqual(semanticLinks(w),before);
    for(const field of ['widgets_values','widgets_values_named','pos','size','title']) assert.deepEqual(clone(w.nodes[1][field]),old[field]);
});
test('repeated JSON round trips are idempotent',()=>{
    let w=fixture(); repair(w); const expected=clone(w);
    for(let i=0;i<5;i++){w=clone(w);assert.equal(repair(w),false);assert.deepEqual(w,expected);}
});
test('renumbered node IDs do not affect port recovery',()=>{
    const w=fixture();w.nodes[1].id=719;
    for(const e of w.links){if(e[1]===10)e[1]=719;if(e[3]===10)e[3]=719;}
    assert.equal(repair(w),true);assert.equal(w.links[1][3],719);assert.equal(w.links[1][4],2);
});
test('unknown input names fail atomically rather than guessing a replacement',()=>{
    const w=fixture();w.nodes[1].inputs[1].name='unknown';const before=clone(w);
    assert.throws(()=>repair(w),/unknown input/);assert.deepEqual(w,before);
});
test('ambiguous link indices and input link claims fail atomically',()=>{
    const w=fixture();w.links[1][4]=0;const before=clone(w);
    assert.throws(()=>repair(w),/ambiguous destination/);assert.deepEqual(w,before);
});
test('missing promoted values fail rather than silently using sampler defaults',()=>{
    const w=fixture();delete w.nodes[1].widgets_values_named.steps;w.nodes[1].widgets_values=[];const before=clone(w);
    assert.throws(()=>repair(w),/missing or invalid saved value/);assert.deepEqual(w,before);
});
test('conflicting named and positional values are not silently reconciled',()=>{
    const w=fixture();w.nodes[1].widgets_values[0]=99;const before=clone(w);
    assert.throws(()=>repair(w),/conflicting saved values/);assert.deepEqual(w,before);
});
test('positional-only snapshots use recorded widget names',()=>{
    const w=fixture();delete w.nodes[1].widgets_values_named;
    w.nodes[1].properties={donut_widget_order:['steps','seed']};repair(w);
    assert.deepEqual(clone(w.nodes[1].widgets_values_named),{steps:12,seed:77});
});
test('unrelated and future-version workflows are not modified',()=>{
    for(const schema of [undefined,4]){const w=fixture();w.extra={donut_streamlining:{schema}};const before=clone(w);assert.equal(repair(w),false);assert.deepEqual(clone(w),before);}
});
test('output reordering remaps source indices by output name as well',()=>{
    const w=fixture(), stage=w.definitions.subgraphs[0], host=w.nodes[1];
    stage.outputs.unshift({id:'other',name:'unused',type:'STRING',linkIds:[]});stage.links[3].target_slot=1;
    host.outputs.push({name:'unused',type:'STRING',links:[]});const before=semanticLinks(w);repair(w);
    assert.equal(w.links[2][2],1);assert.deepEqual(semanticLinks(w),before);
});
test('nested instances repair object-form links in their own scope',()=>{
    const w=fixture(), host=clone(w.nodes[1]);
    const outer={id:'outer',name:'User wrapper',inputs:[{id:'a',name:'model',type:'MODEL',linkIds:[1]},{id:'b',name:'seed',type:'INT',linkIds:[2]}],
        outputs:[{id:'c',name:'latent',type:'LATENT',linkIds:[3]}],nodes:[host],links:[
            {id:1,origin_id:-10,origin_slot:0,target_id:10,target_slot:0,type:'MODEL'},
            {id:2,origin_id:-10,origin_slot:1,target_id:10,target_slot:1,type:'INT'},
            {id:3,origin_id:10,origin_slot:0,target_id:-20,target_slot:0,type:'LATENT'}]};
    w.nodes[1].type='outer';w.nodes[1].widgets_values=[77];w.nodes[1].widgets_values_named={seed:77};w.definitions.subgraphs.push(outer);
    const before=semanticLinks(w);repair(w);
    assert.equal(outer.links[1].target_slot,2);assert.deepEqual(semanticLinks(w),before);assert.equal(repair(w),false);
});
test('export guard repairs detached metadata without mutating live graph data',()=>{
    const live=fixture(), before=clone(live);let calls=0;
    const g={marker:42,serialize(option){assert.equal(this.marker,42);assert.equal(option,'arg');calls++;return clone(live);}};
    assert.equal(guard(g),true);assert.equal(guard(g),false);
    const saved=g.serialize('arg');assert.equal(saved.nodes[1].inputs.length,3);assert.deepEqual(live,before);assert.equal(calls,1);
});
test('schema-3 exports still repair sockets trimmed by another serializer',()=>{
    const w=fixture();w.extra.donut_streamlining.schema=3;assert.equal(repair(w),true);assert.equal(w.nodes[1].inputs.length,3);
});
test('real PNG workflow preserves layout, prompts, LoRAs and all semantic links', {skip:!process.env.DONUT_RELOAD_WORKFLOW},()=>{
    const w=JSON.parse(fs.readFileSync(process.env.DONUT_RELOAD_WORKFLOW)),before=clone(w), links=semanticLinks(w);
    assert.equal(repair(w),true);assert.deepEqual(semanticLinks(w),links);
    const all = x=>[x,...x.definitions.subgraphs].flatMap(g=>g.nodes.map(n=>[`${g.id}:${n.id}`,n]));
    const after=new Map(all(w));
    for(const [id,n] of all(before)){
        const a=after.get(id);for(const k of ['pos','size','title','mode','type','widgets_values','widgets_values_named']) assert.deepEqual(clone(a[k]),n[k],`${id} ${k}`);
    }
    const stage=w.nodes.find(n=>n.id===1014);assert.equal(stage.inputs.length,45);assert.equal(stage.widgets_values.length,29);assert.equal(repair(w),false);
});

test('connected seeds may omit their unused cache and survive JSON reload',()=>{
    const w=fixture(), n=w.nodes[1];n.widgets_values[1]=null;delete n.widgets_values_named.seed;
    const links=semanticLinks(w);repair(w);assert.deepEqual(semanticLinks(w),links);
    assert.equal(n.widgets_values[1],null);assert.equal(n.widgets_values_named.seed,null);
    assert.equal(repair(clone(w)),false);
});
test('missing connected cache does not bypass cable validation',()=>{
    const w=fixture();w.nodes[1].widgets_values[1]=null;delete w.nodes[1].widgets_values_named.seed;
    w.nodes[1].inputs[1].link=999;const before=clone(w);
    assert.throws(()=>repair(w),/ambiguous destination/);assert.deepEqual(w,before);
});
test('large Comfy integer seeds are preserved without rounding or substitution',()=>{
    const w=fixture(), n=w.nodes[1], seed=2**60;n.widgets_values[1]=seed;n.widgets_values_named.seed=seed;
    repair(w);assert.equal(n.widgets_values_named.seed,seed);assert.equal(n.widgets_values[1],seed);
    assert.equal(repair(clone(w)),false);
});
test('invalid nonmissing connected seed remains an atomic error',()=>{
    for(const value of ['bad',1.5]){const w=fixture();w.nodes[1].widgets_values[1]=value;w.nodes[1].widgets_values_named.seed=value;
    const before=clone(w);assert.throws(()=>repair(w),/invalid saved value/);assert.deepEqual(w,before);}
});

test('published workflow: every connected promoted cache can be omitted independently',()=>{
    const original=JSON.parse(fs.readFileSync(path.join(__dirname,'../workflows/v4-beta/DonutWF_v4_beta.json')));
    let checked=0;
    const graphs=w=>[w,...w.definitions.subgraphs];
    for(const [gi,g] of graphs(original).entries())for(const [ni,n] of g.nodes.entries()){
        const order=n.properties?.donut_widget_order;
        if(!order)continue;
        for(const p of n.inputs||[]){
            const index=order.indexOf(p.name);if(p.link==null||index<0)continue;
            const w=clone(original),target=graphs(w)[gi].nodes[ni];
            target.widgets_values[index]=null;delete target.widgets_values_named[p.name];
            const links=semanticLinks(w);repair(w);assert.deepEqual(semanticLinks(w),links);
            assert.equal(target.widgets_values[index],null,`${n.id}:${p.name}`);
            assert.equal(repair(clone(w)),false);checked++;
        }
    }
    assert.ok(checked>=10,`Only checked ${checked} connected caches`);
});

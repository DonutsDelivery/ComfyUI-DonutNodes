const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const source=fs.readFileSync(path.join(__dirname,'../web/donut_vae_global_controls_model.js'),'utf8').replace(/export (function|const)/g,'$1');
const {prepareVaeCorrectionMigration,vaeCorrectionMirrorWidgets}=vm.runInNewContext(source+'\n({prepareVaeCorrectionMigration,vaeCorrectionMirrorWidgets})');
const key=path=>JSON.stringify(path.map(String));
function fixture(initialized=0){
    const paths=[[1,6],[1,3],[1,4],[1,8]];
    const nodes=paths.map((_,index)=>({widgets:[
        {name:'vae_damage_correction',value:index===0},
        {name:'vae_damage_strength',value:index===0?0.49:index===1?1:index===2?0.75:1.5},
    ]}));
    const byPath=new Map(paths.map((path,index)=>[key(path),nodes[index]]));
    const panel={properties:{donut_app_controls:{vae_correction_initialized:initialized,vae_correction_global:{version:1,
        source_path:paths[0],targets:{vae_damage_correction:paths,vae_damage_strength:paths}}}}};
    return {panel,nodes,paths,resolve:path=>byPath.get(key(path))};
}
test('legacy per-stage values migrate from the saved base decode once',()=>{
    const {panel,nodes,resolve}=fixture(), plan=prepareVaeCorrectionMigration(panel,resolve);
    assert.ok(plan); assert.equal(plan.updates.length,6);
    for(const {node,widget,value} of plan.updates) widget.value=value;
    plan.config.vae_correction_initialized=1;
    for(const node of nodes) {
        assert.equal(node.widgets[0].value,true);
        assert.equal(node.widgets[1].value,0.49);
    }
    assert.equal(prepareVaeCorrectionMigration(panel,resolve),null);
});
test('saved global values survive reload without another migration',()=>{
    const {panel,nodes,resolve}=fixture(1); nodes[1].widgets[1].value=1.25;
    assert.equal(prepareVaeCorrectionMigration(panel,resolve),null);
    assert.equal(nodes[1].widgets[1].value,1.25);
});
test('an incomplete target set waits instead of marking migration complete',()=>{
    const {panel,nodes,resolve}=fixture(); delete nodes[3].widgets[1];
    assert.equal(prepareVaeCorrectionMigration(panel,resolve),null);
    assert.equal(panel.properties.donut_app_controls.vae_correction_initialized,0);
});
test('global edits mirror to each target node once and skip the source',()=>{
    const {panel,nodes,resolve}=fixture();
    const result=vaeCorrectionMirrorWidgets(panel.properties.donut_app_controls,'vae_damage_strength',nodes[0],resolve);
    assert.equal(result.length,3);
    assert.ok(result.every(({node})=>node!==nodes[0]));
    for(const {widget} of result) widget.value=0.62;
    assert.deepEqual(nodes.slice(1).map(node=>node.widgets[1].value),[0.62,0.62,0.62]);
});

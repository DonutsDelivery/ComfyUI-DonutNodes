import {test} from 'node:test';
import assert from 'node:assert/strict';
import {addSeedVR2Controls} from '../web/donut_seedvr2_controls_model.js';

function fixture() {
    const stage = id => ({id, type:'DonutTiledUpscale', widgets:[{name:'upscale_engine', value:'Donut'}]});
    const first = stage(91), second = stage(17);
    const group = (id, title) => ({title, controls:[{path:[8, id], widget:'denoise', title:'Denoise'}]});
    const panel = {id:20, properties:{donut_app_controls:{groups:[group(91, 'First upscale'), group(17, 'Second upscale')]}}};
    const links = [[1, 2, 0, 8, 0, 'IMAGE']];
    const root = {nodes:[{id:8, subgraph:{nodes:[first, second]}}, panel], links};
    return {root, panel, first, second, links};
}
test('adds controls to both existing upscale modules without touching links or settings', () => {
    const {root, panel, first, links} = fixture();
    const original = structuredClone(links);
    assert.deepEqual(addSeedVR2Controls(root), [panel]);
    const groups = panel.properties.donut_app_controls.groups;
    assert.equal(groups.length, 6);
    assert.equal(groups.filter(g => g.controls.some(c => c.widget === 'upscale_engine')).length, 2);
    assert.deepEqual(root.links, original);
    assert.equal(first.widgets[0].value, 'Donut');
});
test('idempotent after saving and reloading panel properties', () => {
    const {root, panel} = fixture();
    addSeedVR2Controls(root);
    panel.properties = structuredClone(panel.properties);
    const saved = JSON.stringify(panel.properties);
    assert.deepEqual(addSeedVR2Controls(root), []);
    assert.equal(JSON.stringify(panel.properties), saved);
});
test('does not install controls on old backend definitions or unrelated nodes', () => {
    const {root, first, second} = fixture();
    first.widgets = []; second.type = 'OtherUpscaler';
    assert.deepEqual(addSeedVR2Controls(root), []);
});
test('string IDs and remapped modules resolve without fixed workflow IDs', () => {
    const {root, panel} = fixture();
    for (const group of panel.properties.donut_app_controls.groups) group.controls[0].path = group.controls[0].path.map(String);
    assert.equal(addSeedVR2Controls(root).length, 1);
});
test('SeedVR2 options only appear when its engine is selected', () => {
    const {root, panel} = fixture(); addSeedVR2Controls(root);
    for (const group of panel.properties.donut_app_controls.groups.filter(g => g.advanced)) {
        assert.equal(group.visible_when.widget, 'upscale_engine');
        assert.equal(group.visible_when.value, 'SeedVR2');
        assert.ok(group.controls.some(c => c.widget === 'seedvr2_model_name'));
    }
});

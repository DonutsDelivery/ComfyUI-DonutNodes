const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const source = fs.readFileSync(
    path.join(__dirname, '../web/donut_nag_tau_controls_model.js'), 'utf8'
).replace(/export (const|function)/g, '$1');
const {addNagNoTauControls} = vm.runInNewContext(
    source + '\n({addNagNoTauControls})'
);

function fixture() {
    const panel = {
        id: 11,
        type: 'DonutWorkflowPanel',
        properties: {donut_app_controls: {groups: [{
            title: 'Base sampling · NAG',
            controls: [
                {path:[1014,993], widget:'nag_enabled', title:'Nag enabled'},
                {path:[1014,993], widget:'nag_phi', title:'Nag phi'},
                {path:[1014,993], widget:'nag_tau', title:'Nag tau'},
                {path:[1014,993], widget:'nag_sigma_start', title:'Nag sigma start'},
            ],
        }]}}
    };
    return {nodes:[panel]};
}

test('inserts no-tau toggle immediately after nag_tau on same owner path', () => {
    const graph = fixture();
    const changed = addNagNoTauControls(graph);
    assert.equal(changed.length, 1);
    const controls = graph.nodes[0].properties.donut_app_controls.groups[0].controls;
    const tau = controls.findIndex(c => c.widget === 'nag_tau');
    assert.equal(controls[tau + 1].widget, 'nag_disable_tau_clipping');
    assert.deepEqual([...controls[tau + 1].path], [1014,993]);
});

test('reload is idempotent and does not reset existing value-bearing graph data', () => {
    const graph = fixture();
    graph.nodes.push({id:993,type:'DonutSampler',widgets_values_named:{nag_disable_tau_clipping:true}});
    addNagNoTauControls(graph);
    const saved = JSON.stringify(graph);
    assert.equal(addNagNoTauControls(graph).length, 0);
    assert.equal(JSON.stringify(graph), saved);
    assert.equal(graph.nodes[1].widgets_values_named.nag_disable_tau_clipping, true);
});

test('does not invent a control when the panel has no nag_tau owner', () => {
    const graph = fixture();
    graph.nodes[0].properties.donut_app_controls.groups[0].controls =
        [{path:[1014,993],widget:'nag_phi'}];
    assert.equal(addNagNoTauControls(graph).length, 0);
});

test('existing manual no-tau control is not duplicated', () => {
    const graph = fixture();
    graph.nodes[0].properties.donut_app_controls.groups[0].controls.push({
        path:[1014,993], widget:'nag_disable_tau_clipping', title:'custom'
    });
    addNagNoTauControls(graph);
    const controls = graph.nodes[0].properties.donut_app_controls.groups[0].controls;
    assert.equal(controls.filter(c => c.widget === 'nag_disable_tau_clipping').length, 1);
});

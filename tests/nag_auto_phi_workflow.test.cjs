const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const workflow = () => JSON.parse(
  fs.readFileSync(path.join(__dirname, '../workflows/v5/DonutWF_v5.json'), 'utf8')
);

function objects(root, out = []) {
  if (!root || typeof root !== 'object') return out;
  if (!Array.isArray(root)) out.push(root);
  if (Array.isArray(root)) for (const item of root) objects(item, out);
  else for (const value of Object.values(root)) objects(value, out);
  return out;
}

test('V5 enables alpha-normalized auto phi for every NAG-capable stage', () => {
  const all = objects(workflow());
  const nodes = all.filter(n => n.widgets_values_named
    && Object.prototype.hasOwnProperty.call(n.widgets_values_named, 'nag_phi'));
  assert.equal(nodes.length, 4);
  for (const node of nodes) {
    assert.equal(node.widgets_values_named.nag_auto_phi, true);
    assert.equal(node.widgets_values_named.nag_phi_scale, 1);
    assert.deepEqual(node.widgets_values.slice(-2), [true, 1]);
  }
});

test('V5 NAG panel groups expose auto phi and guidance scale next to manual phi', () => {
  const groups = objects(workflow()).filter(v => Array.isArray(v.controls)
    && v.controls.some(c => c?.widget === 'nag_phi'));
  assert.equal(groups.length, 4);
  for (const group of groups) {
    const widgets = group.controls.map(c => c.widget);
    const phi = widgets.indexOf('nag_phi');
    assert.equal(group.controls[phi].title, 'Manual phi (auto off)');
    assert.equal(widgets[phi + 1], 'nag_auto_phi');
    assert.equal(widgets[phi + 2], 'nag_phi_scale');
    assert.deepEqual(group.controls[phi + 1].path, group.controls[phi].path);
    assert.deepEqual(group.controls[phi + 2].path, group.controls[phi].path);
  }
});

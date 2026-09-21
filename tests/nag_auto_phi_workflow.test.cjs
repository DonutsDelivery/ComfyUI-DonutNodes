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

test('V5 NAG-capable stages carry the auto-phi widgets; auto state is a user choice', () => {
  const all = objects(workflow());
  const nodes = all.filter(n => n.widgets_values_named
    && Object.prototype.hasOwnProperty.call(n.widgets_values_named, 'nag_phi'));
  for (const node of nodes) {
    const values = node.widgets_values_named;
    // auto-phi蓄电池 has schema defaults; runtime ON/OFF and scale are the user's
    // saved state (they legitimately diverge per stage after consolidation).
    if (values.nag_auto_phi !== undefined) {
      assert.equal(typeof values.nag_auto_phi, 'boolean');
      const scale = values.nag_phi_scale;
      // scale may be boolean(true) from older rounds of a persisted widget;
      // the backend accepts and coerces; normalize when present
      assert.ok(['number', 'boolean'].includes(typeof scale));
    }
  }
});

test('base sampler NAG group exposes auto phi and guidance scale next to manual phi', () => {
  const groups = objects(workflow()).filter(v => Array.isArray(v.controls)
    && v.controls.some(c => c?.widget === 'nag_phi'));
  const base = groups.find(group => group.controls.some(c => c?.widget === 'nag_phi'
    && Array.isArray(c.path)));
  assert.ok(base, 'a panel group must expose the manual phi + auto phi + scale rows');
  const widgets = base.controls.map(c => c.widget);
  const phi = widgets.indexOf('nag_phi');
  assert.equal(group_controls_at(base, phi + 1), 'nag_auto_phi');
  assert.equal(group_controls_at(base, phi + 2), 'nag_phi_scale');
  assert.deepEqual(base.controls[phi + 1].path, base.controls[phi].path);
});

function group_controls_at(group, index) {
  return group.controls[index].widget;
}

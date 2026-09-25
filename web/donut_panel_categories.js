import {app} from '../../scripts/app.js';
import {scheduleLayout} from './donut_layout.js?v=15';
import {addSeedVR2Controls} from './donut_seedvr2_controls_model.js';
import {organizeV4Panels, splitV4FinishingPanels, arrangeV4ByFrequency, graphEntries, upgradeV5BaseDecoders} from './donut_panel_categories_model.js?v=9';
import {PANEL_CATEGORY_CSS, syncCategorizedPanels} from './donut_panel_categories_dom.js?v=4';

const observed = new Map();
const hooked = new WeakSet();
let pending = false;
let baseDecodeAvailable = false;
function rootGraph() { return app.rootGraph; }
function synchronize() {
    syncCategorizedPanels(rootGraph());
    scheduleLayout();
}
function observePanels() {
    const roots = new Set();
    for (const {node} of graphEntries(rootGraph())) {
        if (!hooked.has(node) && (node._donutAppControls || node._donutEditStudio)) {
            hooked.add(node);
            const removed = node.onRemoved;
            node.onRemoved = function(...args) {
                try { return removed?.apply(this,args); } finally { refresh(); }
            };
        }
        for (const root of [node._donutAppControls?.root,node._donutEditStudio?.root]) {
            if (!root) continue;
            roots.add(root);
            if (observed.has(root)) continue;
            // Our own re-renders (LoRA row rebuilds, image-size label toggles)
            // mutate this subtree; each mutation used to run synchronize()
            // synchronously — a full graph walk per DOM node — and during
            // heavy panel churn Firefox content processes were crashing in
            // the compositor. Route observer work through the same batched
            // refresh so a burst of mutations collapses into one pass.
            const observer = new MutationObserver(() => refresh());
            // Catalog arrival and row reordering rebuild the original LoRA DOM.
            observer.observe(root,{childList:true,subtree:true});
            // Re-render as a macrotask, not a microtask: a native checkbox
            // fires input and change as two separate dispatches with a
            // microtask checkpoint between them. A microtask here re-rendered
            // after input but before the panel's own change handler committed,
            // writing the stale widget value back into the checkbox and
            // silently cancelling the user's toggle (Editing could never be
            // turned off). A task queued here runs after the whole
            // input → change sequence, when the widget already holds the new
            // value and render() only repaints the same state.
            const change = () => setTimeout(() => {
                for (const {node:entry} of graphEntries(rootGraph())) entry._donutEditStudio?.render();
                synchronize();
            });
            root.addEventListener('change',change); root.addEventListener('input',change);
            observed.set(root,() => {observer.disconnect(); root.removeEventListener('change',change); root.removeEventListener('input',change);});
        }
    }
    for (const [root,dispose] of observed) if (!roots.has(root)) {dispose(); observed.delete(root);}
}
function refresh() {
    if (pending) return;
    pending = true;
    queueMicrotask(() => {
        pending = false;
        const graph = rootGraph();
        if (!graph) return;
        const changed = new Set([...addSeedVR2Controls(graph),...organizeV4Panels(graph)]);
        for (const panel of changed) {panel._donutAppControls?.render(); panel.setDirtyCanvas?.(true,true);}
        observePanels(); synchronize();
    });
}
app.registerExtension({
    name:'Donut.V4PanelCategories',
    setup() {
        const style = document.createElement('style'); style.textContent = PANEL_CATEGORY_CSS; document.head.append(style);
    },
    beforeRegisterNodeDef(_nodeType, definition) {
        if (definition.name === 'DonutVAEDecode') baseDecodeAvailable = true;
    },
    beforeConfigureGraph(data) {
        // A browser refresh alone can load new JS against an old backend.
        // Keep the stock decoder until the new Python node is registered.
        if (baseDecodeAvailable) upgradeV5BaseDecoders(data);
        organizeV4Panels(data);
        splitV4FinishingPanels(data);
        arrangeV4ByFrequency(data);
    },
    afterConfigureGraph:refresh,
    nodeCreated:refresh,
    refreshComboInNodes:refresh,
});

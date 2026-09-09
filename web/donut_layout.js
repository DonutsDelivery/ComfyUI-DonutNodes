import { app } from "../../scripts/app.js";

const style = document.createElement("style");
style.textContent = `
:is(.donut-app-controls,.donut-edit-studio,.donut-reference-studio) select:not([multiple]){font-size:14px;padding:6px 10px;line-height:1.4;min-height:34px;height:34px}
:is(.donut-app-controls,.donut-edit-studio,.donut-reference-studio) select option{font:14px/1.3 system-ui}
:is(.donut-app-controls,.donut-reference-studio):not(.donut-edit-studio) input[type=checkbox]{width:18px;height:18px;flex:none}
:is(.donut-app-controls,.donut-edit-studio) input:not([type=checkbox]):not([type=range]):not([type=file]){padding:6px 8px;line-height:1.4;min-height:34px}
.donut-app-controls .donut-weight-vertical .donut-weight-value input[type=number]{padding:5px 2px;min-height:28px;font-size:12px}
.donut-reference-studio button,.donut-reference-studio input{font:inherit}
`;
document.head.append(style);
const cards = new Map();
let frame;
let resizing;
document.addEventListener("pointerdown", event => {
    const handle = event.target.closest('[role="button"][aria-label^="Resize from"]');
    const wrapper = handle?.closest('[data-node-id]');
    if (!wrapper) return;
    for (const [node, root] of cards) {
        if (String(node.id) === wrapper.dataset.nodeId && node.graph === app.rootGraph) {
            resizing = {node, root, chrome:wrapper.offsetHeight - root.offsetHeight};
            break;
        }
    }
}, true);
document.addEventListener("pointerup", () => {
    if (!resizing) return;
    const {node, root, chrome} = resizing;
    node.properties.panel_width = node.size[0];
    node.properties.panel_min_height = Math.max(0, node.size[1] - chrome);
    root.style.minHeight = `${node.properties.panel_min_height}px`;
    resizing = undefined;
    root.querySelectorAll('textarea').forEach(fitTextarea);
    scheduleLayout();
}, true);
export function fitTextarea(input) {
    if (input.hidden || !input.isConnected || !input.offsetWidth || !input.offsetHeight) return;
    input.style.height = "auto";
    input.style.height = `${Math.max(parseFloat(getComputedStyle(input).minHeight) || 0, input.scrollHeight + 2)}px`;
}
export function scheduleLayout() {
    if (frame) return;
    frame = requestAnimationFrame(() => {
        frame = undefined;
        const graph = app.rootGraph, layout = graph?.extra?.donut_layout;
        if (!layout) return;
        for (const [node, root] of cards) {
            if (node === resizing?.node || node.graph !== graph || !root.closest('.graph-canvas-container') || !root.offsetWidth || !root.offsetHeight) continue;
            const wrapper = root.closest('[data-node-id]');
            const margin = wrapper ? wrapper.offsetWidth - root.clientWidth : 24;
            const style = getComputedStyle(root);
            const padding = parseFloat(style.paddingLeft) + parseFloat(style.paddingRight);
            const banks = [...root.querySelectorAll('.donut-weight-bank')].filter(bank =>
                bank.offsetHeight && !bank.closest('details:not([open]),[hidden]'));
            const contentWidth = Math.max(0, ...banks.map(bank => bank.scrollWidth + padding));
            const width = Math.max(node.properties?.panel_width || node.size?.[0] || node.properties?.panel_min_width || 380, contentWidth + margin);
            const height = node.computeSize()[1];
            if (Math.abs(node.size[0] - width) > 2 || Math.abs(node.size[1] - height) > 2) node.setSize([width, height]);
        }
        const dimensions = node => {
            if (node.flags?.collapsed) return [node.size[0], 40];
            const wrapper = document.querySelector(`.graph-canvas-container [data-node-id="${node.id}"]`);
            const root = cards.get(node);
            if (root && (!root.offsetWidth || !root.offsetHeight)) {
                return node.properties.panel_layout_size || [node.size[0], node.size[1] + 30];
            }
            const size = [Math.max(node.size[0], wrapper?.offsetWidth || 0), Math.max(node.size[1] + 30, wrapper?.offsetHeight || 0)];
            if (root) node.properties.panel_layout_size = size;
            return size;
        };
        let x = layout.origin?.[0] || 0, top = layout.origin?.[1] || 0;
        const header = graph.getNodeById(layout.header);
        if (header) {
            if (header.pos[0] !== x || header.pos[1] !== top) header.pos = [x, top];
            top += dimensions(header)[1] + (layout.gap_y || 140);
        }
        for (const column of layout.columns) {
            let y = top, width = 0;
            for (const id of column) {
                const node = graph.getNodeById(id); if (!node) continue;
                const [w, h] = dimensions(node);
                if (Math.abs(node.pos[0] - x) > 1 || Math.abs(node.pos[1] - y) > 1) node.pos = [x, y];
                width = Math.max(width, w); y += h + (layout.gap_y || 140);
            }
            x += width + (layout.gap_x || 140);
        }
        app.canvas?.setDirty(true, true);
    });
}
export function fitModule(node, dom, root) {
    // Comfy's DOM-widget focus/click handlers select and bring the node to
    // front. Opening native controls should not change canvas selection or
    // remount/reflow the embedded panel. Headers remain selectable normally.
    dom.options.selectOn = [];
    node.properties ||= {};
    // Widget callbacks can call computeSize before our layout pass. Remember
    // the panel width independently so those callbacks cannot collapse it.
    node.properties.panel_width ||= Math.max(node.size?.[0] || 0, node.properties.panel_min_width || 380);
    const width = () => Math.max(node.properties.panel_width, node.properties.panel_min_width || 380);
    // The canvas DOM overlay prefers widget.width over node.width. Comfy's
    // selection/bring-to-front path can cache its ordinary minimum there,
    // leaving a narrow HTML panel inside a correctly sized node. These
    // full-width widgets must always follow their node, including resizes.
    Object.defineProperty(dom, "width", {
        configurable: true,
        get: () => Math.max(node.size?.[0] || 0, width()),
        set: () => {},
    });
    // LiteGraph's legacy sizing path reads only the HEIGHT returned by a
    // widget.computeSize callback. Drag-time remeasurement can therefore
    // replace a wide panel with the ordinary node minimum (about 220px),
    // even though both widget callbacks below report the intended width.
    // Preserve that width at the node boundary used by both renderers.
    const computeNodeSize = node.computeSize;
    node.computeSize = function(...args) {
        const size = computeNodeSize.apply(this, args);
        size[0] = Math.max(size[0], width());
        return size;
    };
    root.style.minHeight = `${node.properties?.panel_min_height || 0}px`;
    root.style.height = "auto"; root.style.maxHeight = "none"; root.style.overflow = "visible";
    let measuredHeight = node.properties?.panel_content_height || Math.max(100, (node.size?.[1] || 180) - 80);
    const height = () => {
        if (root.isConnected && root.offsetWidth > 0 && root.offsetHeight > 0) {
            measuredHeight = Math.ceil(root.scrollHeight);
            node.properties.panel_content_height = measuredHeight;
        }
        return node.properties?.panel_content_height || measuredHeight;
    };
    dom.computeSize = () => [width(), height()];
    dom.computeLayoutSize = () => ({minHeight:height(), maxHeight:Infinity, minWidth:width()});
    dom.options.getMinHeight = height;
    dom.options.getMaxHeight = () => Infinity;
    root.addEventListener('input', event => { if (event.target.tagName === 'TEXTAREA') fitTextarea(event.target); });
    root.addEventListener('toggle', () => { root.querySelectorAll('textarea').forEach(fitTextarea); scheduleLayout(); }, true);
    const observer = new ResizeObserver(scheduleLayout);
    const watch = () => { cards.set(node, root); observer.observe(root); scheduleLayout(); };
    const added = node.onAdded, removed = node.onRemoved;
    node.onAdded = function() { const result = added?.apply(this, arguments); watch(); return result; };
    node.onRemoved = function() { cards.delete(node); observer.disconnect(); return removed?.apply(this, arguments); };
    watch();
}

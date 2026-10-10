// DOM widgets overlay the graph canvas, so wheel events on them never reach
// the canvas element. Use the same handler as the canvas (including the user's
// zoom/pan preference), without changing the panel's controls or saved values.
export function bindPanelWheel(root, node, getCanvas) {
    const wheel = event => {
        if (event.defaultPrevented) return;
        const canvas = getCanvas();
        const container = root.closest('.graph-canvas-container');
        // Panels can also be mounted in App Mode, and removed nodes can retain
        // their DOM briefly. Only forward from the currently displayed graph.
        if (!container || !canvas?.graph || node.graph !== canvas.graph
            || !container.contains(canvas.canvas)
            || typeof canvas.processMouseWheel !== 'function') return;
        if (event.target.closest?.('dialog,[role="dialog"]')) return;

        canvas.processMouseWheel(event);
        // A newer frontend may also forward from an ancestor. Consume only
        // events handled here; leave disabled/out-of-viewport canvases alone.
        if (event.defaultPrevented) event.stopImmediatePropagation();
    };
    root.addEventListener('wheel', wheel, {capture: true, passive: false});
    return () => root.removeEventListener('wheel', wheel, true);
}

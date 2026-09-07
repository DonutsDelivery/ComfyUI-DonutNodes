import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ComfyWidgets } from "../../scripts/widgets.js";
import { createLoraService, installNativeLoras, setHidden } from "./donut_native_lora.js";
import { repairStreamlinedWorkflow, installWorkflowSerializationGuard } from "./donut_workflow_repair.js";
export { decodeRows, moveRow } from "./donut_native_lora.js";

const service = createLoraService(api);
const loaders = new Set();
const fit = node => { node.setSize?.([node.size[0], node.computeSize()[1]]); node.setDirtyCanvas?.(true, true); };
function connectedControls(node) {
    for (const widget of node.widgets || []) {
        const input = node.inputs?.find(p => p.widget?.name === widget.name || p.name === widget.name);
        if (input?.widget && input.link != null) setHidden(widget, true);
        else if (widget._donutVisible) setHidden(widget, false);
    }
}
app.registerExtension({
    name: "Donut.WorkflowStreamlining",
    beforeConfigureGraph(graphData) { repairStreamlinedWorkflow(graphData); },
    afterConfigureGraph() { installWorkflowSerializationGuard(app.graph); },
    loadedGraphNode(node) {
        if (!node.properties?.donut_stage_controls) return;
        const update = () => { connectedControls(node); fit(node); };
        const changed = node.onConnectionsChange;
        node.onConnectionsChange = function() {
            const result = changed?.apply(this, arguments); queueMicrotask(update); return result;
        };
        update();
    },
    async refreshComboInNodes() {
        // Refresh only the catalog. Saved row choices/strengths are not reset.
        if (!loaders.size) return;
        try { await service.catalog(true); } catch (error) { console.warn("[Donut LoRA]", error); }
        await Promise.all([...loaders].map(node => node._donutNativeLoras?.refresh()));
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const name = nodeData.name;
        if (!["DonutDynamicLoRAStack", "DonutLoRALoader", "DonutText", "DonutPromptConditioning", "DonutSeedPlan", "DonutModelMergeKrea2"].includes(name)) return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = created?.apply(this, arguments);
            if (["DonutDynamicLoRAStack", "DonutLoRALoader"].includes(name)) {
                installNativeLoras(this, nodeData, { app, api, service }); loaders.add(this);
                const removed = this.onRemoved, added = this.onAdded;
                this.onRemoved = function() { loaders.delete(this); return removed?.apply(this, arguments); };
                this.onAdded = function() { loaders.add(this); return added?.apply(this, arguments); };
            }
            if (["DonutText", "DonutPromptConditioning"].includes(name)) {
                const preview = ComfyWidgets.STRING(this, "resolved_text", ["STRING", { multiline: true }], app).widget;
                preview.options ||= {}; preview.options.serialize = false; preview.serialize = false;
                if (preview.inputEl) preview.inputEl.readOnly = true;
                setHidden(preview, true);
                const button = this.addWidget("button", "Show / hide resolved text", null, () => { setHidden(preview, !preview.hidden); fit(this); }, { serialize: false });
                button.serialize = false;
                const executed = this.onExecuted;
                this.onExecuted = function(message) { executed?.apply(this, arguments); preview.value = (message.text || []).join("\n"); };
                if (name === "DonutText") {
                    let advanced = false;
                    const update = () => {
                        const linked = this.inputs?.some(p => p.name === "seed" && p.link != null);
                        for (const widget of this.widgets) {
                            if (["max_depth", "missing", "separator"].includes(widget.name)) setHidden(widget, !advanced);
                            if (["seed", "control_after_generate"].includes(widget.name)) setHidden(widget, !!linked);
                        }
                        fit(this);
                    };
                    const settings = this.addWidget("button", "Show / hide text settings", null, () => { advanced = !advanced; update(); }, { serialize: false });
                    settings.serialize = false;
                    const configure = this.onConfigure, changed = this.onConnectionsChange;
                    this.onConfigure = function() { const r = configure?.apply(this, arguments); update(); return r; };
                    this.onConnectionsChange = function() { const r = changed?.apply(this, arguments); queueMicrotask(update); return r; };
                    update();
                }
            }
            if (name === "DonutSeedPlan") {
                for (const seedName of ["text_seed", "sampler_seed", "filename_seed"]) {
                    const index = this.widgets.findIndex(w => w.name === seedName);
                    const control = this.widgets[index + 1];
                    if (index >= 0 && control?.name?.includes("control_after_generate")) control.name = `${seedName}_control`;
                }
            }
            if (name === "DonutModelMergeKrea2") {
                const mode = this.widgets.find(w => w.name === "ratio_mode");
                const update = () => {
                    if (!mode) return;
                    for (const widget of this.widgets) {
                        if (widget.name === "first." || widget.name === "last." || widget.name.startsWith("blocks.") || widget.name.startsWith("txtfusion.")) setHidden(widget, mode.value === "Grouped");
                    }
                    fit(this);
                };
                if (mode) { const cb = mode.callback; mode.callback = function() { cb?.apply(this, arguments); update(); }; }
                const configure = this.onConfigure;
                this.onConfigure = function() { const r = configure?.apply(this, arguments); update(); return r; };
                update();
            }
            return result;
        };
    },
});

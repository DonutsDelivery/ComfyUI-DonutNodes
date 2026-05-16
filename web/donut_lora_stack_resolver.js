import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// DonutLoRAStack resolver extension:
//   - Hides the lora_hash_N widgets and auto-populates them when a LoRA
//     is picked, so the workflow file carries hashes for cross-machine
//     resolution.
//   - Shows toast notifications for resolver progress (find / download).

const HASH_WIDGETS = ["lora_hash_1", "lora_hash_2", "lora_hash_3"];
const NAME_WIDGETS = ["lora_name_1", "lora_name_2", "lora_name_3"];

function hideWidget(widget) {
    if (!widget) return;
    widget.type = "hidden";
    widget.computeSize = () => [0, -4];
    widget.draw = () => {};
    widget.hidden = true;
}

function toast(severity, summary, detail) {
    try {
        if (app.extensionManager?.toast?.add) {
            app.extensionManager.toast.add({
                severity,
                summary,
                detail: detail || "",
                life: severity === "error" ? 8000 : 4000,
            });
            return;
        }
    } catch (e) { /* fall through */ }
    const tag = `[DonutLoRA] ${summary}${detail ? " — " + detail : ""}`;
    if (severity === "error") console.error(tag);
    else console.log(tag);
}

async function fetchAndStoreHash(node, slot) {
    const nameWidget = node.widgets?.find(w => w.name === NAME_WIDGETS[slot]);
    const hashWidget = node.widgets?.find(w => w.name === HASH_WIDGETS[slot]);
    if (!nameWidget || !hashWidget) return;
    const name = nameWidget.value;
    if (!name || name === "None") {
        hashWidget.value = "";
        return;
    }
    try {
        const url = `/donut/lora/get_hash?name=${encodeURIComponent(name)}`;
        const r = await api.fetchApi(url);
        if (!r.ok) return;
        const data = await r.json();
        if (data.hash) {
            hashWidget.value = data.hash;
        } else {
            hashWidget.value = "";
        }
    } catch (e) {
        console.warn("[DonutLoRA] get_hash failed:", e);
    }
}

app.registerExtension({
    name: "donut.LoRAStackResolver",

    async setup() {
        api.addEventListener("donut-lora-resolver", (event) => {
            const d = event.detail || {};
            const name = d.name || d.filename || "";
            switch (d.status) {
                case "found_local":
                    toast("info", `LoRA located: ${name}`, d.message);
                    break;
                case "looking_up":
                    toast("info", `Looking up on Civitai: ${name}`);
                    break;
                case "downloading":
                    toast("info", `Downloading ${name}`, d.size_kb ? `${Math.round(d.size_kb / 1024)} MB` : "");
                    break;
                case "progress":
                    // Skip toasts for progress ticks — too noisy. Log instead.
                    if (d.percent % 10 === 0) {
                        console.log(`[DonutLoRA] ${name}: ${d.percent}%`);
                    }
                    break;
                case "complete":
                    toast("success", `Downloaded ${name}`);
                    break;
                case "not_found":
                    toast("warn", `Not on Civitai: ${name}`);
                    break;
                case "missing":
                    toast("error", `Missing LoRA: ${name}`, d.message);
                    break;
                case "error":
                    toast("error", `Resolver error: ${name}`, d.message);
                    break;
            }
        });
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "DonutLoRAStack") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const ret = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            for (const wname of HASH_WIDGETS) {
                hideWidget(this.widgets?.find(w => w.name === wname));
            }

            for (let slot = 0; slot < NAME_WIDGETS.length; slot++) {
                const nameWidget = this.widgets?.find(w => w.name === NAME_WIDGETS[slot]);
                if (!nameWidget) continue;
                const prev = nameWidget.callback;
                const node = this;
                nameWidget.callback = function(value) {
                    if (prev) prev.call(this, value);
                    fetchAndStoreHash(node, slot);
                };
            }

            return ret;
        };

        // When a workflow is loaded, populate any missing hashes for already-set names.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(info) {
            if (onConfigure) onConfigure.apply(this, arguments);
            for (let slot = 0; slot < NAME_WIDGETS.length; slot++) {
                const hashWidget = this.widgets?.find(w => w.name === HASH_WIDGETS[slot]);
                const nameWidget = this.widgets?.find(w => w.name === NAME_WIDGETS[slot]);
                hideWidget(hashWidget);
                if (hashWidget && (!hashWidget.value) && nameWidget?.value && nameWidget.value !== "None") {
                    fetchAndStoreHash(this, slot);
                }
            }
        };
    },
});

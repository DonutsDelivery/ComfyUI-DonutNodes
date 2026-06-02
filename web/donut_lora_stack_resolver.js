import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// DonutLoRAStack resolver extension.
//
// Stores per-slot SHA256 hashes in node.properties.lora_hashes so the
// workflow file carries them for cross-machine resolution. Listens for
// donut-lora-resolver events from the Python side to show toasts when a
// LoRA is auto-located or downloaded.

const NAME_WIDGETS = ["lora_name_1", "lora_name_2", "lora_name_3"];

function ensureHashStore(node) {
    if (!node.properties) node.properties = {};
    if (!Array.isArray(node.properties.lora_hashes)) {
        node.properties.lora_hashes = ["", "", ""];
    }
    while (node.properties.lora_hashes.length < NAME_WIDGETS.length) {
        node.properties.lora_hashes.push("");
    }
    return node.properties.lora_hashes;
}

function setStoredHash(node, slot, hash) {
    const hashes = ensureHashStore(node);
    hashes[slot] = hash || "";
}

function isHash(value) {
    return typeof value === "string" && /^[0-9a-fA-F]{10,128}$/.test(value.trim());
}

function setWidgetValue(node, name, value) {
    const widget = node.widgets?.find(w => w.name === name);
    if (widget) widget.value = value;
}

function migrateHashWidgetValues(node, info) {
    const values = info?.widgets_values;
    if (!Array.isArray(values) || values.length < 26) return;

    const hashPositions = [8, 16, 24];
    if (!hashPositions.some(pos => isHash(values[pos]))) return;

    const hashes = ensureHashStore(node);
    for (let slot = 0; slot < hashPositions.length; slot++) {
        const hash = values[hashPositions[slot]];
        if (isHash(hash)) hashes[slot] = hash.trim().toUpperCase();
    }

    setWidgetValue(node, "model_type", values[0]);
    setWidgetValue(node, "civitai_lookup", values[1]);

    for (let slot = 0; slot < 3; slot++) {
        const base = 2 + slot * 8;
        const n = slot + 1;
        setWidgetValue(node, `switch_${n}`, values[base]);
        setWidgetValue(node, `lora_name_${n}`, values[base + 1]);
        setWidgetValue(node, `model_weight_${n}`, values[base + 2]);
        setWidgetValue(node, `clip_weight_${n}`, values[base + 3]);
        setWidgetValue(node, `block_preset_${n}`, values[base + 4]);
        setWidgetValue(node, `block_vector_${n}`, values[base + 5]);
    }

    node.setDirtyCanvas?.(true, true);
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
    if (!nameWidget) return;
    ensureHashStore(node);
    const name = nameWidget.value;
    if (!name || name === "None") {
        setStoredHash(node, slot, "");
        return;
    }
    try {
        const r = await api.fetchApi(`/donut/lora/get_hash?name=${encodeURIComponent(name)}`);
        if (!r.ok) return;
        const data = await r.json();
        if (data.hash) {
            setStoredHash(node, slot, data.hash);
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
            const node = this;
            ensureHashStore(node);

            for (let slot = 0; slot < NAME_WIDGETS.length; slot++) {
                const nameWidget = this.widgets?.find(w => w.name === NAME_WIDGETS[slot]);
                if (!nameWidget) continue;
                const prev = nameWidget.callback;
                nameWidget.callback = function(value) {
                    if (prev) prev.call(this, value);
                    fetchAndStoreHash(node, slot);
                };
            }

            return ret;
        };

        // When a workflow loads, populate any missing hashes for already-set names.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(info) {
            if (onConfigure) onConfigure.apply(this, arguments);
            ensureHashStore(this);
            migrateHashWidgetValues(this, info);
            for (let slot = 0; slot < NAME_WIDGETS.length; slot++) {
                const nameWidget = this.widgets?.find(w => w.name === NAME_WIDGETS[slot]);
                if (nameWidget?.value && nameWidget.value !== "None"
                    && !this.properties.lora_hashes[slot]) {
                    fetchAndStoreHash(this, slot);
                }
            }
        };
    },
});

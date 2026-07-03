import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// Donut LoRA Info Display Extension
// Shows CivitAI metadata directly in the node when a LoRA is selected

// Block weight presets - uses "NAME:vector" format like Inspire
// The vector part after ':' is extracted and used

// ---------------------------------------------------------------------------
// LoRA composition display: shows which components (UNet / text encoders) a
// LoRA file contains and which blocks/layers actually hold weights.
// Data comes from /donut/loras/analyze which reads only the safetensors header.

// Expected block counts per model_type, used to size the strips so missing
// trailing blocks are visible (e.g. a KREA2 LoRA trained on blocks 14-25).
const ARCH_GROUP_COUNTS = {
    SDXL: { input_blocks: 9, middle_block: 1, output_blocks: 9 },
    SD15: { input_blocks: 12, middle_block: 1, output_blocks: 12 },
    FLUX: { double_blocks: 19, single_blocks: 38 },
    ZIT: { layers: 30 },
    "ZIT-NE": { layers: 30 },
    KREA2: { blocks: 28 },
};

const GROUP_SHORT_LABELS = {
    input_blocks: "IN",
    middle_block: "MID",
    output_blocks: "OUT",
    double_blocks: "DBL",
    single_blocks: "SGL",
    transformer_blocks: "BLK",
    blocks: "BLK",
    layers: "LYR",
    "txtfusion.layerwise": "TXT-fuse",
    "txtfusion.refiner": "TXT-refine",
    noise_refiner: "N-REF",
    context_refiner: "C-REF",
};

// Text-conditioning components that live inside the diffusion transformer
// (Krea 2 txtfusion / text MLPs). These are NOT a separate CLIP/text encoder:
// they're driven by model_weight, and clip_weight has no effect on them.
function isTextSide(name) {
    return /^txt/.test(name);  // txtfusion.*, txtmlp_* (not tmlp_/tproj_, which are time embeds)
}

const TEXT_SIDE_TOOLTIP = "Fused text conditioning (inside the DiT) — driven by model_weight, not clip_weight";

const loraAnalysisCache = {};

async function fetchLoraAnalysis(name) {
    if (!name || name === "None") return null;
    if (!(name in loraAnalysisCache)) {
        loraAnalysisCache[name] = (async () => {
            try {
                const resp = await api.fetchApi(`/donut/loras/analyze?name=${encodeURIComponent(name)}`);
                return resp.ok ? await resp.json() : null;
            } catch (e) {
                console.log("[DonutNodes] LoRA analyze failed:", e);
                return null;
            }
        })();
    }
    return loraAnalysisCache[name];
}

function makeBlockStrip(group, expectedCount) {
    const present = new Set(group.indices);
    const maxIdx = group.indices.length ? group.indices[group.indices.length - 1] : -1;
    const total = Math.max(maxIdx + 1, expectedCount || 0);

    const strip = document.createElement("div");
    strip.style.display = "flex";
    strip.style.flexWrap = "wrap";
    strip.style.gap = "1px";
    strip.style.alignItems = "center";
    for (let i = 0; i < total; i++) {
        const cell = document.createElement("div");
        cell.style.width = "5px";
        cell.style.height = "9px";
        cell.style.borderRadius = "1px";
        cell.style.backgroundColor = present.has(i) ? "#6fbf6f" : "#3a3a4a";
        cell.title = `${group.name} ${i}: ${present.has(i) ? "has weights" : "empty"}`;
        strip.appendChild(cell);
    }
    return { strip, presentCount: group.indices.length, total };
}

function renderLoraComposition(div, data, modelType) {
    div.innerHTML = "";
    if (!data) {
        div.style.display = "none";
        return;
    }
    div.style.display = "block";

    const note = (text, tooltip) => {
        const el = document.createElement("div");
        el.style.color = "#888";
        el.textContent = text;
        if (tooltip) el.title = tooltip;
        div.appendChild(el);
    };

    if (data.found === false) {
        note("file not found");
        return;
    }
    if (!data.supported) {
        note(`can't inspect: ${data.error || "unsupported format"}`);
        return;
    }
    const comps = data.components || [];
    if (!comps.length) {
        note("no LoRA modules found");
        return;
    }

    // Headline: which components the file contains
    const hasUnet = comps.some(c => c.type === "unet");
    const hasTe = comps.some(c => c.type === "te");
    // Text conditioning fused into the diffusion model (Krea 2 txtfusion / txtmlp).
    const hasFusedText = comps.some(c => c.type === "unet" && (
        c.groups.some(g => isTextSide(g.name)) ||
        (c.ungrouped_names || []).some(isTextSide)
    ));
    let headline = comps.map(c => c.name).join(" + ");
    if (hasUnet && !hasTe) {
        headline += hasFusedText
            ? " + fused text (no separate CLIP/TE)"
            : " only (no CLIP/TE)";
    } else if (hasTe && !hasUnet) {
        headline += " only (no UNet)";
    }
    const header = document.createElement("div");
    header.style.color = "#9ecbff";
    header.style.marginBottom = "2px";
    header.textContent = headline;
    if (hasFusedText && !hasTe) header.title = TEXT_SIDE_TOOLTIP;
    div.appendChild(header);

    // A LoRA with no separate CLIP/TE has nothing for clip_weight to apply to —
    // it's a genuine no-op. Say so, since the widget still accepts a value.
    if (hasUnet && !hasTe) {
        const warn = document.createElement("div");
        warn.style.color = "#e0a04a";
        warn.style.marginBottom = "2px";
        warn.textContent = "clip_weight has no effect — use model_weight";
        warn.title = "This LoRA contains no separate CLIP/text-encoder weights, "
            + "so clip_weight does nothing. Adjust model_weight (and the block vector) instead.";
        div.appendChild(warn);
    }

    const expected = ARCH_GROUP_COUNTS[modelType] || {};
    for (const comp of comps) {
        if (!comp.groups.length) {
            // No numbered blocks — name what it actually targets instead.
            const names = comp.ungrouped_names || [];
            if (names.length) {
                const shown = names.join(", ");
                const more = comp.ungrouped > names.length ? ` +${comp.ungrouped - names.length} more` : "";
                const tip = names.some(isTextSide) ? TEXT_SIDE_TOOLTIP : undefined;
                note(`${comp.name} targets: ${shown}${more}`, tip);
            } else {
                note(`${comp.name}: ${comp.modules} module(s), no per-block info`);
            }
            continue;
        }
        for (const group of comp.groups) {
            const row = document.createElement("div");
            row.style.display = "flex";
            row.style.gap = "4px";
            row.style.alignItems = "center";
            row.style.marginBottom = "1px";

            const textSide = comp.type === "unet" && isTextSide(group.name);
            const label = document.createElement("span");
            label.style.color = textSide ? "#c9a6ff" : "#aaa";
            label.style.whiteSpace = "nowrap";
            label.style.minWidth = "52px";
            label.textContent = comp.type === "te"
                ? comp.name
                : (GROUP_SHORT_LABELS[group.name] || group.name);
            label.title = textSide ? TEXT_SIDE_TOOLTIP : `${comp.name} ${group.name}`;
            row.appendChild(label);

            const expectedCount = comp.type === "unet" ? expected[group.name] : 0;
            const { strip, presentCount, total } = makeBlockStrip(group, expectedCount);
            row.appendChild(strip);

            const count = document.createElement("span");
            count.style.color = "#888";
            count.style.whiteSpace = "nowrap";
            count.textContent = `${presentCount}/${total}`;
            row.appendChild(count);

            div.appendChild(row);
        }
        if (comp.ungrouped > 0) {
            const names = comp.ungrouped_names || [];
            const detail = names.length
                ? `: ${names.join(", ")}${comp.ungrouped > names.length ? " …" : ""}`
                : "";
            note(`+${comp.ungrouped} base/other module(s) (vector pos 0)${detail}`);
        }
    }
}

app.registerExtension({
    name: "donut.LoRAInfoDisplay",

    // Register settings in ComfyUI settings panel
    async setup() {
        // CivitAI API Key setting
        app.ui.settings.addSetting({
            id: "DonutNodes.CivitAI.ApiKey",
            name: "[DonutNodes] CivitAI API Key",
            type: "text",
            defaultValue: "",
            tooltip: "Your CivitAI API key for fetching LoRA metadata. Get one from: https://civitai.com/user/account -> API Keys",
            attrs: {
                style: {
                    fontFamily: "monospace",
                },
                type: "password",  // Hide the API key
            },
            async onChange(value) {
                // Save to server config
                try {
                    await api.fetchApi("/donut/config/civitai_api_key", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ api_key: value }),
                    });
                    console.log("[DonutNodes] CivitAI API key saved");
                } catch (error) {
                    console.error("[DonutNodes] Error saving API key:", error);
                }
            },
        });

        // Auto-lookup toggle
        app.ui.settings.addSetting({
            id: "DonutNodes.CivitAI.AutoLookup",
            name: "[DonutNodes] Auto-fetch LoRA info from CivitAI",
            type: "boolean",
            defaultValue: true,
            tooltip: "Automatically fetch LoRA metadata from CivitAI when selected",
            async onChange(value) {
                try {
                    await api.fetchApi("/donut/config/civitai_auto_lookup", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ auto_lookup: value }),
                    });
                } catch (error) {
                    console.error("[DonutNodes] Error saving setting:", error);
                }
            },
        });

        // Download previews toggle
        app.ui.settings.addSetting({
            id: "DonutNodes.CivitAI.DownloadPreviews",
            name: "[DonutNodes] Download LoRA preview images",
            type: "boolean",
            defaultValue: true,
            tooltip: "Download and cache preview images from CivitAI",
            async onChange(value) {
                try {
                    await api.fetchApi("/donut/config/civitai_download_previews", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ download_previews: value }),
                    });
                } catch (error) {
                    console.error("[DonutNodes] Error saving setting:", error);
                }
            },
        });

        // Show/hide CivitAI browser button
        app.ui.settings.addSetting({
            id: "DonutNodes.CivitAI.ShowBrowserButton",
            name: "[DonutNodes] Show CivitAI Browser button",
            type: "boolean",
            defaultValue: true,
            tooltip: "Show the floating CivitAI browser button in the UI",
            onChange(value) {
                // Update button visibility immediately
                const btn = document.getElementById("donut-civitai-btn");
                if (btn) {
                    btn.style.display = value ? "block" : "none";
                }
            },
        });

        // Load current settings from server
        try {
            const response = await api.fetchApi("/donut/config");
            if (response.ok) {
                const config = await response.json();
                // Settings will be loaded by ComfyUI from localStorage,
                // but we sync with server config on startup
                if (config.civitai?.api_key) {
                    app.ui.settings.setSettingValue("DonutNodes.CivitAI.ApiKey", config.civitai.api_key);
                }
            }
        } catch (error) {
            console.log("[DonutNodes] Could not load server config:", error);
        }
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Hook into DonutLoRAStack node
        if (nodeData.name === "DonutLoRAStack") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function() {
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }

                // Store all presets for filtering
                const node = this;
                const allPresets = {};
                for (let i = 1; i <= 3; i++) {
                    const presetWidget = this.widgets?.find(w => w.name === `block_preset_${i}`);
                    if (presetWidget && presetWidget.options?.values) {
                        allPresets[i] = [...presetWidget.options.values];
                    }
                }

                // Function to filter presets by model type
                const filterPresets = (modelType) => {
                    for (let i = 1; i <= 3; i++) {
                        const presetWidget = node.widgets?.find(w => w.name === `block_preset_${i}`);
                        if (presetWidget && allPresets[i]) {
                            let filtered;
                            if (modelType === "Auto") {
                                filtered = allPresets[i];
                            } else {
                                // Filter to only show presets matching the model type prefix
                                filtered = allPresets[i].filter(p =>
                                    p === "None" || p.startsWith(modelType + "-")
                                );
                            }
                            presetWidget.options.values = filtered;

                            // Reset to None if current value is not in filtered list
                            if (!filtered.includes(presetWidget.value)) {
                                presetWidget.value = "None";
                            }
                        }
                    }
                    node.setDirtyCanvas(true);
                };

                // Add callback to model_type widget
                const modelTypeWidget = this.widgets?.find(w => w.name === "model_type");
                if (modelTypeWidget) {
                    const originalModelTypeCallback = modelTypeWidget.callback;
                    modelTypeWidget.callback = function(value) {
                        if (originalModelTypeCallback) {
                            originalModelTypeCallback.call(this, value);
                        }
                        filterPresets(value);
                        // Strip sizing depends on model type
                        if (node._donutUpdateComposition) {
                            for (let i = 1; i <= 3; i++) node._donutUpdateComposition(i);
                        }
                    };
                    // Apply initial filter
                    filterPresets(modelTypeWidget.value || "Auto");
                }

                // Store filterPresets function on node for use in onConfigure
                this._donutFilterPresets = filterPresets;

                // Find the block_preset widgets and add callbacks to update corresponding block_vector
                for (let i = 1; i <= 3; i++) {
                    const presetWidget = this.widgets?.find(w => w.name === `block_preset_${i}`);
                    if (presetWidget) {
                        const originalCallback = presetWidget.callback;
                        const slotNum = i;

                        presetWidget.callback = function(value) {
                            // Call original callback if exists
                            if (originalCallback) {
                                originalCallback.call(this, value);
                            }

                            // Extract the vector part after ':' (Inspire-style format)
                            // Format is "NAME:vector" e.g. "SDXL-ALL:1,1,1,1,1,1,1,1,1,1,1,1,1"
                            if (value && value !== "None" && value.includes(":")) {
                                const presetVector = value.split(":")[1];
                                const vectorWidget = node.widgets?.find(w => w.name === `block_vector_${slotNum}`);
                                if (vectorWidget) {
                                    vectorWidget.value = presetVector;
                                }
                                node.setDirtyCanvas(true);
                            }
                        };
                    }
                }

                // Create combined info+image widgets for each LoRA slot (side by side)
                this.infoImageWidgets = [];

                for (let i = 1; i <= 3; i++) {
                    // Outer column: composition strip on top, info+image row below
                    const outer = document.createElement("div");
                    outer.style.display = "flex";
                    outer.style.flexDirection = "column";
                    outer.style.gap = "4px";

                    // Composition display (UNet/CLIP components + populated blocks)
                    const compDiv = document.createElement("div");
                    compDiv.style.display = "none";
                    compDiv.style.backgroundColor = "#1a1a2e";
                    compDiv.style.borderRadius = "4px";
                    compDiv.style.padding = "4px 6px";
                    compDiv.style.fontSize = "9px";
                    compDiv.style.fontFamily = "monospace";
                    compDiv.style.maxHeight = "58px";
                    compDiv.style.overflowY = "auto";
                    outer.appendChild(compDiv);

                    // Create container div with flexbox for side-by-side layout
                    const container = document.createElement("div");
                    container.style.display = "flex";
                    container.style.flexDirection = "row";
                    container.style.gap = "8px";
                    container.style.backgroundColor = "#1a1a2e";
                    container.style.borderRadius = "4px";
                    container.style.padding = "6px";
                    container.style.minHeight = "120px";

                    // Text info side
                    const textDiv = document.createElement("div");
                    textDiv.style.flex = "1";
                    textDiv.style.fontSize = "10px";
                    textDiv.style.fontFamily = "monospace";
                    textDiv.style.color = "#eee";
                    textDiv.style.whiteSpace = "pre-wrap";
                    textDiv.style.overflow = "hidden";
                    textDiv.style.padding = "4px";
                    textDiv.textContent = `LoRA ${i} info...`;
                    container.appendChild(textDiv);

                    // Image side
                    const imgDiv = document.createElement("div");
                    imgDiv.style.width = "120px";
                    imgDiv.style.minWidth = "120px";
                    imgDiv.style.display = "flex";
                    imgDiv.style.justifyContent = "center";
                    imgDiv.style.alignItems = "center";
                    imgDiv.style.overflow = "hidden";
                    container.appendChild(imgDiv);
                    outer.appendChild(container);

                    // Add as single DOM widget
                    const widget = this.addDOMWidget(`civitai_panel_${i}`, "div", outer, {
                        serialize: false,
                    });
                    widget.computeSize = () => [this.size[0] - 20, 196];

                    // Store references for updating later
                    this.infoImageWidgets.push({
                        widget: widget,
                        textDiv: textDiv,
                        imgDiv: imgDiv,
                        compDiv: compDiv
                    });
                }

                // Fetch + render LoRA composition for a slot
                const updateComposition = async (slot) => {
                    const nameWidget = node.widgets?.find(w => w.name === `lora_name_${slot}`);
                    const entry = node.infoImageWidgets?.[slot - 1];
                    if (!nameWidget || !entry) return;
                    const name = nameWidget.value;
                    const data = await fetchLoraAnalysis(name);
                    if (nameWidget.value !== name) return; // stale response
                    const modelType = node.widgets?.find(w => w.name === "model_type")?.value || "Auto";
                    renderLoraComposition(entry.compDiv, data, modelType);
                    node.setDirtyCanvas(true);
                };
                this._donutUpdateComposition = updateComposition;

                // Update composition when a LoRA is selected
                for (let i = 1; i <= 3; i++) {
                    const nameWidget = this.widgets?.find(w => w.name === `lora_name_${i}`);
                    if (nameWidget) {
                        const originalCallback = nameWidget.callback;
                        const slotNum = i;
                        nameWidget.callback = function(value) {
                            if (originalCallback) {
                                originalCallback.call(this, value);
                            }
                            updateComposition(slotNum);
                        };
                    }
                }

                // Reorder widgets so each LoRA's config is followed by its info panel
                const reorderedWidgets = [];
                const widgetsByName = {};

                for (const w of this.widgets) {
                    widgetsByName[w.name] = w;
                }

                // Add model_type and civitai_lookup first
                if (widgetsByName["model_type"]) reorderedWidgets.push(widgetsByName["model_type"]);
                if (widgetsByName["civitai_lookup"]) reorderedWidgets.push(widgetsByName["civitai_lookup"]);

                // Add each LoRA group: config widgets, then combined info+image panel
                for (let i = 1; i <= 3; i++) {
                    const loraWidgets = [
                        `switch_${i}`,
                        `lora_name_${i}`,
                        `model_weight_${i}`,
                        `clip_weight_${i}`,
                        `block_preset_${i}`,
                        `block_vector_${i}`,
                        `civitai_panel_${i}`
                    ];
                    for (const name of loraWidgets) {
                        if (widgetsByName[name]) {
                            reorderedWidgets.push(widgetsByName[name]);
                        }
                    }
                }

                // Replace widgets array
                this.widgets = reorderedWidgets;

                // Adjust node size
                this.setSize([this.size[0], this.size[1] + 620]);
            };

            // Apply preset filter when node is configured (loaded from workflow)
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(info) {
                if (onConfigure) {
                    onConfigure.apply(this, arguments);
                }

                // After workflow values are loaded, apply the preset filter
                // based on the loaded model_type value
                const modelTypeWidget = this.widgets?.find(w => w.name === "model_type");
                if (modelTypeWidget && this._donutFilterPresets) {
                    this._donutFilterPresets(modelTypeWidget.value || "Auto");
                }

                // Show composition for LoRAs loaded from the workflow
                if (this._donutUpdateComposition) {
                    for (let i = 1; i <= 3; i++) this._donutUpdateComposition(i);
                }
            };

            // Update display when node executes
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                // Update combined info+image panels
                // message.text contains [lora_info, trigger_words, urls, info_1, info_2, info_3]
                if (this.infoImageWidgets) {
                    // Update text info for each panel
                    if (message.text) {
                        const info1 = message.text[3] || "";
                        const info2 = message.text[4] || "";
                        const info3 = message.text[5] || "";

                        this.infoImageWidgets[0].textDiv.textContent = info1 || "LoRA 1: Off or None";
                        this.infoImageWidgets[1].textDiv.textContent = info2 || "LoRA 2: Off or None";
                        this.infoImageWidgets[2].textDiv.textContent = info3 || "LoRA 3: Off or None";
                    }

                    // Update images for each panel
                    if (message.images) {
                        // message.images is [slot0, slot1, slot2] - may contain null for empty slots
                        for (let i = 0; i < 3; i++) {
                            // Clear the image div
                            this.infoImageWidgets[i].imgDiv.innerHTML = "";

                            const imgData = message.images[i];
                            if (imgData && imgData.filename) {
                                const img = document.createElement("img");
                                img.src = api.apiURL(`/view?filename=${encodeURIComponent(imgData.filename)}&subfolder=${encodeURIComponent(imgData.subfolder || "")}&type=${imgData.type || "temp"}`);
                                img.style.maxWidth = "100%";
                                img.style.maxHeight = "120px";
                                img.style.objectFit = "contain";
                                img.style.borderRadius = "4px";
                                this.infoImageWidgets[i].imgDiv.appendChild(img);
                            } else {
                                // Show placeholder for empty slot
                                const placeholder = document.createElement("div");
                                placeholder.style.color = "#666";
                                placeholder.style.fontSize = "10px";
                                placeholder.style.textAlign = "center";
                                placeholder.textContent = "No preview";
                                this.infoImageWidgets[i].imgDiv.appendChild(placeholder);
                            }
                        }

                        // Remove images from message so default handler doesn't show them again
                        delete message.images;
                    }
                }

                // Call original handler (for any other functionality)
                if (onExecuted) {
                    onExecuted.apply(this, arguments);
                }

                // Trigger resize
                this.setDirtyCanvas(true);
            };
        }

        // Hook into DonutLoRACivitAIInfo node for detailed info display
        if (nodeData.name === "DonutLoRACivitAIInfo") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function() {
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }

                // Create info display
                const infoWidget = ComfyWidgets["STRING"](
                    this,
                    "info_display",
                    ["STRING", { multiline: true }],
                    app
                ).widget;

                infoWidget.inputEl.readOnly = true;
                infoWidget.inputEl.style.fontSize = "11px";
                infoWidget.inputEl.style.fontFamily = "monospace";
                infoWidget.inputEl.style.backgroundColor = "#1a1a2e";
                infoWidget.inputEl.style.minHeight = "100px";
                infoWidget.inputEl.placeholder = "Select a LoRA to see CivitAI info...";
                infoWidget.serializeValue = async () => "";

                this.infoWidget = infoWidget;
                this.setSize([this.size[0], this.size[1] + 80]);
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                if (onExecuted) {
                    onExecuted.apply(this, arguments);
                }

                if (this.infoWidget && message.text) {
                    const [name, version, description, triggerWords, modelUrl, hash, recWeight] = message.text;

                    let display = "";
                    if (name) {
                        display = `Name: ${name}`;
                        if (version) display += ` (${version})`;
                        display += "\n";
                        if (recWeight && recWeight !== "1.0") {
                            display += `Recommended Weight: ${recWeight}\n`;
                        }
                        if (triggerWords) {
                            display += `Triggers: ${triggerWords}\n`;
                        }
                        if (description) {
                            display += `\n${description.substring(0, 200)}${description.length > 200 ? '...' : ''}`;
                        }
                        if (modelUrl) {
                            display += `\n\n${modelUrl}`;
                        }
                    } else {
                        display = "Select a LoRA to see info...";
                    }

                    this.infoWidget.value = display;
                }
            };
        }
    },
});

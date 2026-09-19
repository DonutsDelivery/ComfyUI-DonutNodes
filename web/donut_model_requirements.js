const LOADERS = {
    UNETLoader: {unet_name:"diffusion_models"},
    CheckpointLoaderSimple: {ckpt_name:"checkpoints"},
    CheckpointLoader: {ckpt_name:"checkpoints"},
    CLIPLoader: {clip_name:"text_encoders"},
    DualCLIPLoader: {clip_name1:"text_encoders",clip_name2:"text_encoders"},
    TripleCLIPLoader: {clip_name1:"text_encoders",clip_name2:"text_encoders",clip_name3:"text_encoders"},
    VAELoader: {vae_name:"vae"},
    UpscaleModelLoader: {model_name:"upscale_models"},
    UltralyticsDetectorProvider: {model_name:"ultralytics"},
    SAMLoader: {model_name:"sams"},
    LoadBackgroundRemovalModel: {bg_removal_name:"background_removal"},
    DonutSubjectMaskPreview: {model_name:"background_removal"},
    LoraLoader: {lora_name:"loras"},
    LoraLoaderModelOnly: {lora_name:"loras"},
    ControlNetLoader: {control_net_name:"controlnet"},
    CLIPVisionLoader: {clip_name:"clip_vision"},
    DonutEditStudio: {lora_name:"loras"},
};

const KREA2_SDA_LORA = "krea2/krea2_turbo_sda_v1.0_comfy.safetensors";

export function modelBindings(graph) {
    const bindings = [], seen = new Set();
    function add(node, widget, folder, name, replace) {
        if (typeof name !== "string" || !/\.(safetensors|sft|ckpt|pt|pth|bin|gguf|onnx)$/i.test(name)) return;
        bindings.push({folder, name:name.replaceAll("\\", "/"), update(value) {
            const updated = replace(value);
            if (updated === undefined) return;
            node.graph?.beforeChange();
            widget.value = updated; widget.callback?.(updated);
            node._donutNativeLoras?.restore();
            node.graph?.afterChange();
            node.setDirtyCanvas?.(true, true);
        }});
    }
    function visit(current) {
        if (!current || seen.has(current)) return;
        seen.add(current);
        for (const node of current.nodes || []) {
            visit(node.subgraph);
            const type = node.comfyClass || node.type;
            const selected = name => node.widgets?.find(widget => widget.name === name)?.value;
            const loaders = {...(LOADERS[type] || {})};
            if (type === "DonutSubjectMaskPreview" && selected("model_name") === "sam3.1_multiplex_fp16.safetensors") {
                loaders.model_name = "checkpoints";
            }
            // These models are loaded inside existing modules rather than separate
            // loader nodes. Download every configured feature model whether its
            // toggle is currently on or off, so one click prepares the whole workflow.
            if (type === "DonutTiledUpscale") {
                loaders.seedvr2_model_name = "diffusion_models";
                loaders.seedvr2_vae_name = "vae";
            }
            if (type === "DonutSeedVR2Upscale") {
                loaders.seedvr2_model_name = "diffusion_models";
                loaders.seedvr2_vae_name = "vae";
            }
            if (type === "DonutEditStudio") {
                loaders.mask_b_model = "background_removal";
                if (node.widgets?.some(widget => widget.name === "mask_b_prompt")) {
                    bindings.push({folder:"checkpoints", name:"sam3.1_multiplex_fp16.safetensors", update() {}});
                }
            }
            for (const [name, folder] of Object.entries(loaders)) {
                const widget = node.widgets?.find(widget => widget.name === name);
                if (!widget) continue;
                const original = widget.value;
                add(node, widget, folder, original, value => widget.value === original ? value : undefined);
            }
            if (type === "DonutSampler" && node.widgets?.some(widget => widget.name === "sda_enabled")) {
                // SDA is a fixed adapter rather than a filename widget. Its presence
                // means the workflow exposes the feature, even when currently off.
                bindings.push({folder:"loras", name:KREA2_SDA_LORA, update() {}});
            }
            if (["DonutLoRALoader", "DonutDynamicLoRAStack"].includes(type)) {
                const widget = node.widgets?.find(widget => widget.name === "slots_json");
                if (!widget) continue;
                let rows;
                try { rows = JSON.parse(widget.value); } catch { continue; }
                if (!Array.isArray(rows)) continue;
                for (const row of rows) {
                    add(node, widget, "loras", row.lora_name, value => {
                        const current = JSON.parse(widget.value);
                        const found = current.find(candidate => candidate.id === row.id && candidate.lora_name === row.lora_name);
                        if (!found) return undefined;
                        found.lora_name = value;
                        return JSON.stringify(current);
                    });
                }
            }
        }
    }
    visit(graph);
    return bindings;
}

// Pure presentation data for the Registry's links-only panel. URLs come only
// from the packaged catalog, never workflow metadata or a guessed filename.
export function manualModelFiles(bindings, catalog) {
    const seen = new Set(), result = [];
    const basename = name => name.split("/").at(-1);
    for (const {folder, name} of bindings) {
        const path = `models/${folder}/${name}`;
        if (seen.has(path)) continue;
        seen.add(path);
        const candidates = catalog.filter(entry => entry.folder === folder);
        let entry = candidates.find(entry => entry.filename === name);
        if (!entry) {
            const matches = candidates.filter(entry => basename(entry.filename) === basename(name));
            if (matches.length === 1) entry = matches[0];
        }
        // Catalog validation happens while staging; also reject unsafe schemes
        // here so a damaged catalog can never become a script link in the UI.
        let url = null;
        try {
            const parsed = new URL(entry?.url);
            if (parsed.protocol === "https:" && !parsed.username && !parsed.password) url = parsed.href;
        } catch { /* Unknown models are listed with their location, without a URL. */ }
        result.push({path, url, size:entry?.size, sha256:entry?.sha256,
            requires_nodes:entry?.requires_nodes || []});
    }
    return result;
}

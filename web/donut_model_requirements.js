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
    LoraLoader: {lora_name:"loras"},
    LoraLoaderModelOnly: {lora_name:"loras"},
    ControlNetLoader: {control_net_name:"controlnet"},
    CLIPVisionLoader: {clip_name:"clip_vision"},
    DonutEditStudio: {lora_name:"loras"},
};

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
            if (node.mode === 2 || node.mode === 4) continue;
            visit(node.subgraph);
            const type = node.comfyClass || node.type;
            for (const [name, folder] of Object.entries(LOADERS[type] || {})) {
                const widget = node.widgets?.find(widget => widget.name === name);
                if (!widget) continue;
                const original = widget.value;
                add(node, widget, folder, original, value => widget.value === original ? value : undefined);
            }
            if (["DonutLoRALoader", "DonutDynamicLoRAStack"].includes(type)) {
                const widget = node.widgets?.find(widget => widget.name === "slots_json");
                if (!widget) continue;
                let rows;
                try { rows = JSON.parse(widget.value); } catch { continue; }
                if (!Array.isArray(rows)) continue;
                for (const row of rows) {
                    if (row.enabled === false) continue;
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

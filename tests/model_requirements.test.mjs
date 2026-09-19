import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import test from "node:test";
import vm from "node:vm";

const scope = vm.createContext({});
vm.runInContext(readFileSync(new URL("../web/donut_model_requirements.js", import.meta.url), "utf8").replaceAll("export ", ""), scope);
const make = (type, values) => ({type, widgets:Object.entries(values).map(([name,value]) => ({name,value}))});

test("finds models in nested subgraphs and every configured Donut LoRA row", () => {
    const graph = {nodes:[make("VAELoader", {vae_name:"vae/test.safetensors"}), {
        subgraph:{nodes:[make("UNETLoader", {unet_name:"base.safetensors"}), make("DonutLoRALoader", {slots_json:JSON.stringify([
            {id:"1",enabled:true,lora_name:"krea2/one.safetensors"}, {id:"2",enabled:false,lora_name:"two.safetensors"},
        ])})]},
    }]};
    const bindings = scope.modelBindings(graph);
    assert.equal(bindings.length, 4);
    assert.deepEqual(Array.from(bindings, b => b.name), ["vae/test.safetensors", "base.safetensors", "krea2/one.safetensors", "two.safetensors"]);
    bindings[2].update("moved/one.safetensors");
    const rows = JSON.parse(graph.nodes[1].subgraph.nodes[1].widgets[0].value);
    assert.equal(rows[0].lora_name, "moved/one.safetensors");
    assert.equal(rows[1].lora_name, "two.safetensors");
});

test("native SDA is prepared whether it is currently enabled or disabled", () => {
    const enabled = make("DonutSampler", {sda_enabled:true});
    const disabled = make("DonutSampler", {sda_enabled:false});
    assert.deepEqual(Array.from(scope.modelBindings({nodes:[enabled]}), binding => [binding.folder, binding.name]), [
        ["loras", "krea2/krea2_turbo_sda_v1.0_comfy.safetensors"],
    ]);
    assert.deepEqual(Array.from(scope.modelBindings({nodes:[disabled]}), binding => [binding.folder, binding.name]), [
        ["loras", "krea2/krea2_turbo_sda_v1.0_comfy.safetensors"],
    ]);
});

test("ignores notes, empty selectors and non-model text but prepares bypassed loaders", () => {
    const graph = {nodes:[make("Note", {text:"https://example.com/evil.safetensors"}),
        {...make("UNETLoader", {unet_name:"skip.safetensors"}),mode:4}, make("VAELoader", {vae_name:"pixel_space"})]};
    assert.deepEqual(Array.from(scope.modelBindings(graph), binding => [binding.folder, binding.name]), [
        ["diffusion_models", "skip.safetensors"],
    ]);
});

test("finishing a download does not overwrite a newer user selection", () => {
    const node = make("UNETLoader", {unet_name:"original.safetensors"});
    const [binding] = scope.modelBindings({nodes:[node]});
    node.widgets[0].value = "new-selection.safetensors";
    binding.update("resolved.safetensors");
    assert.equal(node.widgets[0].value, "new-selection.safetensors");
});

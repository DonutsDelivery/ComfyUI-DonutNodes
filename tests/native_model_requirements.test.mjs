import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

// The web modules import ComfyUI's app.js as ESM, and this directory has no
// package.json "type": "module", so node loads web/*.js as CommonJS and
// rejects named ESM imports. Evaluate the module source in a vm context and
// collect its exports, matching the repo's other web-extension tests.
function loadModule(relPath) {
    const source = fs.readFileSync(new URL(relPath, import.meta.url), "utf8")
        .replace(/^import .*;\n/gm, "")
        .replace(/export function/g, "function")
        .replace(/export const/g, "const")
        .replace(/export \{[^}]*\};?/g, "");
    const sandbox = { __exports: {}, URL };
    vm.createContext(sandbox);
    vm.runInContext(source + "\nthis.__exports = {modelBindings, manualModelFiles};", sandbox);
    return sandbox.__exports;
}
const { modelBindings, manualModelFiles } = loadModule("../web/donut_model_requirements.js");
const sameValues = (a, b) => assert.deepEqual(JSON.parse(JSON.stringify(a)), JSON.parse(JSON.stringify(b)));

const catalog = JSON.parse(fs.readFileSync(new URL("../model_sources.json", import.meta.url))).models;
const M3 = "seedvr2_3b_int8_convrot.safetensors", M7 = "seedvr2_7b_int8_convrot.safetensors";
const VAE = "seedvr2_ema_vae_fp16.safetensors", BG = "birefnet.safetensors";
function node(type, settings = {}) {
    return {type, widgets:Object.entries(settings).map(([name,value]) => ({name,value}))};
}
function stage(settings = {}) {
    return node("DonutTiledUpscale", {upscale_engine:"SeedVR2", seedvr2_model_name:M3,
        seedvr2_vae_name:VAE, ...settings});
}
const graph = (...nodes) => ({nodes});
const refs = value => modelBindings(value).map(({folder,name}) => [folder,name]);
const widget = (n, name) => n.widgets.find(w => w.name === name);

test("the selected SeedVR2 model and VAE are discovered inside existing subgraphs", () => {
    sameValues(refs(graph({subgraph:graph(stage())})), [["diffusion_models",M3],["vae",VAE]]);
});
test("7B selection does not also download 3B", () => {
    sameValues(refs(graph(stage({seedvr2_model_name:M7}))), [["diffusion_models",M7],["vae",VAE]]);
});
test("Donut default and old stages require no SeedVR2 files", () => {
    sameValues(refs(graph(stage({upscale_engine:"Donut"}), node("DonutTiledUpscale"))), []);
});
test("muted and bypassed nodes/subgraphs are skipped", () => {
    for (const mode of [2,4]) {
        sameValues(refs(graph({...stage(),mode})), []);
        sameValues(refs(graph({mode,subgraph:graph(stage())})), []);
    }
});
test("selected feature configurations can be prepared before enabling execution", () => {
    sameValues(refs(graph(stage({enabled:false}))), [["diffusion_models",M3],["vae",VAE]]);
    sameValues(refs(graph(node("DonutEditStudio", {
        enabled:false, use_reference_b:false, mask_b_mode:"Auto subject", mask_b_model:BG,
    }))), [["background_removal",BG]]);
});
test("auto masks need BiRefNet but Off/Saved/External and legacy studios do not", () => {
    for (const mode of ["Off", "Saved mask", "External mask"]) {
        sameValues(refs(graph(node("DonutEditStudio", {mask_b_mode:mode, mask_b_model:BG}))), []);
    }
    sameValues(refs(graph(node("DonutEditStudio"))), []);
    sameValues(refs(graph(node("DonutEditStudio", {mask_b_mode:"Auto subject", mask_b_model:BG}))), [["background_removal",BG]]);
});
test("native background loader and queued preview jobs have bindings too", () => {
    sameValues(refs(graph(node("LoadBackgroundRemovalModel", {bg_removal_name:BG}),
        node("DonutSubjectMaskPreview", {model_name:BG}))), [["background_removal",BG],["background_removal",BG]]);
});
test("comfyClass takes precedence over UI type", () => {
    const n = stage(); n.comfyClass = n.type; n.type = "visual-wrapper";
    assert.equal(refs(graph(n))[0][1], M3);
});
test("existing loader, SDA and LoRA row discovery is unchanged", () => {
    const n = node("DonutLoRALoader", {slots_json:JSON.stringify([
        {id:"a", lora_name:"one.safetensors"}, {id:"b",lora_name:"two.safetensors",enabled:false},
    ])});
    const got = refs(graph(node("UNETLoader",{unet_name:"model.safetensors"}),
        node("DonutSampler",{sda_enabled:true}), n));
    assert.equal(got.length,3); sameValues(got[0],["diffusion_models","model.safetensors"]);
    assert.equal(got[2][1],"one.safetensors");
});
test("verified renamed files update the original feature widgets", () => {
    const a=stage(), b=node("DonutEditStudio",{mask_b_mode:"Auto subject",mask_b_model:BG});
    const bindings=modelBindings(graph(a,b));
    bindings[0].update("native/renamed.safetensors");
    bindings[1].update("seed/vae.safetensors");
    bindings[2].update("matting/renamed.safetensors");
    assert.equal(widget(a,"seedvr2_model_name").value,"native/renamed.safetensors");
    assert.equal(widget(a,"seedvr2_vae_name").value,"seed/vae.safetensors");
    assert.equal(widget(b,"mask_b_model").value,"matting/renamed.safetensors");
});
test("download completion never overwrites a newer model choice", () => {
    const n=stage(), bindings=modelBindings(graph(n));
    widget(n,"seedvr2_model_name").value=M7;
    bindings[0].update("renamed-3b.safetensors");
    assert.equal(widget(n,"seedvr2_model_name").value,M7);
});
test("Windows subfolders normalize without losing the rebind guard", () => {
    const n=stage({seedvr2_model_name:`native\\${M3}`});
    const bindings=modelBindings(graph(n)); assert.equal(bindings[0].name,`native/${M3}`);
    bindings[0].update(M3); assert.equal(widget(n,"seedvr2_model_name").value,M3);
});
test("cyclic/shared subgraphs do not recurse or duplicate bindings endlessly", () => {
    const nested=graph(stage()); nested.nodes.push({subgraph:nested});
    assert.equal(modelBindings(graph({subgraph:nested},{subgraph:nested})).length,2);
});
test("Registry links include both selected stages, one shared VAE and BiRefNet", () => {
    const bindings=modelBindings(graph(stage(),stage({seedvr2_model_name:M7}),
        node("DonutEditStudio",{mask_b_mode:"Auto subject",mask_b_model:BG})));
    const files=manualModelFiles(bindings,catalog); assert.equal(files.length,4);
    for (const file of files) {
        assert.match(file.url,/^https:\/\/huggingface\.co\/Comfy-Org\//);
        assert.match(file.url,/\/resolve\/[a-f0-9]{40}\//);
        assert.match(file.sha256,/^[a-f0-9]{64}$/); assert.ok(file.size>0);
        assert.ok(file.requires_nodes.length);
    }
});
test("Registry uses the requested save path for known models moved into subfolders", () => {
    const files=manualModelFiles([{folder:"diffusion_models",name:`mine/${M3}`}],catalog);
    assert.equal(files[0].path,`models/diffusion_models/mine/${M3}`);
    assert.ok(files[0].url.endsWith(M3));
});
test("unknown, ambiguous and unsafe links are not invented or activated", () => {
    const ref={folder:"loras",name:"custom/model.safetensors"};
    assert.equal(manualModelFiles([ref],catalog)[0].url,null);
    const entry={folder:"loras",filename:"one/model.safetensors",url:"https://huggingface.co/file"};
    assert.equal(manualModelFiles([ref],[entry,{...entry,filename:"two/model.safetensors"}])[0].url,null);
    for (const url of ["javascript:alert(1)","http://example.com/file","https://user:pass@example.com/file"]) {
        assert.equal(manualModelFiles([ref],[{...entry,url}])[0].url,null);
    }
});

// Run the actual Registry panel with a minimal DOM and no network API. The real
// browser is not available in this test; DOM construction and click logic run.
class Element {
    constructor(tag){this.tagName=tag;this.children=[];this.style={setProperty(){}};this.options={};}
    append(...children){this.children.push(...children);}
    replaceChildren(...children){this.children=children;}
    setAttribute(name,value){this[name]=value;}
}
function descendants(element, tag) {
    return (element.children||[]).flatMap(child=>typeof child==='object'
        ? [...(child.tagName===tag?[child]:[]), ...descendants(child,tag)] : []);
}
test("Registry panel creates clickable source links and exact locations, without download routes", () => {
    let NodeType;
    const app={rootGraph:graph(stage(),node("DonutEditStudio",{mask_b_mode:"Auto subject",mask_b_model:BG})),
        registerExtension(ext){ext.registerCustomNodes();}};
    const source=fs.readFileSync(new URL("../distribution/registry/donut_model_downloads.js",import.meta.url),"utf8")
        .replace(/^import .*;\n/gm, "");
    const context=vm.createContext({app,fitModule(){},modelBindings,manualModelFiles,MODEL_CATALOG:catalog,
        document:{createElement:tag=>new Element(tag),createTextNode:text=>text},
        LGraphNode:class{addDOMWidget(_name,_type,root){this.root=root;return {options:{}};}},
        LiteGraph:{registerNodeType(_name,cls){NodeType=cls;}}});
    vm.runInContext(source,context);
    const n=new NodeType();
    assert.equal(descendants(n.root,"li").length,0);
    descendants(n.root,"button")[0].onclick();
    const entries=descendants(n.root,"li"); assert.equal(entries.length,3);
    for (const entry of entries) {
        assert.match(descendants(entry,"code")[0].textContent,/^models\//);
        assert.equal(descendants(entry,"a")[0].rel,"noopener noreferrer");
        assert.match(descendants(entry,"a")[0].href,/^https:\/\/huggingface\.co\//);
    }
    assert.doesNotMatch(source,/fetch\s*\(|XMLHttpRequest|\/donut\/models\/(download|status|cancel)/);
});

test("Git Download missing sends feature names only, rebinds and refreshes the original modules", async () => {
    let NodeType, sent, combos=0, maskRefreshes=0, studioRenders=0;
    const a=stage(), b=node("DonutEditStudio",{mask_b_mode:"Auto subject",mask_b_model:BG});
    b._donutSubjectMask={refresh(){maskRefreshes++;}};
    b._donutEditStudio={render(){studioRenders++;}};
    const app={rootGraph:graph({subgraph:graph(a,b)}),
        async refreshComboInNodes(){combos++;}, registerExtension(ext){ext.registerCustomNodes();}};
    const api={async fetchApi(path, request){
        assert.equal(path,"/donut/models/download"); assert.equal(request.method,"POST");
        sent=JSON.parse(request.body);
        return {ok:true,async json(){return {id:"test",state:"complete",running:false,total:3,
            results:sent.models.map(item=>({...item,status:"verified",
                resolved_name:`shared/${item.name}`}))};}};
    }};
    const source=fs.readFileSync(new URL("../web/donut_model_downloads.js",import.meta.url),"utf8")
        .replace(/^import .*;\n/gm, "");
    vm.runInNewContext(source,{app,api,fitModule(){},modelBindings,
        document:{createElement:tag=>new Element(tag)},
        LGraphNode:class{addDOMWidget(_name,_type,root){this.root=root;return {options:{}};}},
        LiteGraph:{registerNodeType(_name,cls){NodeType=cls;}}});
    const n=new NodeType(); assert.equal(sent,undefined);
    const button=descendants(n.root,"button")[0]; await button.onclick();
    assert.deepEqual(sent,{models:[{folder:"diffusion_models",name:M3},
        {folder:"vae",name:VAE},{folder:"background_removal",name:BG}]});
    assert.equal(combos,1); assert.equal(maskRefreshes,1); assert.equal(studioRenders,1);
    assert.equal(widget(a,"seedvr2_model_name").value,`shared/${M3}`);
    assert.equal(widget(b,"mask_b_model").value,`shared/${BG}`);
    assert.equal(button.disabled,false);
});

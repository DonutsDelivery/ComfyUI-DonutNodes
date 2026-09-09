import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { createProgress } from "./donut_progress.js";
import { fitModule } from "./donut_layout.js?v=15";

app.registerExtension({
    name: "Donut.LatestPreview",
    registerCustomNodes() {
        class DonutLatestPreview extends LGraphNode {
            constructor() {
                super("Latest result");
                this.isVirtualNode = true;
                this.serialize_widgets = false;
                this.properties = {panel_min_width:1800, sources:{"912":"Base generation", "913":"First upscale", "914":"Final image"}};
                const root = document.createElement("div");
                root.className = "donut-latest-preview";
                const title = document.createElement("h2"), status = document.createElement("p"), image = document.createElement("img");
                title.textContent = "Latest result";
                image.alt = "Latest generated image";
                image.hidden = true;
                const selector = document.createElement("select");
                selector.setAttribute("aria-label", "Displayed preview");
                for (const [value, label] of [["latest", "Always latest"], ["912", "1 · Base generation"], ["913", "2 · First upscale"], ["914", "3 · Final image"]]) {
                    const option = document.createElement("option");
                    option.value = value; option.textContent = label; selector.append(option);
                }
                const promptModule = document.createElement("section");
                promptModule.className = "donut-result-prompt";
                const promptTitle = document.createElement("h3"), promptText = document.createElement("pre");
                promptTitle.textContent = "Final prompt · expanded wildcards";
                promptText.setAttribute("aria-label", "Final expanded prompt");
                promptModule.append(promptTitle, promptText);
                const progress = createProgress(api, () => app.rootGraph);
                root.append(title, selector, status, progress.element, image, promptModule);
                const promptsByRun = new Map();
                const show = (file, stage, prompt) => {
                    promptText.textContent = prompt ?? "Generate an image to capture its expanded prompt.";
                    status.textContent = stage || "Waiting for a generation…";
                    image.hidden = !file;
                    if (file) image.src = api.apiURL(`/view?${new URLSearchParams({filename:file.filename,subfolder:file.subfolder || "",type:file.type || "temp"})}`);
                };
                const refresh = () => {
                    const selected = this.properties.preview_selection || "latest";
                    selector.value = selected;
                    if (selected === "latest") show(this.properties.last_image, this.properties.last_stage, this.properties.last_prompt);
                    else {
                        const file = this.properties.stage_images?.[selected];
                        const stage = this.properties.sources?.[selected];
                        show(file, file ? stage : `${stage} · waiting for an image…`, this.properties.stage_prompts?.[selected]);
                    }
                };
                selector.onchange = () => {
                    this.graph?.beforeChange();
                    this.properties.preview_selection = selector.value;
                    this.graph?.afterChange();
                    refresh();
                };
                const receive = ({detail}) => {
                    const expanded = detail.output?.donut_final_prompt;
                    if (detail.prompt_id && Array.isArray(expanded) && typeof expanded[0] === "string") {
                        promptsByRun.set(detail.prompt_id, expanded[0]);
                        if (promptsByRun.size > 32) promptsByRun.delete(promptsByRun.keys().next().value);
                    }
                    const id = String(detail.display_node ?? detail.node);
                    const stage = this.properties.sources?.[id];
                    const images = detail.output?.images;
                    if (!stage || !Array.isArray(images) || !images.length) return;
                    const file = images.at(-1);
                    if (!file?.filename) return;
                    this.properties.last_image = file;
                    this.properties.last_stage = stage;
                    this.properties.last_prompt = promptsByRun.get(detail.prompt_id) ?? null;
                    this.properties.stage_prompts ||= {};
                    this.properties.stage_prompts[id] = this.properties.last_prompt;
                    this.properties.stage_images ||= {};
                    this.properties.stage_images[id] = file;
                    refresh();
                };
                this.onAdded = () => {
                    api.addEventListener("executed", receive);
                    progress.attach();
                    refresh();
                };
                this.onRemoved = () => { api.removeEventListener("executed", receive); progress.detach(); };
                this.onConfigure = refresh;
                image.onerror = () => { image.hidden = true; status.textContent = "Previous preview is unavailable. Waiting for a generation…"; };
                refresh();
                const dom = this.addDOMWidget("latest_result", "custom", root, {serialize:false,hideOnZoom:false,getValue:()=>"",setValue:()=>{}});
                dom.serialize = false;
                fitModule(this, dom, root);
            }
        }
        DonutLatestPreview.title = "Latest result";
        DonutLatestPreview.category = "donut/interface";
        LiteGraph.registerNodeType("DonutLatestPreview", DonutLatestPreview);
    },
    setup() {
        const style = document.createElement("style");
        style.textContent = `.donut-latest-preview{box-sizing:border-box;width:100%;padding:14px;background:#15191e;color:#eee;border-top:4px solid #92e4c7;border-radius:7px;font:15px/1.4 system-ui}.donut-latest-preview h2{font-size:23px;margin:0 0 6px}.donut-latest-preview select{font:14px system-ui;color:#eee;background:#242a31;border:1px solid #52615d;border-radius:5px;padding:7px 10px;margin:0 0 12px}.donut-latest-preview option{font:14px system-ui}.donut-latest-preview p{margin:0 0 10px}.donut-latest-preview img{display:block;width:100%;height:700px;object-fit:contain;background:#0c0e11}.donut-latest-preview img[hidden]{display:none}.donut-latest-preview:has(img[hidden]){min-height:750px}`;
        style.textContent += `.donut-result-prompt{margin-top:18px;padding-top:14px;border-top:1px solid #52615d}.donut-result-prompt h3{font-size:16px;margin:0 0 10px}.donut-result-prompt pre{white-space:pre-wrap;overflow-wrap:anywhere;user-select:text;font:14px/1.6 system-ui;margin:0;padding:14px;background:#0c0e11;border-radius:6px}`;
        style.textContent += `.donut-latest-preview{display:flex;flex-direction:column}.donut-execution-progress{margin-bottom:14px}.donut-execution-progress progress{width:100%;height:12px;accent-color:#92e4c7}.donut-execution-progress p{font-size:14px}`;
        document.head.append(style);
    }
});

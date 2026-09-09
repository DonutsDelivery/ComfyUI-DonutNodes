import { app } from "../../scripts/app.js";
import { fitModule } from "./donut_layout.js?v=15";
import { modelBindings } from "./donut_model_requirements.js";

// Preserve the serialized node type so existing workflows continue to load.
// This panel performs no network requests or model-file writes.
function install(node) {
    const root = document.createElement("div");
    root.className = "donut-app-controls donut-section-controls donut-model-downloads";
    const title = document.createElement("h2");
    title.textContent = "Model files";
    const status = document.createElement("p");
    status.textContent = "Automatic model downloads are unavailable in this build. Install the selected files in the folders below, then refresh ComfyUI.";
    const button = document.createElement("button");
    button.textContent = "List selected models";
    const list = document.createElement("ul");
    list.setAttribute("aria-live", "polite");
    button.onclick = () => {
        list.replaceChildren();
        const paths = new Set(modelBindings(app.rootGraph).map(({ folder, name }) => `models/${folder}/${name}`));
        for (const path of paths) {
            const item = document.createElement("li");
            item.textContent = path;
            list.append(item);
        }
        if (!paths.size) list.textContent = "No model files selected.";
    };
    const community = document.createElement("nav");
    community.setAttribute("aria-label", "Support and community");
    for (const [label, url] of [["Support on Ko-fi", "https://ko-fi.com/donutsdelivery"], ["Join Discord", "https://discord.gg/FYVZCupZ5J"]]) {
        const link = document.createElement("a");
        link.textContent = label;
        link.href = url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        community.append(link, document.createTextNode(" "));
    }
    root.append(title, status, button, list, community);
    const dom = node.addDOMWidget("download_missing", "custom", root, {
        serialize: false, hideOnZoom: false, getValue: () => "", setValue: () => {},
    });
    dom.options.serialize = false;
    fitModule(node, dom, root);
}

app.registerExtension({
    name: "Donut.ModelDownloads",
    registerCustomNodes() {
        class DonutModelDownloads extends LGraphNode {
            constructor() {
                super("Model files");
                this.isVirtualNode = true;
                this.serialize_widgets = false;
                this.properties = { panel_min_width: 820 };
                this.size = [820, 270];
                install(this);
            }
        }
        DonutModelDownloads.title = "Model files";
        DonutModelDownloads.category = "donut/interface";
        LiteGraph.registerNodeType("DonutModelDownloads", DonutModelDownloads);
    },
});

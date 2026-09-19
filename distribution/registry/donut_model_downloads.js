import { app } from "../../scripts/app.js";
import { fitModule } from "./donut_layout.js?v=15";
import { modelBindings, manualModelFiles } from "./donut_model_requirements.js";
// Generated from the packed model_sources.json by tools/prepare_registry.py.
import { MODEL_CATALOG } from "./donut_registry_catalog.js";

// Preserve the serialized node type so existing workflows continue to load.
// This panel performs no network requests or model-file writes.
function install(node) {
    const root = document.createElement("div");
    root.className = "donut-app-controls donut-section-controls donut-model-downloads";
    const title = document.createElement("h2");
    title.textContent = "Model files";
    const status = document.createElement("p");
    status.textContent = "Automatic model downloads are unavailable in this build. Choose the features first (SeedVR2 engine or Auto subject), then list their files. Download using the links and save in the indicated folders; refresh ComfyUI afterwards. Custom extra_model_paths.yaml roots may replace these default folders.";
    const button = document.createElement("button");
    button.textContent = "List selected models";
    const list = document.createElement("ul");
    list.setAttribute("aria-live", "polite");
    button.onclick = () => {
        list.replaceChildren();
        const files = manualModelFiles(modelBindings(app.rootGraph), MODEL_CATALOG);
        for (const file of files) {
            const item = document.createElement("li");
            const location = document.createElement("code"); location.textContent = file.path;
            item.append(document.createTextNode("Save as: "), location);
            if (file.url) {
                const link = document.createElement("a");
                link.textContent = "Download from upstream"; link.href = file.url;
                link.target = "_blank"; link.rel = "noopener noreferrer";
                item.append(document.createTextNode(" — "), link);
                const details = document.createElement("small");
                details.textContent = ` ${file.size} bytes · SHA-256: ${file.sha256}`;
                item.append(details);
            } else {
                item.append(document.createTextNode(" — No catalog link; obtain this file from its publisher."));
            }
            if (file.requires_nodes.length) {
                const requirement = document.createElement("p");
                requirement.textContent = `Requires native ComfyUI nodes: ${file.requires_nodes.join(", ")}. Update ComfyUI and restart if missing.`;
                item.append(requirement);
            }
            item.style.overflowWrap = "anywhere";
            list.append(item);
        }
        if (!files.length) list.textContent = "No model files selected.";
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

import { app } from "../../scripts/app.js";
import { fitModule } from "./donut_layout.js?v=16";
import { modelBindings, manualModelFiles } from "./donut_model_requirements.js?v=3";
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
    status.textContent = "Install models using the optional standalone installer below, or list the workflow’s configured models and download them individually. Disabled features are included. Custom extra_model_paths.yaml roots may replace the default folders shown.";
    const installer = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "Install models with a standalone script";
    const download = document.createElement("a");
    download.textContent = "Download model installer ZIP (Windows / Linux / macOS)";
    download.href = "https://github.com/DonutsDelivery/ComfyUI-DonutNodes/raw/refs/heads/main/distribution/manual/DonutNodes-model-installer.zip";
    download.target = "_blank"; download.rel = "noopener noreferrer";
    const steps = document.createElement("ol");
    for (const text of [
        "Extract the ZIP into your ComfyUI directory, beside main.py. Keep the model-installer folder and its files together.",
        "Windows: open model-installer and double-click install-models.bat. Portable ComfyUI’s Python is detected automatically; otherwise install Python 3.9+ and enable Add Python to PATH.",
        "Linux / macOS: open a terminal in model-installer and run: sh install-models.sh. Python 3.9+ is required.",
        "If asked, enter the ComfyUI directory. Restricted downloads may ask for a provider API token. Restart ComfyUI when installation finishes.",
    ]) {
        const step = document.createElement("li"); step.textContent = text; steps.append(step);
    }
    const scope = document.createElement("p");
    scope.textContent = "The installer downloads the entire model catalog, including optional alternatives and both SeedVR2 sizes—not just the current workflow selections. It shows the total size, verifies checksums, and skips matching installed files. It uses ComfyUI/models; custom model paths are not read. ComfyUI and custom nodes must already be installed.";
    installer.append(summary, download, steps, scope);
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
                requirement.textContent = `Requires nodes: ${file.requires_nodes.join(", ")}. Install the matching node pack or update ComfyUI, then restart if missing.`;
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
    root.append(title, status, installer, button, list, community);
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

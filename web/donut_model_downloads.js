import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { fitModule } from "./donut_layout.js?v=15";
import { modelBindings } from "./donut_model_requirements.js";

async function request(path, data) {
    const response = await api.fetchApi(path, data === undefined ? {} : {
        method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(data),
    });
    if (!response.ok) throw new Error(await response.text());
    return response.json();
}

function install(node) {
    const root = document.createElement("div");
    root.className = "donut-app-controls donut-section-controls donut-model-downloads";
    root.style.setProperty("--donut-accent", "#8bddbf");
    const title = document.createElement("h2"); title.textContent = "Download missing";
    const button = document.createElement("button"); button.textContent = "Download missing";
    const progress = document.createElement("progress"); progress.max = 100; progress.hidden = true;
    const status = document.createElement("p"); status.textContent = "Checks model hashes and installs missing files from DonutNodes’ upstream links.";
    status.setAttribute("aria-live", "polite");
    const errors = document.createElement("div");
    const community = document.createElement("nav");
    community.className = "donut-community-links";
    community.setAttribute("aria-label", "Support and community");
    for (const [label, url] of [["Support on Ko-fi", "https://ko-fi.com/donutsdelivery"], ["Join Discord", "https://discord.gg/FYVZCupZ5J"]]) {
        const link = document.createElement("a");
        link.textContent = label; link.href = url; link.target = "_blank"; link.rel = "noopener noreferrer";
        community.append(link);
    }
    root.append(title, button, progress, status, errors, community);
    let job, timer, disposed = false, bindings = [], finished;
    const refresh = async result => {
        if (finished === result.id) return;
        finished = result.id;
        await app.refreshComboInNodes();
        for (const item of result.results) {
            if (!item.resolved_name || item.resolved_name === item.name) continue;
            for (const binding of bindings) {
                if (binding.folder === item.folder && binding.name === item.name) binding.update(item.resolved_name);
            }
        }
        const visit = graph => {
            for (const current of graph?.nodes || []) { current._donutAppControls?.render(); if (current.subgraph) visit(current.subgraph); }
        };
        visit(app.rootGraph);
    };
    async function render(result) {
        job = result;
        button.textContent = result.running ? "Cancel" : "Download missing";
        progress.hidden = !result.running;
        errors.replaceChildren();
        for (const item of result.results || []) {
            if (item.status !== "error") continue;
            const message = document.createElement("p"); message.textContent = `${item.name}: ${item.message}`;
            errors.append(message);
        }
        if (result.running) {
            if (result.total_bytes) progress.value = result.bytes / result.total_bytes * 100;
            else progress.removeAttribute("value");
            const percent = result.total_bytes ? ` · ${Math.floor(result.bytes / result.total_bytes * 100)}%` : "";
            status.textContent = `${result.state === "downloading" ? "Downloading" : "Checking hashes"} · ${result.completed + 1}/${result.total}\n${result.current}${percent}`;
            clearTimeout(timer);
            if (!disposed) timer = setTimeout(poll, 1000);
        } else if (result.id) {
            const ready = result.results.filter(item => item.status !== "error").length;
            const downloaded = result.results.filter(item => item.status === "downloaded").length;
            const unlisted = result.results.filter(item => item.status === "unlisted").length;
            status.textContent = result.state === "cancelled" ? `Cancelled. ${ready} models ready.` :
                result.state === "error" ? result.message : `${ready}/${result.total} models ready · ${downloaded} downloaded.`;
            if (unlisted) status.textContent += ` ${unlisted} installed files have no catalog hash.`;
            await refresh(result);
        }
    }
    async function poll(initial = false) {
        try {
            const result = await request("/donut/models/status");
            if (initial && !result.running) return;
            if (initial) bindings = modelBindings(app.rootGraph);
            await render(result);
        }
        catch (error) { status.textContent = error.message; button.textContent = "Download missing"; progress.hidden = true; job = undefined; }
    }
    button.onclick = async () => {
        button.disabled = true;
        try {
            if (job?.running) {
                await request("/donut/models/cancel", {id:job.id});
                await poll();
            } else {
                bindings = modelBindings(app.rootGraph);
                const models = bindings.map(({folder,name}) => ({folder,name}));
                await render(await request("/donut/models/download", {models}));
            }
        } catch (error) { status.textContent = error.message; }
        finally { button.disabled = false; }
    };
    const dom = node.addDOMWidget("download_missing", "custom", root, {
        serialize:false,hideOnZoom:false,getValue:() => "",setValue:() => {},
    });
    dom.options.serialize = false;
    fitModule(node, dom, root);
    const added = node.onAdded, removed = node.onRemoved;
    node.onAdded = function() { const result = added?.apply(this, arguments); disposed = false; void poll(true); return result; };
    node.onRemoved = function() { disposed = true; clearTimeout(timer); return removed?.apply(this, arguments); };
}

app.registerExtension({
    name:"Donut.ModelDownloads",
    setup() {
        const style = document.createElement("style");
        style.textContent = `.donut-model-downloads button{background:#8bddbf;color:#10251d;font-size:20px;font-weight:700;padding:12px 22px}.donut-model-downloads progress{display:block;width:100%;margin:16px 0;accent-color:#8bddbf}.donut-model-downloads p{white-space:pre-line;overflow-wrap:anywhere}.lg-node:has(.donut-model-downloads) .lg-node-header{color:#f5f7fb!important}`;
        style.textContent += `.donut-community-links{display:flex;flex-wrap:wrap;gap:10px;border-top:1px solid #ffffff25;margin-top:16px;padding-top:16px}.donut-community-links a{display:inline-flex;align-items:center;padding:9px 12px;border:1px solid #ffffff35;border-radius:6px;background:#242a31;color:#eee;font:600 14px/1.4 system-ui;text-decoration:none}.donut-community-links a:hover{background:#33413d;border-color:#8bddbf}.donut-community-links a:focus-visible{outline:2px solid #8bddbf;outline-offset:3px}`;
        document.head.append(style);
    },
    registerCustomNodes() {
        class DonutModelDownloads extends LGraphNode {
            constructor() {
                super("Download missing");
                this.isVirtualNode = true;
                this.serialize_widgets = false;
                this.properties = {panel_min_width:820};
                this.size = [820, 270];
                install(this);
            }
        }
        DonutModelDownloads.title = "Download missing";
        DonutModelDownloads.category = "donut/interface";
        LiteGraph.registerNodeType("DonutModelDownloads", DonutModelDownloads);
    },
});

import { app } from "../../scripts/app.js";

app.registerExtension({
  name: "Donut.DependencyCheck",
  nodeCreated(node) {
    if (node.comfyClass !== "DonutDependencyCheck" || node._donutDependencyReport) return;
    const text = document.createElement("textarea");
    text.className = "comfy-multiline-input";
    text.readOnly = true;
    text.spellcheck = false;
    text.placeholder = "Queue this node to inspect dependencies. No packages will be changed.";
    text.style.fontFamily = "monospace";
    text.style.fontSize = "12px";
    text.style.width = "100%";
    text.style.height = "100%";
    text.setAttribute("aria-label", "DonutNodes dependency report");
    const widget = node.addDOMWidget("dependency_report", "customtext", text, {
      serialize: false,
      getValue: () => "",
      setValue: () => {},
    });
    widget.options.serialize = false;
    node._donutDependencyReport = text;
    node.setSize([Math.max(node.size[0], 520), Math.max(node.size[1], 320)]);

    // Only this diagnostic node instance; no global/prototype patches or
    // executable HTML. The report is selectable for copying into bug reports.
    const previous = node.onExecuted;
    node.onExecuted = function (message) {
      const result = previous?.apply(this, arguments);
      const value = message?.text;
      text.value = Array.isArray(value) ? value.map(String).join("\n") : String(value ?? "");
      this.setDirtyCanvas?.(true, true);
      return result;
    };
  },
});

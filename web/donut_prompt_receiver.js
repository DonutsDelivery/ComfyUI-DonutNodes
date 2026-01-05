import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "donut.PromptReceiver",
    async nodeCreated(node) {
        if (node.comfyClass === "DonutPromptReceiver") {
            // Hide internal _api_* widgets
            const hiddenWidgets = ["_api_prompt", "_api_face_prompt", "_api_negative_prompt", "_api_prompt_id"];
            for (const name of hiddenWidgets) {
                const widget = node.widgets?.find(w => w.name === name);
                if (widget) {
                    widget.type = "hidden";
                    widget.computeSize = () => [0, -4];
                }
            }

            // Cancel All - interrupt + clear queue
            node.addWidget("button", "Cancel All", "cancel", function() {
                fetch("/interrupt", { method: "POST" });
                fetch("/queue", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ clear: true })
                }).then(() => console.log("[DonutPromptReceiver] Cancelled all"))
                  .catch(err => console.error("[DonutPromptReceiver] Failed:", err));
            });

            // Skip current only
            node.addWidget("button", "Skip Current", "skip", function() {
                fetch("/interrupt", { method: "POST" })
                    .then(() => console.log("[DonutPromptReceiver] Skipped"))
                    .catch(err => console.error("[DonutPromptReceiver] Failed:", err));
            });

            // Send to Claude - sends user_prompt + meta_prompt + num_prompts to simple-claude-gui
            node.addWidget("button", "Send to Claude", "send_claude", function() {
                const userPromptWidget = node.widgets?.find(w => w.name === "user_prompt");
                const metaPromptWidget = node.widgets?.find(w => w.name === "meta_prompt");
                const numPromptsWidget = node.widgets?.find(w => w.name === "num_prompts");
                const portWidget = node.widgets?.find(w => w.name === "port");
                const outHostWidget = node.widgets?.find(w => w.name === "outgoing_host");
                const outPortWidget = node.widgets?.find(w => w.name === "outgoing_port");
                const outProtocolWidget = node.widgets?.find(w => w.name === "outgoing_protocol");
                const loopEnabledWidget = node.widgets?.find(w => w.name === "loop_enabled");

                const userPrompt = userPromptWidget?.value || "";
                const metaPrompt = metaPromptWidget?.value || "";
                const numPrompts = numPromptsWidget?.value || 1;
                const port = portWidget?.value || 8001;
                const outHost = outHostWidget?.value || "localhost";
                const outPort = outPortWidget?.value || 5728;
                const outProtocol = outProtocolWidget?.value || "http";
                const loopEnabled = loopEnabledWidget?.value || false;

                if (!userPrompt.trim()) {
                    console.error("[DonutPromptReceiver] No user prompt to send");
                    return;
                }

                const apiInfo = `
ComfyUI Prompt API (port ${port}):
- POST http://localhost:${port}/prompt with JSON body
- Single: {"prompt": "main prompt", "face_prompt": "face details", "negative_prompt": "bad things"}
- Multiple: {"prompts": [{"prompt": "...", "face_prompt": "...", "negative_prompt": "..."}, ...]}
- face_prompt and negative_prompt are optional
`;

                const fullPrompt = `${apiInfo}

Apply this meta prompt: "${metaPrompt}"
To this base prompt: "${userPrompt}"

Generate ${numPrompts} unique variation(s) and queue them using the API above.`;

                // Notify backend of this send_to_claude action for loop feature
                // Also sync the loop_enabled toggle state
                fetch(`http://localhost:${port}/loop/config`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        enabled: loopEnabled,
                        user_prompt: userPrompt,
                        meta_prompt: metaPrompt,
                        num_prompts: numPrompts,
                        host: outHost,
                        port: outPort,
                        protocol: outProtocol,
                        incoming_port: port
                    })
                }).catch(err => console.warn("[DonutPromptReceiver] Failed to notify loop config:", err));

                const outgoingUrl = `${outProtocol}://${outHost}:${outPort}/prompt`;
                fetch(outgoingUrl, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ prompt: fullPrompt })
                }).then(() => console.log(`[DonutPromptReceiver] Sent to ${outgoingUrl}`))
                  .catch(err => console.error(`[DonutPromptReceiver] Failed to send to ${outgoingUrl}:`, err));
            });
        }

        if (node.comfyClass === "DonutImageReporter") {
            // Hide internal _api_prompt_id widget
            const hiddenWidgets = ["_api_prompt_id"];
            for (const name of hiddenWidgets) {
                const widget = node.widgets?.find(w => w.name === name);
                if (widget) {
                    widget.type = "hidden";
                    widget.computeSize = () => [0, -4];
                    widget.serializeValue = () => widget.value;
                }
            }
            // Force node resize to remove hidden widget space
            setTimeout(() => {
                node.setSize(node.computeSize());
            }, 100);
        }
    }
});

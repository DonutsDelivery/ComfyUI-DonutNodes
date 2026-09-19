import { app } from "../../scripts/app.js";
import { addSeedVR2Controls } from "./donut_seedvr2_controls_model.js";

let pending = false;
function refresh() {
    if (pending) return;
    pending = true;
    queueMicrotask(() => {
        pending = false;
        if (!app.rootGraph) return;
        for (const panel of addSeedVR2Controls(app.rootGraph)) {
            panel._donutAppControls?.render();
            panel.setDirtyCanvas?.(true, true);
        }
    });
}
app.registerExtension({
    name:"Donut.SeedVR2UpscaleControls",
    afterConfigureGraph:refresh,
    nodeCreated:refresh,
});

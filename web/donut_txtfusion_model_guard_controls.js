import {app} from '../../scripts/app.js';
import {addModelTxtfusionGuardControls} from './donut_txtfusion_model_guard_controls_model.js?v=1';
let pending = false;
function refresh() {
    if (pending) return;
    pending = true;
    queueMicrotask(() => {
        pending = false;
        for (const panel of addModelTxtfusionGuardControls(app.rootGraph || app.graph)) {
            panel._donutAppControls?.render();
            panel.setDirtyCanvas?.(true, true);
        }
    });
}
app.registerExtension({
    name: 'Donut.ModelTxtfusionRMSGuard',
    beforeConfigureGraph: data => { addModelTxtfusionGuardControls(data); },
    afterConfigureGraph: refresh, nodeCreated: refresh, refreshComboInNodes: refresh,
});

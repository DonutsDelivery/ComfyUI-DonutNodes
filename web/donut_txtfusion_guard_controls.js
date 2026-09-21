import {app} from '../../scripts/app.js';
import {addTxtfusionGuardControls} from './donut_txtfusion_guard_controls_model.js';

let pending = false;
function refresh() {
    if (pending) return;
    pending = true;
    queueMicrotask(() => {
        pending = false;
        for (const panel of addTxtfusionGuardControls(app.rootGraph)) {
            panel._donutAppControls?.render();
            panel.setDirtyCanvas?.(true, true);
        }
    });
}
app.registerExtension({
    name:'Donut.TxtfusionInternalGuardControls',
    beforeConfigureGraph(data) { addTxtfusionGuardControls(data); },
    afterConfigureGraph:refresh,
    nodeCreated:refresh,
    refreshComboInNodes:refresh,
});

import {app} from '../../scripts/app.js';
import {addNagNoTauControls} from './donut_nag_tau_controls_model.js';

let pending = false;
function refresh() {
    if (pending) return;
    pending = true;
    queueMicrotask(() => {
        pending = false;
        for (const panel of addNagNoTauControls(app.rootGraph)) {
            panel._donutAppControls?.render();
            panel.setDirtyCanvas?.(true, true);
        }
    });
}
app.registerExtension({
    name:'Donut.NagDisableTauClipping',
    beforeConfigureGraph(data) { addNagNoTauControls(data); },
    afterConfigureGraph:refresh,
    nodeCreated:refresh,
    refreshComboInNodes:refresh,
});

// DOM adapters reuse the existing panel's widget callbacks and LoRA row editor.
// No second slots_json implementation, catalog service, or model loader.
import {app} from '../../scripts/app.js';
import {graphEntries, SIZE_FIELDS, widgetValue} from './donut_panel_categories_model.js?v=11';
import {sizingVisibility} from './donut_reference_crop_geometry.js';
export const PANEL_CATEGORY_CSS = `
.donut-section-columns>section:has(textarea),
.donut-section-columns>section:has(.donut-lora-row),
.donut-section-columns>section:has(.donut-vector-control){grid-column:1/-1}
.donut-app-controls .donut-panel-lora-search{width:100%;min-width:0}
.donut-app-controls label:has(.donut-panel-lora-search){grid-column:1/-1}
.donut-edit-studio .donut-sizing-proxied>.de-row{display:none!important}
`;
export function upgradePanelLoraPickers(root) {
    if (typeof LiteGraph === 'undefined' || !LiteGraph.ContextMenu) return 0;
    let count = 0;
    for (const select of root.querySelectorAll('select[aria-label^="Installed LoRA "]')) {
        if (select.dataset.donutSearchPicker) continue;
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'donut-panel-lora-search';
        button.setAttribute('aria-label',select.getAttribute('aria-label'));
        button.setAttribute('aria-haspopup','menu');
        button.textContent = `${select.value} ▾`;
        button.title = select.value;
        button.addEventListener('click',event => {
            event.preventDefault(); event.stopPropagation();
            // Read the complete, current catalog when opened. Unlike a
            // datalist, the selected filename never pre-filters the choices.
            const values = [...select.options].map(option => option.value);
            const rect = button.getBoundingClientRect();
            const anchor = event.detail ? event : new MouseEvent('click',{
                clientX:rect.left, clientY:rect.bottom,
            });
            // DOM controls bypass LiteGraph's canvas pointer handlers. Its
            // native filter still expects the active canvas to be initialized.
            if (typeof LGraphCanvas !== 'undefined' && app.canvas) LGraphCanvas.active_canvas = app.canvas;
            // This is the same menu and built-in Filter list field used by
            // ComfyUI's original combo widgets, including keyboard navigation.
            new LiteGraph.ContextMenu(values, {
                event:anchor, className:'dark', scale:1,
                callback(value) {
                    if (!button.isConnected || ![...select.options].some(option => option.value === value)) return;
                    if (value !== select.value) {
                        select.value = value;
                        select.dispatchEvent(new Event('change',{bubbles:true}));
                    }
                    button.textContent = `${select.value} ▾`;
                    button.title = select.value;
                },
            });
        });
        select.dataset.donutSearchPicker = 'native';
        select.hidden = true; select.setAttribute('aria-hidden','true'); select.tabIndex = -1;
        select.insertAdjacentElement('afterend',button);
        count++;
    }
    return count;
}
export function syncCategorizedPanels(rootGraph) {
    const entries = graphEntries(rootGraph), byPath = new Map(entries.map(entry => [JSON.stringify(entry.path.map(String)),entry.node]));
    const owners = new Map();
    for (const {node:panel} of entries) {
        const root = panel._donutAppControls?.root;
        if (!root) continue;
        upgradePanelLoraPickers(root);
        for (const group of panel.properties?.donut_app_controls?.groups || []) {
            if (!group.donut_image_size) continue;
            const studio = byPath.get(JSON.stringify(group.donut_image_size.map(String)));
            const fields = SIZE_FIELDS.map(([name,label]) => ({name, input:root.querySelector(`[aria-label="${label}"]`)}));
            if (!studio || fields.some(field => !field.input)) continue;
            owners.set(studio,panel);
            const cropHidden = sizingVisibility(Object.fromEntries(['enabled','geometry_mode','output_canvas','resolution_mode'].map(name => [name,widgetValue(studio,name)])));
            const mode = widgetValue(studio,'resolution_mode');
            const reference = widgetValue(studio,'enabled') && String(mode).startsWith('Reference A');
            for (const {name,input} of fields) {
                const hidden = cropHidden ? cropHidden[name] : ['width','height'].includes(name) ? mode !== 'Custom'
                    : name === 'aspect_ratio' ? mode === 'Custom' || reference
                    : name === 'megapixels' ? mode === 'Custom' || (reference && mode === 'Reference A · crop only') : false;
                const label = input.closest('label'); if (label) label.hidden = hidden;
            }
        }
    }
    for (const {node} of entries) {
        const root = node._donutEditStudio?.root;
        if (!root) continue;
        const section = root.querySelector('[aria-label="Output sizing mode"]')?.closest('.de-section');
        if (!section) continue;
        const external = owners.has(node);
        section.classList.toggle('donut-sizing-proxied',external);
        const heading = section.querySelector('.de-subhead');
        if (heading) {
            heading.dataset.donutOriginalTitle ||= heading.textContent;
            const title = external ? 'Effective output size · controls in Generation setup' : heading.dataset.donutOriginalTitle;
            if (heading.textContent !== title) heading.textContent = title;
        }
        // Standalone studios and ambiguous/copied layouts retain local sizing.
        // Only hide the duplicate after the destination fields actually exist.
    }
}

// DOM adapters reuse the existing panel's widget callbacks and LoRA row editor.
// No second slots_json implementation, catalog service, or model loader.
import {graphEntries, SIZE_FIELDS, widgetValue} from './donut_panel_categories_model.js';
import {sizingVisibility} from './donut_reference_crop_geometry.js';
let pickerId = 0;
export const PANEL_CATEGORY_CSS = `
.donut-section-columns>section:has(textarea),
.donut-section-columns>section:has(.donut-lora-row),
.donut-section-columns>section:has(.donut-vector-control){grid-column:1/-1}
.donut-app-controls .donut-panel-lora-search{width:100%;min-width:0}
.donut-app-controls label:has(.donut-panel-lora-search){grid-column:1/-1}
.donut-edit-studio .donut-sizing-proxied>.de-row{display:none!important}
`;
export function upgradePanelLoraPickers(root) {
    let count = 0;
    for (const select of root.querySelectorAll('select[aria-label^="Installed LoRA "]')) {
        if (select.dataset.donutSearchPicker) continue;
        const input = document.createElement('input'), list = document.createElement('datalist');
        const id = `donut-panel-lora-${++pickerId}`;
        list.id = id;
        for (const option of select.options) {
            const item = document.createElement('option'); item.value = option.value; list.append(item);
        }
        input.type = 'text'; input.value = select.value;
        input.setAttribute('list',id); input.setAttribute('aria-label',select.getAttribute('aria-label'));
        input.className = 'donut-panel-lora-search'; input.placeholder = 'Type to filter installed LoRAs…';
        input.autocomplete = 'off'; input.spellcheck = false;
        const commit = () => {
            const value = input.value.trim();
            // Validate against the ORIGINAL live select, not a stale list or
            // typed partial filename. Its original onchange handles row state.
            if (![...select.options].some(option => option.value === value)) { input.value = select.value; return; }
            if (value !== select.value) {
                select.value = value;
                select.dispatchEvent(new Event('change',{bubbles:true}));
            }
        };
        input.addEventListener('change',commit);
        input.addEventListener('keydown',event => {
            if (event.key === 'Enter') { event.preventDefault(); commit(); input.blur(); }
            if (event.key === 'Escape') { input.value = select.value; input.blur(); }
        });
        select.dataset.donutSearchPicker = id;
        select.hidden = true; select.setAttribute('aria-hidden','true'); select.tabIndex = -1;
        select.insertAdjacentElement('afterend',list); select.insertAdjacentElement('afterend',input);
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
            const title = external ? 'Effective output size · controls in Generate & finish' : heading.dataset.donutOriginalTitle;
            if (heading.textContent !== title) heading.textContent = title;
        }
        // Standalone studios and ambiguous/copied layouts retain local sizing.
        // Only hide the duplicate after the destination fields actually exist.
    }
}

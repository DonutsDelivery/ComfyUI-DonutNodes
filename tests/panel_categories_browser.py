"""Isolated Chromium DOM regression checks; not a complete ComfyUI session.

Run: python tests/panel_categories_browser.py --chromium /usr/bin/chromium
Playwright/Chromium are developer test dependencies, never runtime dependencies.
"""
import argparse
from pathlib import Path
import re
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chromium', default='/usr/bin/chromium')
    args = parser.parse_args()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=args.chromium, headless=True, args=['--no-sandbox'])
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.set_content('<!doctype html><html><head></head><body></body></html>')
        # Evaluate the real pure modules without imports; this isolates DOM
        # behavior and makes no local/remote network or server assumptions.
        for filename in ['donut_panel_categories_model.js', 'donut_panel_categories_dom.js']:
            source = (ROOT / 'web' / filename).read_text(encoding='utf-8')
            source = re.sub(r'^import .*;\n', '', source, flags=re.M)
            source = re.sub(r'export (function|const)', r'\1', source)
            page.add_script_tag(content=source)
        page.add_script_tag(content='window.dom={PANEL_CATEGORY_CSS,upgradePanelLoraPickers,syncCategorizedPanels}; window.model={SIZE_FIELDS};')
        page.evaluate('''async () => {
            const {dom,model} = window;
            document.body.replaceChildren();
            const style=document.createElement('style'); style.textContent=dom.PANEL_CATEGORY_CSS; document.head.append(style);
            window.checks=0;
            const check=(condition,message)=>{if(!condition) throw new Error(message); checks++;};
            const panel=document.createElement('div'); panel.className='donut-app-controls'; document.body.append(panel);
            const row=document.createElement('label'); const select=document.createElement('select');
            select.setAttribute('aria-label','Installed LoRA 1');
            for(const value of ['None','faces/Character.safetensors','style/Watercolour.safetensors']) {const option=document.createElement('option'); option.value=value; option.textContent=value; select.append(option);}
            select.value='faces/Character.safetensors'; row.append(select); panel.append(row);
            let calls=0; select.onchange=()=>calls++;
            check(dom.upgradePanelLoraPickers(panel)===1,'upgrade visible panel select');
            check(dom.upgradePanelLoraPickers(panel)===0,'idempotent picker install');
            const input=panel.querySelector('input'); input.value='Water'; input.dispatchEvent(new Event('change'));
            check(calls===0 && input.value==='faces/Character.safetensors','reject incomplete filenames without modifying row');
            input.value='style/Watercolour.safetensors'; input.dispatchEvent(new Event('change'));
            check(calls===1 && select.value===input.value,'call original row onchange');
            const other=document.createElement('div'); other.append(row.cloneNode(true));
            other.querySelector('input').remove(); other.querySelector('datalist').remove(); delete other.querySelector('select').dataset.donutSearchPicker;
            document.body.append(other); dom.upgradePanelLoraPickers(other);
            check(input.getAttribute('list')!==other.querySelector('input').getAttribute('list'),'unique IDs for copied row numbers');
            const studioRoot=document.createElement('div'); studioRoot.className='donut-edit-studio';
            studioRoot.innerHTML='<div class="de-section"><div class="de-subhead">Output size</div><div class="de-row"><select aria-label="Output sizing mode"></select></div><div class="de-output">1152 × 896</div></div>';
            document.body.append(studioRoot);
            const values={enabled:false,resolution_mode:'Preset',aspect_ratio:'4:3 Standard',megapixels:1,width:1152,height:896,multiple:'64'};
            const studio={id:700,type:'DonutEditStudio',widgets:Object.entries(values).map(([name,value])=>({name,value})),_donutEditStudio:{root:studioRoot}};
            const generateRoot=document.createElement('div'); document.body.append(generateRoot);
            for(const [name,title] of model.SIZE_FIELDS) {const label=document.createElement('label'); const input=document.createElement('input'); input.setAttribute('aria-label',title); label.append(input); generateRoot.append(label);}
            const generate={id:800,type:'DonutWorkflowPanel',_donutAppControls:{root:generateRoot},properties:{donut_app_controls:{groups:[{donut_image_size:[700]}]}}};
            const graph={nodes:[studio,generate]}; dom.syncCategorizedPanels(graph);
            check(getComputedStyle(studioRoot.querySelector('.de-row')).display==='none','hide duplicate Edit Studio controls');
            check(getComputedStyle(studioRoot.querySelector('.de-output')).display!=='none','retain effective-size readout');
            check(generateRoot.querySelector('[aria-label="Custom output width"]').closest('label').hidden,'hide custom width in preset mode');
            studio.widgets.find(w=>w.name==='resolution_mode').value='Custom'; dom.syncCategorizedPanels(graph);
            check(!generateRoot.querySelector('[aria-label="Custom output width"]').closest('label').hidden,'show custom width in Custom mode');
            check(generateRoot.querySelector('[aria-label="Output aspect ratio"]').closest('label').hidden,'hide unused aspect in Custom mode');
            studio.widgets.find(w=>w.name==='resolution_mode').value='Reference A · crop only'; studio.widgets.find(w=>w.name==='enabled').value=true; dom.syncCategorizedPanels(graph);
            check(generateRoot.querySelector('[aria-label="Output megapixels"]').closest('label').hidden,'hide inactive MP in reference crop-only mode');
            graph.nodes=[studio]; dom.syncCategorizedPanels(graph);
            check(getComputedStyle(studioRoot.querySelector('.de-row')).display!=='none','restore local sizing when destination panel is absent');
        }''')
        assert not errors, errors
        print(f"PASS: {page.evaluate('checks')} browser checks; no page errors")
        browser.close()


if __name__ == '__main__':
    main()

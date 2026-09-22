"""Optional Chromium raster parity checks, NOT a ComfyUI end-to-end UI test.

Run with an installed browser: python tests/test_tone_lab_browser.py /usr/bin/chromium
Requires playwright and Pillow for tests only; runtime node requires neither.
"""
import base64
import io
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image
import torch
from playwright.sync_api import sync_playwright

from test_tone_lab import ROOT, load_node, fixture_model
import donut_tone_engine as engine


def main(executable):
    source = (ROOT / 'tests/fixtures/tone_lab_v4_reference.cjs').read_text().replace('module.exports=buildFeatureEngine();', 'window.E=buildFeatureEngine();')
    rows = []
    with tempfile.TemporaryDirectory() as tmp, sync_playwright() as p:
        module = load_node(Path(tmp))
        browser = p.chromium.launch(executable_path=executable, headless=True, args=['--no-sandbox'])
        page = browser.new_page()
        page.add_script_tag(content=source)
        rng = np.random.default_rng(472)
        model = fixture_model(varying=True)
        for h, w in [(31, 37), (620, 800), (768, 1024), (403, 237), (896, 1152), (1536, 2048)]:
            a = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
            buf = io.BytesIO();Image.fromarray(a).save(buf, format='PNG')
            ref = page.evaluate('''async ({url,model}) => {
                const i=new Image();i.src=url;await i.decode();
                const scale=Math.min(1,256/Math.max(i.naturalWidth,i.naturalHeight));
                const c=document.createElement('canvas');c.width=Math.max(1,Math.round(i.naturalWidth*scale));c.height=Math.max(1,Math.round(i.naturalHeight*scale));
                const ctx=c.getContext('2d',{colorSpace:'srgb',willReadFrequently:true});ctx.drawImage(i,0,0,c.width,c.height);
                const a=ctx.getImageData(0,0,c.width,c.height).data,s=E.analyze(a,c.width,c.height);
                return {pixels:Array.from(a),w:c.width,h:c.height,features:s.features,pred:E.predict(s.features,model)};
            }''', {'url':'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode(), 'model':model['model']})
            proxy = module.analysis_proxy(torch.from_numpy(a).float() / 255)
            pixels = np.array(ref['pixels'], dtype=np.uint8).reshape(ref['h'], ref['w'], 4)
            pd = np.abs(proxy.astype(float) - pixels)
            # Canvas backend/browser versions can use another raster path.
            # Report the difference, don't claim portable bit-exact resizing.
            stats = engine.analyze_rgba(proxy)
            pred = engine.predict(stats['features'], engine.validate_export(model))
            fd = np.abs(np.array(stats['features']) - ref['features'])
            row = {'input':[w,h], 'proxy':[ref['w'],ref['h']], 'pixel_max_error':float(pd.max()),
                   'pixel_mean_error':float(pd.mean()), 'feature_max_error':float(fd.max()),
                   'gamma_error':abs(pred['gamma']-ref['pred']['gamma']), 'gain_error':abs(pred['gain']-ref['pred']['gain'])}
            rows.append(row)
        browser.close()
    print(json.dumps(rows, indent=2))
    if any(r['pixel_max_error'] > 0 or r['feature_max_error'] > 1e-10 or r['gamma_error'] > 1e-10 for r in rows):
        raise SystemExit('Raster parity differs on this browser. Inspect errors before claiming end-to-end pixel parity.')


if __name__ == '__main__':
    if len(sys.argv) != 2:raise SystemExit('Usage: python tests/test_tone_lab_browser.py /path/to/chromium')
    main(sys.argv[1])

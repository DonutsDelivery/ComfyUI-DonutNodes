"""Standalone browser smoke test, independent of a running ComfyUI server.
Requires Playwright plus Chromium. Native BiRefNet inference is NOT exercised.
"""
from pathlib import Path
import argparse
import base64
import io
import json

from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def uri(image):
    data = io.BytesIO(); image.save(data, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(data.getvalue()).decode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chromium', default=None, help='Optional system Chromium executable')
    args = parser.parse_args()
    source = Image.new('RGB', (1600, 1200), (180, 100, 30))
    mask = Image.new('L', (1600, 1200), 0)
    ImageDraw.Draw(mask).ellipse((300, 100, 1325, 1125), fill=255)
    module = 'data:text/javascript;base64,' + base64.b64encode((ROOT / 'web/donut_subject_mask_editor.js').read_bytes()).decode()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path=args.chromium)
        page = browser.new_page(viewport={'width': 1200, 'height': 950})
        page.set_default_timeout(5000)
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.set_content('<html><body></body></html>')
        page.evaluate('''async ({module, source, mask}) => {
            const {openSubjectMaskEditor} = await import(module);
            const load = url => new Promise((resolve, reject) => {const img = new Image(); img.onload = () => resolve(img); img.onerror = reject; img.src = url;});
            const image = await load(source), maskImage = await load(mask);
            window.openMask = (empty = false) => openSubjectMaskEditor({image, maskImage:empty ? null : maskImage, onApply: async blob => {
                const url = URL.createObjectURL(blob), image = await load(url);
                const c = document.createElement('canvas'); c.width = image.width; c.height = image.height;
                const ctx = c.getContext('2d'); ctx.drawImage(image, 0, 0); URL.revokeObjectURL(url);
                window.saved = {width:c.width, height:c.height, first:ctx.getImageData(0,0,1,1).data[0], center:ctx.getImageData(800,600,1,1).data[0]};
            }});
            window.openMask();
        }''', {'module': module, 'source': uri(source), 'mask': uri(mask)})
        page.get_by_role('button', name='Show mask', exact=True).click()
        page.get_by_role('button', name='Invert', exact=True).click()
        page.get_by_role('button', name='Undo', exact=True).click()
        page.get_by_role('button', name='Use subject mask', exact=True).click()
        page.wait_for_function('window.saved !== undefined')
        saved = page.evaluate('window.saved')
        assert saved == {'width': 1600, 'height': 1200, 'first': 0, 'center': 255}, saved
        page.wait_for_function("document.querySelector('dialog') === null")
        page.evaluate('() => {window.openMask();}')
        page.get_by_role('button', name='Cancel', exact=True).click()
        page.wait_for_function("document.querySelector('dialog') === null")
        page.evaluate('() => {window.openMask(true);}')
        visible = page.locator('dialog canvas').evaluate('(c) => Array.from(c.getContext("2d").getImageData(0,0,1,1).data).slice(0,3)')
        assert visible == [180,100,30], visible
        page.get_by_role('button', name='Cancel', exact=True).click()
        page.wait_for_function("document.querySelector('dialog') === null")
        assert not errors, errors
        print(json.dumps({'checks': ['mask/invert/undo roundtrip', 'full-resolution PNG preserves matte', 'Apply closes', 'Cancel closes', 'empty-mask editor shows source', 'no page errors'], 'saved': saved}))
        browser.close()


if __name__ == '__main__':
    main()

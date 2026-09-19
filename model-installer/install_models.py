"""Standalone DonutNodes model installer. Python 3.9+, standard library only."""
import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from urllib.parse import urlsplit

HOSTS = ('huggingface.co', 'hf.co', 'civitai.com', 'civitai.red', 'civitai.green',
         'github.com', 'githubusercontent.com', 'dl.fbaipublicfiles.com',
         'civitai-delivery-worker-prod.5ac0637cfd0766c97916cefa3764fbdf.r2.cloudflarestorage.com')
FOLDERS = {'diffusion_models', 'checkpoints', 'text_encoders', 'vae', 'loras',
           'upscale_models', 'ultralytics', 'sams', 'controlnet', 'clip_vision',
           'embeddings', 'background_removal'}


def checked_url(url):
    p = urlsplit(url)
    if (p.scheme != 'https' or p.username or p.password or p.port not in (None, 443)
            or not any(p.hostname == h or (p.hostname or '').endswith('.' + h) for h in HOSTS)):
        raise ValueError('Unsupported upstream URL or redirect')
    return url


class Redirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        checked_url(newurl)
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected and urlsplit(req.full_url).hostname != urlsplit(newurl).hostname:
            redirected.remove_header('Authorization')
        return redirected


def catalog(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('version') != 1:
        raise ValueError('Unsupported model catalog version')
    seen = set()
    for e in data['models']:
        name = e['filename']
        if (e['folder'] not in FOLDERS or not isinstance(name, str) or '\\' in name
                or ':' in name or any(p in ('', '.', '..') for p in name.split('/'))):
            raise ValueError('Unsafe model destination')
        key = (e['folder'], name)
        if key in seen or type(e['size']) is not int or e['size'] <= 0 or not re.fullmatch(r'[0-9a-fA-F]{64}', e['sha256']):
            raise ValueError('Invalid size, checksum or duplicate model')
        seen.add(key)
        checked_url(e['url'])
    return data['models']


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def install(entry, root, opener, tokens):
    target = root / 'models' / entry['folder'] / entry['filename']
    if not target.resolve().is_relative_to((root / 'models').resolve()):
        raise ValueError('Destination escapes models directory')
    if target.exists():
        if target.stat().st_size == entry['size'] and digest(target).lower() == entry['sha256'].lower():
            return 'Already installed and verified'
        raise ValueError('Different file already exists; kept untouched: ' + str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(target.parent).free < entry['size']:
        raise ValueError('Not enough free disk space')
    host = urlsplit(entry['url']).hostname
    provider = 'CIVITAI_API_KEY' if host in ('civitai.com', 'civitai.red', 'civitai.green') else 'HF_TOKEN'
    def request():
        headers = {'User-Agent': 'DonutNodes standalone model installer', 'Accept-Encoding': 'identity'}
        if tokens.get(provider):
            headers['Authorization'] = 'Bearer ' + tokens[provider]
        return opener.open(urllib.request.Request(checked_url(entry['url']), headers=headers), timeout=60)
    try:
        response = request()
    except urllib.error.HTTPError as error:
        if error.code not in (401, 403) or not sys.stdin.isatty():
            raise
        error.close()
        tokens[provider] = getpass.getpass(f'{host} requires access. Enter {provider} (hidden, not saved): ')
        response = request()
    temporary = None
    try:
        with response:
            fd, temporary = tempfile.mkstemp(prefix='.donut-', suffix='.part', dir=target.parent)
            h, count, last = hashlib.sha256(), 0, -1
            with os.fdopen(fd, 'wb') as output:
                while True:
                    block = response.read(4 * 1024 * 1024)
                    if not block:
                        break
                    count += len(block)
                    if count > entry['size']:
                        raise ValueError('Download exceeds expected size')
                    output.write(block)
                    h.update(block)
                    percent = count * 100 // entry['size']
                    if percent != last:
                        print(f'\r  {percent}%', end='', flush=True)
                        last = percent
            print()
            if count != entry['size'] or h.hexdigest().lower() != entry['sha256'].lower():
                raise ValueError('Download failed size/SHA-256 verification')
            os.link(temporary, target)  # Atomic publication; never replace an existing file.
        return 'Downloaded and verified'
    finally:
        if temporary:
            Path(temporary).unlink(missing_ok=True)


def find_root(explicit=None):
    candidates = [Path(explicit).expanduser()] if explicit else [Path.cwd(), *Path(__file__).resolve().parents]
    for candidate in candidates:
        for root in (candidate, candidate / 'ComfyUI'):
            if (root / 'main.py').is_file() and (root / 'models').is_dir():
                return root.resolve()
    raise ValueError('Choose the ComfyUI directory containing main.py and models/')


def main():
    parser = argparse.ArgumentParser(description='Install ALL models in the bundled DonutNodes catalog, including optional alternatives.')
    parser.add_argument('--comfyui', help='ComfyUI directory (or portable parent directory)')
    parser.add_argument('--list', action='store_true', help='List models and total size without downloading')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    source = here / 'model_sources.json'
    if not source.exists():
        source = here.parent / 'model_sources.json'
    entries = catalog(source)
    for entry in entries:
        print(f"models/{entry['folder']}/{entry['filename']} ({entry['size'] / 1e9:.2f} GB)")
    print(f'Total catalog: {sum(e["size"] for e in entries) / 1e9:.2f} GB. Verified existing files are skipped.')
    if args.list:
        return 0
    try:
        root = find_root(args.comfyui)
    except ValueError:
        if not sys.stdin.isatty():
            raise
        root = find_root(input('ComfyUI directory: ').strip().strip('"'))
    print(f'Installing into {root / "models"}')
    tokens = {key: os.environ.get(key, '') for key in ('HF_TOKEN', 'CIVITAI_API_KEY')}
    opener = urllib.request.build_opener(Redirects())
    failures = 0
    for entry in entries:
        print('\n' + entry['filename'])
        try:
            print(install(entry, root, opener, tokens))
        except Exception as error:
            failures += 1
            # HTTP exception URLs may contain signed credentials; do not print them.
            if isinstance(error, urllib.error.HTTPError):
                print(f'Failed: upstream HTTP {error.code}; check account access and retry.')
            else:
                print(f'Failed: {error}')
    print(f'\n{len(entries) - failures}/{len(entries)} models ready. Restart ComfyUI after installing.')
    return 1 if failures else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError) as error:
        print(f'Installer stopped: {error}', file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print('\nCancelled. Run again to skip completed models.', file=sys.stderr)
        sys.exit(130)

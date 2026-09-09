"""Local text-file library for prompt wildcard authoring."""
import os
from pathlib import Path
import re
import tempfile

from aiohttp import web
import folder_paths
from server import PromptServer

from .donut_prompt import MAX_TEXT, expand_text, wildcard_roots


def wildcard_path(name):
    if not isinstance(name, str) or not re.fullmatch(r'[a-zA-Z][a-zA-Z0-9_/-]*', name) or '..' in name or '//' in name:
        raise ValueError('Use a name such as haircolor or clothes/shirt.')
    root = (Path(folder_paths.get_user_directory()) / 'wildcards').resolve()
    path = (root / (name + '.txt')).resolve()
    if not path.is_relative_to(root):
        raise ValueError('The wildcard must stay inside the wildcard folder.')
    return path


def list_wildcards():
    files = {}
    for root in wildcard_roots():
        if not root.is_dir():
            continue
        for path in sorted(root.rglob('*.txt')):
            if path.is_file() and path.resolve().is_relative_to(root):
                name = path.relative_to(root).with_suffix('').as_posix()
                if re.fullmatch(r'[a-zA-Z][a-zA-Z0-9_/-]*', name):
                    files.setdefault(name, path)
    return files


def save_wildcard(name, text):
    path = wildcard_path(name)
    if not isinstance(text, str) or len(text.encode('utf-8')) > MAX_TEXT:
        raise ValueError('Keep the wildcard file below 1 MB.')
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError('Add at least one option, one per line.')
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as output:
            output.write('\n'.join(lines) + '\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


@PromptServer.instance.routes.get('/donut/wildcards')
async def catalog(request):
    return web.json_response({'names': sorted(list_wildcards()),
                              'directory': str(Path(folder_paths.get_user_directory()) / 'wildcards')})


@PromptServer.instance.routes.get('/donut/wildcards/file')
async def read_file(request):
    name = request.query.get('name', '')
    try:
        wildcard_path(name)
        path = list_wildcards().get(name)
        if path is None:
            raise web.HTTPNotFound(text='Wildcard not found.')
        if path.stat().st_size > MAX_TEXT:
            raise ValueError('This wildcard file exceeds 1 MB.')
        return web.json_response({'name': name, 'text': path.read_text(encoding='utf-8-sig')})
    except ValueError as error:
        raise web.HTTPBadRequest(text=str(error)) from None


@PromptServer.instance.routes.post('/donut/wildcards/file')
async def write_file(request):
    data = await request.json()
    try:
        path = save_wildcard(data.get('name'), data.get('text'))
        return web.json_response({'name': data['name'], 'path': str(path)})
    except ValueError as error:
        raise web.HTTPBadRequest(text=str(error)) from None


@PromptServer.instance.routes.post('/donut/wildcards/preview')
async def preview(request):
    data = await request.json()
    try:
        return web.json_response({'text': expand_text(data.get('text', ''), int(data.get('seed', 0)))})
    except (ValueError, TypeError) as error:
        raise web.HTTPBadRequest(text=str(error)) from None

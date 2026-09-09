"""Download workflow models only from the catalog shipped with DonutNodes."""
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import threading
from urllib.parse import urljoin, urlsplit
import uuid

from aiohttp import web
import folder_paths
import requests
from server import PromptServer

from .shared.config import get_civitai_api_key


CATALOG_PATH = Path(__file__).with_name("model_sources.json")
FOLDERS = {"diffusion_models", "checkpoints", "text_encoders", "vae", "loras",
           "upscale_models", "ultralytics", "sams", "controlnet", "clip_vision", "embeddings"}
DOWNLOAD_HOSTS = ("huggingface.co", "hf.co", "civitai.com", "civitai.red",
                  "civitai.green", "dl.fbaipublicfiles.com", "github.com", "githubusercontent.com",
                  "civitai-delivery-worker-prod.5ac0637cfd0766c97916cefa3764fbdf.r2.cloudflarestorage.com")
CHUNK_SIZE = 4 * 1024 * 1024


def model_name(value):
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise ValueError("Invalid model filename.")
    name = value.replace("\\", "/")
    if name.startswith("/") or ":" in name or any(part in ("", ".", "..") for part in name.split("/")):
        raise ValueError("Model filenames must stay inside their model folder.")
    return name


def download_url(value):
    parsed = urlsplit(value)
    host = parsed.hostname or ""
    if (parsed.scheme != "https" or parsed.username or parsed.password or parsed.port not in (None, 443)
            or not any(host == suffix or host.endswith("." + suffix) for suffix in DOWNLOAD_HOSTS)):
        raise ValueError("The download redirected outside supported upstream hosts.")
    return value


def load_catalog():
    data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if data.get("version") != 1:
        raise ValueError("Unsupported Donut model catalog version.")
    entries = data["models"]
    seen = set()
    for entry in entries:
        name = model_name(entry["filename"])
        key = (entry["folder"], name)
        if entry["folder"] not in FOLDERS or key in seen:
            raise ValueError("Invalid or duplicate model catalog entry.")
        seen.add(key)
        if not re.fullmatch(r"[a-fA-F0-9]{64}", entry["sha256"]) or entry["size"] <= 0:
            raise ValueError("A catalog model needs its exact SHA-256 and byte size.")
        download_url(entry["url"])
    return entries


def catalog_entry(catalog, folder, name):
    exact = [entry for entry in catalog if entry["folder"] == folder and entry["filename"] == name]
    if exact:
        return exact[0]
    # Workflows commonly move the same upstream filename into a subfolder.
    matches = [entry for entry in catalog if entry["folder"] == folder
               and PurePosixPath(entry["filename"]).name == PurePosixPath(name).name]
    return matches[0] if len(matches) == 1 else None


class Cancelled(Exception):
    pass


class ModelDownloads:
    def __init__(self):
        self.lock = threading.Lock()
        self.cancelled = threading.Event()
        self.job = {"id": None, "running": False, "state": "idle", "results": []}
        self.hashes = {}

    def update(self, **values):
        with self.lock:
            self.job.update(values)

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.job)

    def start(self, references):
        if not isinstance(references, list) or not 1 <= len(references) <= 2048:
            raise ValueError("No model loaders found in this workflow.")
        refs = []
        for reference in references:
            if not isinstance(reference, dict) or reference.get("folder") not in FOLDERS:
                raise ValueError("Unsupported model folder.")
            ref = {"folder": reference["folder"], "name": model_name(reference.get("name"))}
            if ref not in refs:
                refs.append(ref)
        catalog = load_catalog()
        with self.lock:
            if self.job["running"]:
                return copy.deepcopy(self.job)
            self.cancelled.clear()
            self.job = {"id": uuid.uuid4().hex, "running": True, "state": "checking",
                        "total": len(refs), "completed": 0, "current": "", "bytes": 0,
                        "total_bytes": 0, "results": []}
            result = copy.deepcopy(self.job)
        threading.Thread(target=self.run, args=(refs, catalog), daemon=True).start()
        return result

    def check_cancelled(self):
        if self.cancelled.is_set():
            raise Cancelled()

    @staticmethod
    def fingerprint(path):
        stat = path.stat()
        return [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_dev, stat.st_ino]

    def digest(self, path):
        key, stamp = str(path.resolve()), self.fingerprint(path)
        cached = self.hashes.get(key)
        if cached and cached.get("stamp") == stamp:
            return cached["sha256"]
        self.update(state="checking", bytes=0, total_bytes=stamp[0])
        digest, count = hashlib.sha256(), 0
        with path.open("rb") as handle:
            while chunk := handle.read(CHUNK_SIZE):
                self.check_cancelled()
                digest.update(chunk)
                count += len(chunk)
                self.update(bytes=count)
        if self.fingerprint(path) != stamp:
            raise ValueError("The model changed while checking it. Retry Download missing.")
        self.hashes[key] = {"stamp": stamp, "sha256": digest.hexdigest()}
        return digest.hexdigest()

    def find_model(self, entry, requested_name):
        folder = entry["folder"]
        names = list(dict.fromkeys([requested_name, entry["filename"]] + folder_paths.get_filename_list(folder)))
        conflicts = []
        for name in names:
            self.check_cancelled()
            full = folder_paths.get_full_path(folder, name)
            if full is None:
                continue
            path = Path(full)
            if path.stat().st_size == entry["size"] and self.digest(path).lower() == entry["sha256"].lower():
                return name
            if name in (requested_name, entry["filename"]):
                conflicts.append(name)
        if conflicts:
            raise ValueError("Different file/hash already exists: " + conflicts[0] + ". Existing file kept.")
        return None

    @staticmethod
    def destination(entry):
        roots = folder_paths.get_folder_paths(entry["folder"])
        if not roots:
            raise ValueError("The model loader has not registered its destination folder.")
        root = Path(roots[0])
        # Prefer the modern built-in directory over its legacy alias, but keep
        # custom/default roots placed first by extra_model_paths.yaml.
        standard = Path(folder_paths.models_dir)
        if root in (standard / "unet", standard / "clip"):
            modern = standard / entry["folder"]
            if str(modern) in roots:
                root = modern
        root = root.resolve()
        target = root / model_name(entry["filename"])
        if not target.resolve().is_relative_to(root):
            raise ValueError("The model destination escapes its configured folder.")
        return target

    def fetch(self, entry):
        target = self.destination(entry)
        if os.path.lexists(target):
            raise ValueError("A file already exists at the download destination. Existing file kept.")
        target.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(target.parent).free < entry["size"]:
            raise ValueError("Not enough free space in the model folder.")
        current = entry["url"]
        origin = urlsplit(current).hostname
        token = get_civitai_api_key() if origin in ("civitai.com", "civitai.red") else (
            os.environ.get("HF_TOKEN", "") if origin == "huggingface.co" else "")
        response = None
        try:
            for _ in range(8):
                self.check_cancelled()
                headers = {"User-Agent": "DonutNodes model downloader", "Accept-Encoding": "identity"}
                if token and urlsplit(current).hostname == origin:
                    headers["Authorization"] = "Bearer " + token
                response = requests.get(download_url(current), headers=headers, stream=True,
                                        timeout=(15, 60), allow_redirects=False)
                if not response.is_redirect:
                    break
                location = response.headers.get("Location", "")
                response.close()
                current = download_url(urljoin(current, location))
            else:
                raise ValueError("Too many upstream redirects.")
            if response.status_code in (401, 403):
                raise ValueError("Upstream requires access. Set your local Civitai key or HF_TOKEN, then retry.")
            if response.status_code != 200:
                raise ValueError(f"Upstream returned HTTP {response.status_code}.")
            self.update(state="downloading", bytes=0, total_bytes=entry["size"])
            descriptor, temporary = tempfile.mkstemp(prefix=".donut-", suffix=".part", dir=target.parent)
            try:
                digest, count = hashlib.sha256(), 0
                with os.fdopen(descriptor, "wb") as output:
                    for chunk in response.iter_content(CHUNK_SIZE):
                        self.check_cancelled()
                        count += len(chunk)
                        if count > entry["size"]:
                            raise ValueError("Upstream file is larger than the catalog entry.")
                        output.write(chunk)
                        digest.update(chunk)
                        self.update(bytes=count)
                if count != entry["size"] or digest.hexdigest().lower() != entry["sha256"].lower():
                    raise ValueError("Download failed SHA-256 verification. Partial file removed.")
                self.check_cancelled()
                # Publish the verified file atomically without replacing a file
                # created by another process while we were downloading.
                os.link(temporary, target)
            finally:
                Path(temporary).unlink(missing_ok=True)
            self.hashes[str(target.resolve())] = {"stamp": self.fingerprint(target), "sha256": digest.hexdigest()}
            folder_paths.filename_list_cache.pop(entry["folder"], None)
            return entry["filename"]
        finally:
            if response is not None:
                response.close()

    def run(self, references, catalog):
        cache = Path(folder_paths.get_user_directory()) / "donutnodes" / "model_hashes.json"
        try:
            if cache.is_file():
                try:
                    self.hashes = json.loads(cache.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    self.hashes = {}
            for index, ref in enumerate(references):
                self.check_cancelled()
                self.update(current=ref["name"], state="checking", bytes=0, total_bytes=0)
                result = dict(ref)
                try:
                    if ref["folder"] not in folder_paths.folder_names_and_paths:
                        raise ValueError("Install the loader that provides the " + ref["folder"] + " folder.")
                    entry = catalog_entry(catalog, ref["folder"], ref["name"])
                    if entry is None:
                        if folder_paths.get_full_path(ref["folder"], ref["name"]):
                            result.update(status="unlisted", resolved_name=ref["name"])
                        else:
                            raise ValueError("No upstream source is provided in the DonutNodes catalog.")
                    else:
                        found = self.find_model(entry, ref["name"])
                        result.update(status="verified" if found else "downloaded",
                                      resolved_name=found or self.fetch(entry))
                except (OSError, ValueError, requests.RequestException) as error:
                    message = "Network error. Retry Download missing." if isinstance(error, requests.RequestException) else str(error)
                    result.update(status="error", message=message)
                with self.lock:
                    self.job["results"].append(result)
                    self.job["completed"] = index + 1
            self.update(state="complete")
        except Cancelled:
            self.update(state="cancelled")
        except Exception:
            self.update(state="error", message="Model check failed. See the ComfyUI terminal.")
            raise
        finally:
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache.with_suffix(".tmp")
                temporary.write_text(json.dumps(self.hashes), encoding="utf-8")
                os.replace(temporary, cache)
            except OSError:
                pass  # A read-only user directory only disables the hash cache.
            self.update(running=False)


downloads = ModelDownloads()


@PromptServer.instance.routes.get("/donut/models/status")
async def status(request):
    return web.json_response(downloads.snapshot())


@PromptServer.instance.routes.post("/donut/models/download")
async def start(request):
    try:
        data = await request.json()
        if not isinstance(data, dict) or set(data) != {"models"}:
            raise ValueError("Send model names only. Download URLs come from the repository catalog.")
        return web.json_response(downloads.start(data["models"]))
    except (ValueError, KeyError, TypeError) as error:
        raise web.HTTPBadRequest(text=str(error)) from None


@PromptServer.instance.routes.post("/donut/models/cancel")
async def cancel(request):
    data = await request.json()
    with downloads.lock:
        if data.get("id") == downloads.job["id"] and downloads.job["running"]:
            downloads.cancelled.set()
    return web.json_response(downloads.snapshot())

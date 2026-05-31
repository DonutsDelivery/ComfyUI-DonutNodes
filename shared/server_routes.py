"""
DonutNodes server-side API routes.

Provides endpoints for configuration management and LoRA browser
that can be called from the JavaScript frontend.
"""

import json
import os
import hashlib
import urllib.request
from aiohttp import web
from pathlib import Path

try:
    from server import PromptServer
    HAS_SERVER = True
except ImportError:
    HAS_SERVER = False

try:
    import folder_paths
    HAS_FOLDER_PATHS = True
except ImportError:
    HAS_FOLDER_PATHS = False

from .config import load_config, save_config, get_config

# Import CivitAI cache for LoRA browser
try:
    from .civitai_api import CivitAICache, get_civitai_cache_dir, search_models, get_model_by_id, civitai_urlopen
    HAS_CIVITAI = True
except ImportError:
    HAS_CIVITAI = False

# Import download manager
try:
    from .civitai_download import get_downloader, get_download_path, normalize_base_model, invalidate_folder_cache, MODEL_TYPE_TO_FOLDER
    HAS_DOWNLOADER = True
except ImportError:
    HAS_DOWNLOADER = False

# Import hash computation
try:
    from .lora_hash import get_or_compute_hash, get_cached_hash
    HAS_LORA_HASH = True
except ImportError:
    HAS_LORA_HASH = False


def find_file_by_hash(sha256: str, folder_types: list = None) -> dict:
    """
    Search for a file with the given SHA256 hash across model folders.

    Args:
        sha256: The SHA256 hash to search for (case-insensitive)
        folder_types: List of folder types to search (e.g., ["loras", "checkpoints"])
                     If None, searches loras, checkpoints, embeddings, controlnet, vae, upscale_models

    Returns:
        dict with 'found', 'filename', 'full_path', 'folder_type' if found, else {'found': False}
    """
    if not HAS_FOLDER_PATHS or not HAS_LORA_HASH:
        return {"found": False, "error": "Required modules not available"}

    sha256 = sha256.upper()

    if folder_types is None:
        folder_types = ["loras", "checkpoints", "embeddings", "controlnet", "vae", "upscale_models"]

    extensions = ('.safetensors', '.pt', '.ckpt', '.bin', '.pth')

    for folder_type in folder_types:
        try:
            paths = folder_paths.get_folder_paths(folder_type)
        except:
            continue

        for base_dir in paths:
            if not os.path.exists(base_dir):
                continue
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    if file.endswith(extensions):
                        full_path = os.path.join(root, file)
                        cached = get_cached_hash(full_path)
                        if cached and "SHA256" in cached:
                            if cached["SHA256"].upper() == sha256:
                                rel_path = os.path.relpath(full_path, base_dir)
                                return {
                                    "found": True,
                                    "filename": rel_path,
                                    "full_path": full_path,
                                    "folder_type": folder_type
                                }

    return {"found": False}


def register_routes():
    """Register API routes with ComfyUI's server."""
    if not HAS_SERVER:
        print("[DonutNodes] Server not available, skipping route registration")
        return

    routes = PromptServer.instance.routes

    @routes.get('/donut/config')
    async def get_donut_config(request):
        """Return current DonutNodes configuration."""
        config = load_config(force_reload=True)
        return web.json_response(config)

    @routes.post('/donut/config/civitai_api_key')
    async def set_civitai_api_key(request):
        """Save CivitAI API key to config."""
        try:
            data = await request.json()
            api_key = data.get("api_key", "")

            config = load_config(force_reload=True)
            if "civitai" not in config:
                config["civitai"] = {}
            config["civitai"]["api_key"] = api_key

            if save_config(config):
                return web.json_response({"status": "ok"})
            else:
                return web.json_response({"status": "error", "message": "Failed to save config"}, status=500)
        except Exception as e:
            return web.json_response({"status": "error", "message": str(e)}, status=500)

    @routes.post('/donut/config/civitai_auto_lookup')
    async def set_civitai_auto_lookup(request):
        """Save CivitAI auto-lookup setting."""
        try:
            data = await request.json()
            auto_lookup = data.get("auto_lookup", True)

            config = load_config(force_reload=True)
            if "civitai" not in config:
                config["civitai"] = {}
            config["civitai"]["auto_lookup"] = bool(auto_lookup)

            if save_config(config):
                return web.json_response({"status": "ok"})
            else:
                return web.json_response({"status": "error", "message": "Failed to save config"}, status=500)
        except Exception as e:
            return web.json_response({"status": "error", "message": str(e)}, status=500)

    @routes.post('/donut/config/civitai_download_previews')
    async def set_civitai_download_previews(request):
        """Save CivitAI download previews setting."""
        try:
            data = await request.json()
            download_previews = data.get("download_previews", True)

            config = load_config(force_reload=True)
            if "civitai" not in config:
                config["civitai"] = {}
            config["civitai"]["download_previews"] = bool(download_previews)

            if save_config(config):
                return web.json_response({"status": "ok"})
            else:
                return web.json_response({"status": "error", "message": "Failed to save config"}, status=500)
        except Exception as e:
            return web.json_response({"status": "error", "message": str(e)}, status=500)

    @routes.post('/donut/config/civitai_prefer_sfw')
    async def set_civitai_prefer_sfw(request):
        """Save CivitAI prefer SFW setting."""
        try:
            data = await request.json()
            prefer_sfw = data.get("prefer_sfw", True)

            config = load_config(force_reload=True)
            if "civitai" not in config:
                config["civitai"] = {}
            config["civitai"]["prefer_sfw"] = bool(prefer_sfw)

            if save_config(config):
                return web.json_response({"status": "ok"})
            else:
                return web.json_response({"status": "error", "message": "Failed to save config"}, status=500)
        except Exception as e:
            return web.json_response({"status": "error", "message": str(e)}, status=500)

    # ============================================
    # LoRA Browser Routes
    # ============================================

    def compute_lora_hash_for_lookup(filepath: str) -> str:
        """Get hash of a LoRA file for CivitAI lookup.

        Uses full SHA256 hash (first 10 chars) which is what CivitAI accepts.
        The hash is cached for subsequent lookups.
        """
        if HAS_LORA_HASH:
            try:
                # Full SHA256, cached for speed on subsequent lookups
                full_hash = get_or_compute_hash(filepath, hash_type="SHA256", use_cache=True)
                return full_hash[:10]  # CivitAI accepts first 10 chars
            except Exception as e:
                print(f"[DonutNodes] Error computing hash: {e}")
                return ""
        else:
            # Fallback: compute full SHA256 manually
            sha256 = hashlib.sha256()
            try:
                with open(filepath, 'rb') as f:
                    for chunk in iter(lambda: f.read(8192), b''):
                        sha256.update(chunk)
                return sha256.hexdigest()[:10].upper()
            except:
                return ""

    @routes.get('/donut/loras/list')
    async def list_loras(request):
        """List all available LoRAs with basic info.

        Query params:
            include_meta: If "true", include cached CivitAI metadata (uses cached hashes only)
        """
        if not HAS_FOLDER_PATHS:
            return web.json_response({"error": "folder_paths not available"}, status=500)

        include_meta = request.query.get("include_meta", "false").lower() == "true"

        # Get cache if needed
        cache = None
        if include_meta and HAS_CIVITAI:
            cache = CivitAICache()

        try:
            lora_paths = folder_paths.get_folder_paths("loras")
            loras = []

            for lora_dir in lora_paths:
                if not os.path.exists(lora_dir):
                    continue
                for root, dirs, files in os.walk(lora_dir):
                    for file in files:
                        if file.endswith(('.safetensors', '.pt', '.ckpt')):
                            full_path = os.path.join(root, file)
                            rel_path = os.path.relpath(full_path, lora_dir)

                            lora_data = {
                                "name": rel_path,
                                "filename": file,
                                "full_path": full_path
                            }

                            # Add cached metadata if requested (only use cached hash, don't compute)
                            if include_meta and cache and HAS_LORA_HASH:
                                from .lora_hash import get_cached_hash
                                cached_hashes = get_cached_hash(full_path)
                                if cached_hashes and "SHA256" in cached_hashes:
                                    full_sha256 = cached_hashes["SHA256"]
                                    file_hash = full_sha256[:10]
                                    lora_data["hash"] = file_hash
                                    lora_data["sha256"] = full_sha256  # Full hash for deletion
                                    info = cache.get_cached_info(file_hash)
                                    if info:
                                        lora_data["civitai_name"] = info.model_name
                                        lora_data["civitai_version"] = info.version_name
                                        lora_data["base_model"] = info.base_model
                                        lora_data["has_preview"] = True
                                        # Add model/version IDs for update checking
                                        lora_data["model_id"] = info.model_id
                                        lora_data["version_id"] = info.model_version_id
                                    else:
                                        lora_data["has_preview"] = False
                                else:
                                    # No cached hash yet
                                    lora_data["has_preview"] = False

                            loras.append(lora_data)

            # Sort alphabetically
            loras.sort(key=lambda x: x["name"].lower())

            return web.json_response({"loras": loras, "count": len(loras)})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    @routes.get('/donut/loras/info')
    async def get_lora_info(request):
        """Get CivitAI info for a specific LoRA, fetching if needed."""
        if not HAS_FOLDER_PATHS or not HAS_CIVITAI:
            return web.json_response({"error": "Required modules not available"}, status=500)

        lora_name = request.query.get("name", "")
        if not lora_name:
            return web.json_response({"error": "No lora name provided"}, status=400)

        try:
            # Find the LoRA file
            lora_path = folder_paths.get_full_path("loras", lora_name)
            if not lora_path or not os.path.exists(lora_path):
                return web.json_response({"error": "LoRA not found"}, status=404)

            # Get hash
            file_hash = compute_lora_hash_for_lookup(lora_path)
            if not file_hash:
                return web.json_response({
                    "name": lora_name,
                    "hash": None,
                    "civitai": None,
                    "error": "Could not compute hash"
                })

            # Get cache
            cache = CivitAICache()  # Uses default cache dir if none configured

            # Try to get cached info first
            info = cache.get_cached_info(file_hash)

            # If not cached, fetch from CivitAI
            if info is None:
                info = cache.get_or_fetch_info(file_hash, download_preview=True)

            if info:
                # Get preview images using cache helper methods
                hash_prefix = file_hash[:10]  # AutoV2 format
                preview_paths = []
                for i in range(4):
                    img_path = cache._get_image_path(file_hash, 'jpg', i)
                    if img_path.exists():
                        preview_paths.append(str(img_path))
                    else:
                        # Try other extensions
                        for ext in ['png', 'webp']:
                            img_path = cache._get_image_path(file_hash, ext, i)
                            if img_path.exists():
                                preview_paths.append(str(img_path))
                                break

                # Check for collage
                collage_path = cache.get_preview_collage_path(file_hash)
                if not collage_path:
                    # Create collage if doesn't exist
                    cache.create_preview_collage(info, max_images=4)
                    collage_path = cache.get_preview_collage_path(file_hash)

                return web.json_response({
                    "name": lora_name,
                    "hash": hash_prefix,
                    "civitai": info.to_dict(),
                    "preview_count": len(preview_paths),
                    "has_collage": collage_path is not None
                })
            else:
                return web.json_response({
                    "name": lora_name,
                    "hash": file_hash[:10],
                    "civitai": None,
                    "error": "Not found on CivitAI"
                })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return web.json_response({"error": str(e)}, status=500)

    @routes.get('/donut/loras/preview')
    async def get_lora_preview(request):
        """Get preview image for a LoRA."""
        if not HAS_CIVITAI:
            return web.json_response({"error": "CivitAI module not available"}, status=500)

        hash_prefix = request.query.get("hash", "")
        image_type = request.query.get("type", "collage")  # collage, 0, 1, 2, 3

        if not hash_prefix:
            return web.json_response({"error": "No hash provided"}, status=400)

        try:
            cache = CivitAICache()  # Uses default cache dir if none configured
            images_dir = cache.images_dir

            # Find the image
            if image_type == "collage":
                img_path = images_dir / f"{hash_prefix}_collage.jpg"
            elif image_type == "0":
                # Try different extensions
                for ext in ['jpg', 'png', 'webp']:
                    img_path = images_dir / f"{hash_prefix}.{ext}"
                    if img_path.exists():
                        break
            else:
                for ext in ['jpg', 'png', 'webp']:
                    img_path = images_dir / f"{hash_prefix}_{image_type}.{ext}"
                    if img_path.exists():
                        break

            if img_path.exists():
                # Return the image
                with open(img_path, 'rb') as f:
                    img_data = f.read()

                content_type = 'image/jpeg'
                if str(img_path).endswith('.png'):
                    content_type = 'image/png'
                elif str(img_path).endswith('.webp'):
                    content_type = 'image/webp'

                return web.Response(body=img_data, content_type=content_type)
            else:
                return web.json_response({"error": "Image not found"}, status=404)

        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    # ============================================
    # CivitAI Browser/Search Routes
    # ============================================

    @routes.get('/donut/civitai/search')
    async def civitai_search(request):
        """Search CivitAI models."""
        if not HAS_CIVITAI:
            return web.json_response({"error": "CivitAI module not available"}, status=500)

        try:
            # Parse query parameters
            query = request.query.get("query", "")
            # Handle multiple types (can be repeated params or comma-separated)
            types_list = request.query.getall("types", [])
            if not types_list:
                types_single = request.query.get("types", "")
                types_list = [t.strip() for t in types_single.split(",") if t.strip()] if types_single else None
            else:
                types_list = [t for t in types_list if t.strip()]
                types_list = types_list if types_list else None
            sort = request.query.get("sort", "Highest Rated")
            period = request.query.get("period", "AllTime")
            nsfw = request.query.get("nsfw", "false").lower() == "true"
            # Handle multiple baseModels (can be repeated params or comma-separated)
            base_models_list = request.query.getall("baseModels", [])
            if not base_models_list:
                base_models_single = request.query.get("baseModels", "")
                base_models_list = [b.strip() for b in base_models_single.split(",") if b.strip()] if base_models_single else None
            else:
                base_models_list = [b for b in base_models_list if b.strip()]
                base_models_list = base_models_list if base_models_list else None
            limit = int(request.query.get("limit", "20"))
            page = int(request.query.get("page", "1"))
            cursor = request.query.get("cursor", "")  # Cursor for pagination
            tag = request.query.get("tag", "")
            username = request.query.get("username", "")

            print(f"[CivitAI Search] query={query}, types={types_list}, page={page}, cursor={cursor[:20] if cursor else 'None'}, nsfw={nsfw}, baseModels={base_models_list}")

            # Get API key from config
            config = load_config()
            api_key = config.get("civitai", {}).get("api_key")

            # Perform search
            result = search_models(
                query=query,
                types=types_list,
                sort=sort,
                period=period,
                nsfw=nsfw,
                base_models=base_models_list,
                limit=limit,
                page=page,
                cursor=cursor if cursor else None,
                api_key=api_key,
                tag=tag if tag else None,
                username=username if username else None
            )

            if result:
                return web.json_response(result)
            else:
                return web.json_response({"items": [], "metadata": {"totalItems": 0, "currentPage": 1, "pageSize": limit}})

        except Exception as e:
            import traceback
            traceback.print_exc()
            return web.json_response({"error": str(e)}, status=500)

    @routes.get('/donut/civitai/model/{model_id}')
    async def civitai_get_model(request):
        """Get detailed model info by ID."""
        if not HAS_CIVITAI:
            return web.json_response({"error": "CivitAI module not available"}, status=500)

        try:
            model_id = int(request.match_info['model_id'])

            # Get API key from config
            config = load_config()
            api_key = config.get("civitai", {}).get("api_key")

            result = get_model_by_id(model_id, api_key=api_key)

            if result:
                return web.json_response(result)
            else:
                return web.json_response({"error": "Model not found"}, status=404)

        except ValueError:
            return web.json_response({"error": "Invalid model ID"}, status=400)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    @routes.post('/donut/civitai/check-updates')
    async def civitai_check_updates(request):
        """Check if local models have updates available on CivitAI.

        Accepts a list of {model_id, sha256} pairs and returns
        which ones have newer versions available (by comparing SHA256 hashes).

        Also accepts an optional all_local_hashes list to check if the latest
        version is already downloaded under a different filename.
        """
        if not HAS_CIVITAI:
            return web.json_response({"error": "CivitAI module not available"}, status=500)

        try:
            data = await request.json()
            models_to_check = data.get("models", [])
            # Set of ALL local hashes - to check if we already have the latest version
            all_local_hashes = set(h.upper() for h in data.get("all_local_hashes", []))

            if not models_to_check:
                return web.json_response({"updates": []})

            # Get API key from config
            config = load_config()
            api_key = config.get("civitai", {}).get("api_key")

            updates = []

            # Check each model (rate-limited by get_model_by_id)
            for item in models_to_check:
                model_id = item.get("model_id")
                local_sha256 = item.get("sha256", "").upper()

                if not model_id or not local_sha256:
                    continue

                try:
                    # Fetch current model info from CivitAI
                    model_data = get_model_by_id(model_id, api_key=api_key)

                    if model_data and model_data.get("modelVersions"):
                        # First version is the latest
                        latest_version = model_data["modelVersions"][0]
                        latest_files = latest_version.get("files", [])

                        if latest_files:
                            latest_sha256 = latest_files[0].get("hashes", {}).get("SHA256", "").upper()

                            # Check if we already have the latest version downloaded
                            # (either this file or any other local file)
                            already_have_latest = (
                                latest_sha256 == local_sha256 or
                                latest_sha256 in all_local_hashes
                            )

                            # Only report update if we don't have the latest anywhere
                            if latest_sha256 and not already_have_latest:
                                updates.append({
                                    "model_id": model_id,
                                    "model_name": model_data.get("name", "Unknown"),
                                    "local_sha256": local_sha256,
                                    "latest_sha256": latest_sha256,
                                    "latest_version_name": latest_version.get("name", ""),
                                    "latest_version_id": latest_version.get("id"),
                                    "latest_base_model": latest_version.get("baseModel", ""),
                                    "latest_download_url": latest_version.get("downloadUrl", ""),
                                    "latest_filename": latest_files[0].get("name", "")
                                })
                except Exception as e:
                    print(f"[CivitAI] Error checking updates for model {model_id}: {e}")
                    continue

            return web.json_response({"updates": updates})

        except Exception as e:
            import traceback
            traceback.print_exc()
            return web.json_response({"error": str(e)}, status=500)

    @routes.post('/donut/civitai/download')
    async def civitai_download(request):
        """Start a model download."""
        if not HAS_DOWNLOADER:
            return web.json_response({"error": "Download module not available"}, status=500)

        try:
            data = await request.json()
            download_url = data.get("downloadUrl")
            model_type = data.get("modelType", "LORA")
            base_model = data.get("baseModel", "")
            filename = data.get("filename", "model.safetensors")
            sha256 = data.get("sha256")  # Pre-known hash from CivitAI
            skip_duplicate_check = data.get("skipDuplicateCheck", False)

            if not download_url:
                return web.json_response({"error": "No download URL provided"}, status=400)

            # Check for existing file with same hash before downloading
            if sha256 and not skip_duplicate_check:
                existing = find_file_by_hash(sha256)
                if existing.get("found"):
                    return web.json_response({
                        "error": "duplicate",
                        "message": f"File already exists: {existing['filename']}",
                        "existingFile": existing["filename"],
                        "existingPath": existing["full_path"],
                        "folderType": existing["folder_type"]
                    }, status=409)  # 409 Conflict

            # Get download path based on model type and base model
            save_path = get_download_path(model_type, base_model, filename)

            # Get API key for authenticated downloads
            config = load_config(force_reload=True)  # Force reload to get latest API key
            api_key = config.get("civitai", {}).get("api_key")
            print(f"[DonutNodes Download] API key present: {bool(api_key)}, URL: {download_url[:50]}...")

            # Start download with model_type for cache invalidation
            downloader = get_downloader()
            download_id = downloader.start_download(
                download_url=download_url,
                save_path=save_path,
                api_key=api_key,
                model_type=model_type,
                sha256=sha256  # Pass hash to save after download
            )

            return web.json_response({
                "downloadId": download_id,
                "savePath": save_path,
                "status": "started"
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return web.json_response({"error": str(e)}, status=500)

    @routes.get('/donut/civitai/download/status/{download_id}')
    async def civitai_download_status(request):
        """Get download status."""
        if not HAS_DOWNLOADER:
            return web.json_response({"error": "Download module not available"}, status=500)

        try:
            download_id = request.match_info['download_id']
            downloader = get_downloader()
            status = downloader.get_status(download_id)

            if status:
                return web.json_response({
                    "downloadId": status.download_id,
                    "filename": status.filename,
                    "filepath": status.filepath,
                    "status": status.status,
                    "totalSize": status.total_size,
                    "downloadedSize": status.downloaded_size,
                    "progress": (status.downloaded_size / status.total_size * 100) if status.total_size > 0 else 0,
                    "speedBps": status.speed_bps,
                    "error": status.error
                })
            else:
                return web.json_response({"error": "Download not found"}, status=404)

        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    @routes.get('/donut/civitai/downloads')
    async def civitai_all_downloads(request):
        """Get all download statuses."""
        if not HAS_DOWNLOADER:
            return web.json_response({"error": "Download module not available"}, status=500)

        try:
            downloader = get_downloader()
            all_downloads = downloader.get_all_downloads()

            downloads_list = []
            for download_id, status in all_downloads.items():
                downloads_list.append({
                    "downloadId": status.download_id,
                    "filename": status.filename,
                    "filepath": status.filepath,
                    "status": status.status,
                    "totalSize": status.total_size,
                    "downloadedSize": status.downloaded_size,
                    "progress": (status.downloaded_size / status.total_size * 100) if status.total_size > 0 else 0,
                    "speedBps": status.speed_bps,
                    "error": status.error
                })

            return web.json_response({"downloads": downloads_list})

        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    @routes.post('/donut/civitai/download/cancel/{download_id}')
    async def civitai_cancel_download(request):
        """Cancel a download."""
        if not HAS_DOWNLOADER:
            return web.json_response({"error": "Download module not available"}, status=500)

        try:
            download_id = request.match_info['download_id']
            downloader = get_downloader()

            if downloader.cancel_download(download_id):
                return web.json_response({"status": "cancelled"})
            else:
                return web.json_response({"error": "Download not found or already completed"}, status=400)

        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    @routes.get('/donut/civitai/image')
    async def civitai_proxy_image(request):
        """Proxy CivitAI images to avoid CORS issues."""
        image_url = request.query.get("url", "")
        if not image_url:
            return web.json_response({"error": "No URL provided"}, status=400)

        try:
            headers = {
                "User-Agent": "ComfyUI-DonutNodes/1.0"
            }
            req = urllib.request.Request(image_url, headers=headers)
            with civitai_urlopen(req, timeout=10) as response:
                content_type = response.headers.get('Content-Type', 'image/jpeg')
                data = response.read()
                return web.Response(body=data, content_type=content_type)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    @routes.post('/donut/refresh_folder_cache')
    async def refresh_folder_cache(request):
        """
        Refresh the folder_paths cache to make newly downloaded files visible.
        This should be called after a download completes to update node dropdowns.
        """
        try:
            data = await request.json()
            folder_name = data.get("folder", None)  # Optional: specific folder to refresh

            if HAS_DOWNLOADER:
                invalidate_folder_cache(folder_name)

            # Also directly clear folder_paths cache if available
            if HAS_FOLDER_PATHS:
                if hasattr(folder_paths, 'filename_list_cache'):
                    if folder_name:
                        if folder_name in folder_paths.filename_list_cache:
                            del folder_paths.filename_list_cache[folder_name]
                    else:
                        folder_paths.filename_list_cache.clear()

            return web.json_response({
                "status": "ok",
                "message": f"Cache refreshed for: {folder_name or 'all folders'}"
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    @routes.get('/donut/loras/hashes')
    async def get_local_lora_hashes(request):
        """
        Get all SHA256 hashes from locally downloaded LoRAs.

        Returns a set of hashes that can be used to check if a CivitAI model
        has already been downloaded. Only returns cached hashes (fast).
        """
        if not HAS_FOLDER_PATHS:
            return web.json_response({"error": "folder_paths not available"}, status=500)

        if not HAS_LORA_HASH:
            return web.json_response({"error": "lora_hash module not available"}, status=500)

        try:
            from .lora_hash import get_cached_hash

            lora_paths = folder_paths.get_folder_paths("loras")
            hashes = set()

            for lora_dir in lora_paths:
                if not os.path.exists(lora_dir):
                    continue
                for root, dirs, files in os.walk(lora_dir):
                    for file in files:
                        if file.endswith(('.safetensors', '.pt', '.ckpt')):
                            full_path = os.path.join(root, file)
                            cached = get_cached_hash(full_path)
                            if cached and "SHA256" in cached:
                                # Store full SHA256 (uppercase) for comparison with CivitAI
                                hashes.add(cached["SHA256"].upper())

            return web.json_response({
                "hashes": list(hashes),
                "count": len(hashes)
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return web.json_response({"error": str(e)}, status=500)

    @routes.get('/donut/loras/duplicates')
    async def find_duplicate_loras(request):
        """
        Find duplicate LoRA files by scanning for identical SHA256 hashes.
        Returns groups of files that share the same hash.
        """
        if not HAS_FOLDER_PATHS:
            return web.json_response({"error": "folder_paths not available"}, status=500)

        if not HAS_LORA_HASH:
            return web.json_response({"error": "lora_hash module not available"}, status=500)

        try:
            from .lora_hash import get_cached_hash

            lora_paths = folder_paths.get_folder_paths("loras")

            # Map hash -> list of files
            hash_to_files = {}

            for lora_dir in lora_paths:
                if not os.path.exists(lora_dir):
                    continue
                for root, dirs, files in os.walk(lora_dir):
                    for file in files:
                        if file.endswith(('.safetensors', '.pt', '.ckpt')):
                            full_path = os.path.join(root, file)
                            cached = get_cached_hash(full_path)
                            if cached and "SHA256" in cached:
                                sha256 = cached["SHA256"].upper()
                                rel_path = os.path.relpath(full_path, lora_dir)
                                file_size = os.path.getsize(full_path)

                                if sha256 not in hash_to_files:
                                    hash_to_files[sha256] = []

                                hash_to_files[sha256].append({
                                    "filename": rel_path,
                                    "full_path": full_path,
                                    "size": file_size
                                })

            # Filter to only groups with duplicates (2+ files)
            duplicates = []
            for sha256, files in hash_to_files.items():
                if len(files) > 1:
                    # Sort by path length (shorter = likely original)
                    files.sort(key=lambda f: len(f["filename"]))
                    duplicates.append({
                        "sha256": sha256,
                        "files": files,
                        "count": len(files)
                    })

            # Sort by count descending
            duplicates.sort(key=lambda d: d["count"], reverse=True)

            total_wasted = sum(
                sum(f["size"] for f in d["files"][1:])  # All but first file
                for d in duplicates
            )

            return web.json_response({
                "duplicates": duplicates,
                "totalGroups": len(duplicates),
                "totalDuplicateFiles": sum(d["count"] - 1 for d in duplicates),
                "totalWastedBytes": total_wasted
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return web.json_response({"error": str(e)}, status=500)

    @routes.get('/donut/loras/by-hash')
    async def get_lora_by_hash(request):
        """
        Find a local LoRA file by its SHA256 hash.
        Returns the relative filename that can be used with ComfyUI nodes.
        """
        if not HAS_FOLDER_PATHS:
            return web.json_response({"error": "folder_paths not available"}, status=500)

        if not HAS_LORA_HASH:
            return web.json_response({"error": "lora_hash module not available"}, status=500)

        sha256 = request.query.get("sha256", "").upper()
        if not sha256:
            return web.json_response({"error": "No sha256 provided"}, status=400)

        try:
            from .lora_hash import get_cached_hash

            lora_paths = folder_paths.get_folder_paths("loras")

            for lora_dir in lora_paths:
                if not os.path.exists(lora_dir):
                    continue
                for root, dirs, files in os.walk(lora_dir):
                    for file in files:
                        if file.endswith(('.safetensors', '.pt', '.ckpt')):
                            full_path = os.path.join(root, file)
                            cached = get_cached_hash(full_path)
                            if cached and "SHA256" in cached:
                                if cached["SHA256"].upper() == sha256:
                                    # Found it! Return relative path
                                    rel_path = os.path.relpath(full_path, lora_dir)
                                    return web.json_response({
                                        "found": True,
                                        "filename": rel_path,
                                        "full_path": full_path
                                    })

            return web.json_response({"found": False})
        except Exception as e:
            import traceback
            traceback.print_exc()
            return web.json_response({"error": str(e)}, status=500)

    @routes.delete('/donut/loras/by-hash')
    async def delete_lora_by_hash(request):
        """
        Delete a local LoRA file by its SHA256 hash.
        Also removes the associated .hash cache file and cached preview images.
        """
        if not HAS_FOLDER_PATHS:
            return web.json_response({"error": "folder_paths not available"}, status=500)

        if not HAS_LORA_HASH:
            return web.json_response({"error": "lora_hash module not available"}, status=500)

        sha256 = request.query.get("sha256", "").upper()
        if not sha256:
            return web.json_response({"error": "No sha256 provided"}, status=400)

        try:
            from .lora_hash import get_cached_hash

            lora_paths = folder_paths.get_folder_paths("loras")

            for lora_dir in lora_paths:
                if not os.path.exists(lora_dir):
                    continue
                for root, dirs, files in os.walk(lora_dir):
                    for file in files:
                        if file.endswith(('.safetensors', '.pt', '.ckpt')):
                            full_path = os.path.join(root, file)
                            cached = get_cached_hash(full_path)
                            if cached and "SHA256" in cached:
                                if cached["SHA256"].upper() == sha256:
                                    # Found it! Delete the file
                                    rel_path = os.path.relpath(full_path, lora_dir)
                                    try:
                                        os.remove(full_path)
                                        # Also remove the hash cache file
                                        hash_file = full_path + ".hash"
                                        if os.path.exists(hash_file):
                                            os.remove(hash_file)

                                        # Delete cached preview images and metadata
                                        hash_prefix = sha256[:10]
                                        cache_dir = Path(__file__).parent.parent / "civitai_cache"
                                        images_dir = cache_dir / "images"
                                        metadata_dir = cache_dir / "metadata"
                                        deleted_cache_files = []

                                        # Delete metadata JSON
                                        for meta_file in [metadata_dir / f"{hash_prefix}.json",
                                                          metadata_dir / f"{sha256[:16]}.json"]:
                                            if meta_file.exists():
                                                meta_file.unlink()
                                                deleted_cache_files.append(str(meta_file.name))

                                        # Delete preview images (single, indexed, and collage)
                                        if images_dir.exists():
                                            for img_file in images_dir.glob(f"{hash_prefix}*"):
                                                img_file.unlink()
                                                deleted_cache_files.append(str(img_file.name))
                                            # Also check legacy 16-char prefix
                                            for img_file in images_dir.glob(f"{sha256[:16]}*"):
                                                img_file.unlink()
                                                deleted_cache_files.append(str(img_file.name))

                                        # Invalidate folder cache
                                        invalidate_folder_cache("loras")
                                        return web.json_response({
                                            "deleted": True,
                                            "filename": rel_path,
                                            "full_path": full_path,
                                            "deleted_cache_files": deleted_cache_files
                                        })
                                    except OSError as e:
                                        return web.json_response({
                                            "error": f"Failed to delete: {e}"
                                        }, status=500)

            return web.json_response({"deleted": False, "error": "File not found"})
        except Exception as e:
            import traceback
            traceback.print_exc()
            return web.json_response({"error": str(e)}, status=500)

    @routes.delete('/donut/loras/by-path')
    async def delete_lora_by_path(request):
        """
        Delete a local LoRA file by its full path.
        Fallback for files without cached hash.
        """
        if not HAS_FOLDER_PATHS:
            return web.json_response({"error": "folder_paths not available"}, status=500)

        file_path = request.query.get("path", "")
        if not file_path:
            return web.json_response({"error": "No path provided"}, status=400)

        # Security check: ensure path is within a loras directory
        lora_paths = folder_paths.get_folder_paths("loras")
        is_valid_path = False
        for lora_dir in lora_paths:
            if file_path.startswith(os.path.abspath(lora_dir)):
                is_valid_path = True
                break

        if not is_valid_path:
            return web.json_response({"error": "Invalid path - not in loras directory"}, status=403)

        if not os.path.exists(file_path):
            return web.json_response({"error": "File not found"}, status=404)

        try:
            filename = os.path.basename(file_path)
            os.remove(file_path)

            # Also remove the hash cache file if it exists
            hash_file = file_path + ".hash"
            if os.path.exists(hash_file):
                os.remove(hash_file)

            # Invalidate folder cache
            invalidate_folder_cache("loras")

            return web.json_response({
                "deleted": True,
                "filename": filename,
                "full_path": file_path
            })
        except OSError as e:
            return web.json_response({"error": f"Failed to delete: {e}"}, status=500)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return web.json_response({"error": str(e)}, status=500)

    @routes.get('/donut/loras/filename')
    async def get_lora_filename(request):
        """
        Get the relative filename for a LoRA given its full path.
        Used by frontend to set node widget values after download.
        """
        if not HAS_FOLDER_PATHS:
            return web.json_response({"error": "folder_paths not available"}, status=500)

        full_path = request.query.get("path", "")
        if not full_path:
            return web.json_response({"error": "No path provided"}, status=400)

        try:
            # Find the relative path from the loras folder
            lora_paths = folder_paths.get_folder_paths("loras")

            for lora_dir in lora_paths:
                if full_path.startswith(lora_dir):
                    rel_path = os.path.relpath(full_path, lora_dir)
                    return web.json_response({
                        "filename": rel_path,
                        "full_path": full_path
                    })

            # If not in any lora path, just return the basename
            return web.json_response({
                "filename": os.path.basename(full_path),
                "full_path": full_path
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    @routes.get('/donut/lora/get_hash')
    async def get_lora_hash(request):
        """Return SHA256 hash for a LoRA, computing on first call. Used by
        the DonutLoRAStack frontend to populate the hidden lora_hash_N
        widget so the workflow file carries hashes for cross-machine resolution."""
        if not HAS_FOLDER_PATHS or not HAS_LORA_HASH:
            return web.json_response({"hash": ""})
        name = request.query.get("name", "")
        if not name or name == "None":
            return web.json_response({"hash": ""})
        path = folder_paths.get_full_path("loras", name)
        if not path or not os.path.exists(path):
            return web.json_response({"hash": "", "error": "not_found"})
        try:
            cache = None
            if HAS_CIVITAI:
                try:
                    from .civitai_api import get_cache as _gc
                    cache = _gc()
                except Exception:
                    cache = None
            hash_cache_dir = str(cache.cache_dir / "hashes") if cache else None
            sha = get_or_compute_hash(path, hash_type="SHA256",
                                      use_cache=True, cache_dir=hash_cache_dir)
            return web.json_response({"hash": sha})
        except Exception as e:
            return web.json_response({"hash": "", "error": str(e)}, status=500)

    # ============================================
    # Prompt Injection Styles Routes
    # ============================================

    @routes.get('/donut/styles/by_category')
    async def get_styles_by_category(request):
        """Return all style categories and their styles for dynamic dropdown (legacy)."""
        try:
            from ..DonutPromptInjection import ALL_STYLES
            # Return category -> list of style names
            result = {}
            for category, styles in ALL_STYLES.items():
                result[category] = list(styles.keys())
            return web.json_response(result)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    @routes.get('/donut/styles/hierarchy')
    async def get_style_hierarchy(request):
        """Return hierarchical style structure for dynamic dropdowns."""
        try:
            from ..DonutPromptInjection import STYLE_HIERARCHY
            # Return main_category -> subcategory -> list of style names
            result = {}
            for main_cat, subcategories in STYLE_HIERARCHY.items():
                result[main_cat] = {}
                for subcat, styles in subcategories.items():
                    result[main_cat][subcat] = list(styles.keys())
            return web.json_response(result)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    @routes.get('/donut/locations/hierarchy')
    async def get_locations_hierarchy(request):
        """Return location categories and their locations for dynamic dropdowns."""
        try:
            from ..DonutPromptInjection import ALL_LOCATIONS
            # Return category -> list of location names
            result = {}
            for category, locations in ALL_LOCATIONS.items():
                result[category] = list(locations.keys())
            return web.json_response(result)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    print("[DonutNodes] Server routes registered")


# Register routes when module is imported
register_routes()

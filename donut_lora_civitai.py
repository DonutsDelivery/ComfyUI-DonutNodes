"""
DonutLoRACivitAI - LoRA stack with CivitAI metadata integration.

Features:
- Automatic hash computation and CivitAI lookup
- Local metadata caching
- Display of official names, descriptions, trigger words
- Preview image support
"""

import os
import folder_paths
from typing import Optional, Dict, Any, Tuple, List

from .shared.lora_hash import get_or_compute_hash, compute_sha256
from .shared.civitai_api import (
    CivitAICache,
    CivitAIModelInfo,
    get_cache,
    fetch_model_by_hash
)


class DonutLoRACivitAIInfo:
    """
    Fetch and display CivitAI information for a LoRA file.

    This node computes the hash of a LoRA file and looks up its metadata
    on CivitAI, caching the results locally for future use.
    """

    class_type = "CUSTOM"
    aux_id = "DonutsDelivery/ComfyUI-DonutNodes"

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "lora_name": (loras, {"tooltip": "Select a LoRA to look up on CivitAI"}),
            },
            "optional": {
                "api_key": ("STRING", {
                    "default": "",
                    "placeholder": "Optional CivitAI API key",
                    "tooltip": "API key for authenticated requests (optional)"
                }),
                "force_refresh": (["No", "Yes"], {
                    "default": "No",
                    "tooltip": "Force refresh from CivitAI even if cached"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "FLOAT", "IMAGE")
    RETURN_NAMES = ("name", "version", "description", "trigger_words", "model_url", "hash", "recommended_weight", "preview_image")
    FUNCTION = "lookup_lora"
    CATEGORY = "donut/LoRA"
    OUTPUT_NODE = True

    def lookup_lora(self, lora_name: str, api_key: str = "",
                    force_refresh: str = "No") -> Tuple[str, str, str, str, str, str, float, Any]:
        """Look up LoRA information on CivitAI."""

        # Default outputs
        empty_image = None

        if lora_name == "None":
            return ("", "", "", "", "", "", 1.0, empty_image)

        # Get the full path to the LoRA file
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path or not os.path.exists(lora_path):
            print(f"[DonutLoRACivitAI] LoRA file not found: {lora_name}")
            return (lora_name, "", "File not found", "", "", "", 1.0, empty_image)

        # Compute hash
        print(f"[DonutLoRACivitAI] Computing hash for: {lora_name}")
        file_hash = get_or_compute_hash(lora_path, hash_type="SHA256", use_cache=True)
        print(f"[DonutLoRACivitAI] Hash: {file_hash}")

        # Get cache
        cache = get_cache()

        # Check if we should force refresh
        if force_refresh == "Yes":
            info = fetch_model_by_hash(file_hash, api_key=api_key if api_key else None)
            if info:
                cache.save_info(info)
                cache.download_and_cache_preview(info)
        else:
            # Use cache or fetch
            info = cache.get_or_fetch_info(
                file_hash,
                api_key=api_key if api_key else None,
                download_preview=True
            )

        if info is None:
            # Not found - provide search URL as fallback
            search_url = f"https://civitai.com/search/models?sortBy=models_v9&query={file_hash}"
            print(f"[DonutLoRACivitAI] No CivitAI info found for hash: {file_hash}")
            print(f"[DonutLoRACivitAI] Try searching manually: {search_url}")
            return (lora_name, "", f"Not found on CivitAI. Search: {search_url}", "", search_url, file_hash, 1.0, empty_image)

        # Load preview image if available
        preview_image = empty_image
        preview_path = cache.get_preview_image_path(file_hash)
        if preview_path and os.path.exists(preview_path):
            try:
                import torch
                from PIL import Image
                import numpy as np

                img = Image.open(preview_path)
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Convert to tensor in ComfyUI format (B, H, W, C)
                img_array = np.array(img).astype(np.float32) / 255.0
                preview_image = torch.from_numpy(img_array).unsqueeze(0)
            except Exception as e:
                print(f"[DonutLoRACivitAI] Error loading preview: {e}")

        # Clean up description (remove HTML tags)
        description = info.description or ""
        if description:
            import re
            description = re.sub(r'<[^>]+>', '', description)
            description = description[:500] + "..." if len(description) > 500 else description

        result = (
            info.model_name,
            info.version_name,
            description,
            info.get_trigger_words_str(),
            info.model_url,
            file_hash,
            info.recommended_weight,
            preview_image
        )

        # Return with UI data for JavaScript extension
        return {
            "ui": {"text": [info.model_name, info.version_name, description,
                           info.get_trigger_words_str(), info.model_url, file_hash,
                           str(info.recommended_weight)]},
            "result": result
        }


class DonutLoRALibrary:
    """
    LoRA Library Manager - View and manage cached CivitAI metadata.

    This node displays information about cached LoRA metadata and allows
    bulk operations on the library.
    """

    class_type = "CUSTOM"
    aux_id = "DonutsDelivery/ComfyUI-DonutNodes"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["Show Stats", "List Cached", "Scan All LoRAs", "Clear Cache"],),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "placeholder": "CivitAI API key for scanning"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "manage_library"
    CATEGORY = "donut/LoRA"
    OUTPUT_NODE = True

    def manage_library(self, action: str, api_key: str = "") -> Tuple[str]:
        """Manage the LoRA metadata library."""

        cache = get_cache()

        if action == "Show Stats":
            stats = cache.get_cache_stats()
            output = (
                f"LoRA Library Statistics:\n"
                f"------------------------\n"
                f"Cached models: {stats['metadata_count']}\n"
                f"Preview images: {stats['image_count']}\n"
                f"Total size: {stats['total_size_mb']} MB\n"
                f"Cache location: {stats['cache_dir']}"
            )

        elif action == "List Cached":
            metadata_dir = cache.metadata_dir
            entries = []
            for json_file in sorted(metadata_dir.glob("*.json")):
                try:
                    import json
                    with open(json_file, "r") as f:
                        data = json.load(f)
                        name = data.get("model_name", "Unknown")
                        version = data.get("version_name", "")
                        entries.append(f"- {name}" + (f" ({version})" if version else ""))
                except:
                    pass

            if entries:
                output = "Cached LoRA Metadata:\n" + "\n".join(entries[:50])
                if len(entries) > 50:
                    output += f"\n... and {len(entries) - 50} more"
            else:
                output = "No cached metadata found."

        elif action == "Scan All LoRAs":
            # Scan all LoRA files and fetch metadata
            loras = folder_paths.get_filename_list("loras")
            output_lines = ["Scanning LoRA files...\n"]
            found = 0
            not_found = 0

            for lora_name in loras:
                lora_path = folder_paths.get_full_path("loras", lora_name)
                if not lora_path or not os.path.exists(lora_path):
                    continue

                try:
                    file_hash = get_or_compute_hash(lora_path, use_cache=True)

                    # Check if already cached
                    existing = cache.get_cached_info(file_hash)
                    if existing:
                        found += 1
                        continue

                    # Fetch from API
                    info = cache.get_or_fetch_info(
                        file_hash,
                        api_key=api_key if api_key else None,
                        download_preview=True
                    )
                    if info:
                        found += 1
                        output_lines.append(f"+ {lora_name} -> {info.get_display_name()}")
                    else:
                        not_found += 1
                        output_lines.append(f"- {lora_name} [not found]")

                except Exception as e:
                    output_lines.append(f"! {lora_name} [error: {str(e)[:30]}]")

            output_lines.append(f"\nScan complete: {found} found, {not_found} not on CivitAI")
            output = "\n".join(output_lines)

        elif action == "Clear Cache":
            cache.clear_cache()
            output = "Cache cleared successfully."

        else:
            output = "Unknown action"

        return (output,)


class DonutLoRAHashLookup:
    """
    Direct hash lookup on CivitAI.

    Paste a hash directly to look up model information without needing the file.
    """

    class_type = "CUSTOM"
    aux_id = "DonutsDelivery/ComfyUI-DonutNodes"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hash": ("STRING", {
                    "default": "",
                    "placeholder": "SHA256 hash (e.g., 7B238076F630...)",
                    "tooltip": "Paste the SHA256 hash of a model file"
                }),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "placeholder": "CivitAI API key"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("name", "description", "trigger_words", "model_url", "recommended_weight")
    FUNCTION = "lookup_hash"
    CATEGORY = "donut/LoRA"
    OUTPUT_NODE = True

    def lookup_hash(self, hash: str, api_key: str = "") -> Tuple[str, str, str, str, float]:
        """Look up a model by hash."""

        if not hash or len(hash) < 8:
            return ("", "Please enter a valid hash", "", "", 1.0)

        # Clean up hash
        hash = hash.strip().upper()

        print(f"[DonutLoRAHashLookup] Looking up hash: {hash[:16]}...")

        cache = get_cache()
        info = cache.get_or_fetch_info(
            hash,
            api_key=api_key if api_key else None,
            download_preview=True
        )

        if info is None:
            search_url = f"https://civitai.com/search/models?sortBy=models_v9&query={hash}"
            return ("Not found", f"Model not found. Try searching: {search_url}", "", search_url, 1.0)

        # Clean description
        description = info.description or ""
        if description:
            import re
            description = re.sub(r'<[^>]+>', '', description)
            description = description[:500] + "..." if len(description) > 500 else description

        return (
            info.get_display_name(),
            description,
            info.get_trigger_words_str(),
            info.model_url,
            info.recommended_weight
        )


class DonutOpenCivitAI:
    """
    Open CivitAI page in browser.

    Takes a model URL or hash and opens the CivitAI page in your default browser.
    Useful when a LoRA isn't found via API - you can search manually.
    """

    class_type = "CUSTOM"
    aux_id = "DonutsDelivery/ComfyUI-DonutNodes"

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "action": (["Open Model Page", "Search by Hash", "Open CivitAI Home"],),
            },
            "optional": {
                "lora_name": (loras, {"default": "None"}),
                "model_url": ("STRING", {"default": "", "placeholder": "https://civitai.com/models/..."}),
                "hash": ("STRING", {"default": "", "placeholder": "SHA256 hash"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "open_page"
    CATEGORY = "donut/LoRA"
    OUTPUT_NODE = True

    def open_page(self, action: str, lora_name: str = "None",
                  model_url: str = "", hash: str = "") -> Tuple[str]:
        """Open CivitAI page in browser."""
        import webbrowser

        url = None

        if action == "Open Model Page":
            if model_url:
                url = model_url
            elif lora_name != "None":
                # Compute hash and look up
                lora_path = folder_paths.get_full_path("loras", lora_name)
                if lora_path and os.path.exists(lora_path):
                    file_hash = get_or_compute_hash(lora_path, use_cache=True)
                    cache = get_cache()
                    info = cache.get_cached_info(file_hash)
                    if info and info.model_url:
                        url = info.model_url
                    else:
                        # Fall back to search
                        url = f"https://civitai.com/search/models?sortBy=models_v9&query={file_hash}"

        elif action == "Search by Hash":
            search_hash = hash.strip().upper() if hash else None
            if not search_hash and lora_name != "None":
                lora_path = folder_paths.get_full_path("loras", lora_name)
                if lora_path and os.path.exists(lora_path):
                    search_hash = get_or_compute_hash(lora_path, use_cache=True)

            if search_hash:
                url = f"https://civitai.com/search/models?sortBy=models_v9&query={search_hash}"

        elif action == "Open CivitAI Home":
            url = "https://civitai.com"

        if url:
            try:
                webbrowser.open(url)
                return (f"Opened: {url}",)
            except Exception as e:
                return (f"Error opening browser: {e}\nURL: {url}",)
        else:
            return ("No URL to open. Select a LoRA or provide a URL/hash.",)


# Node registration
NODE_CLASS_MAPPINGS = {
    "DonutLoRACivitAIInfo": DonutLoRACivitAIInfo,
    "DonutLoRALibrary": DonutLoRALibrary,
    "DonutLoRAHashLookup": DonutLoRAHashLookup,
    "DonutOpenCivitAI": DonutOpenCivitAI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutLoRACivitAIInfo": "Donut LoRA CivitAI Info",
    "DonutLoRALibrary": "Donut LoRA Library Manager",
    "DonutLoRAHashLookup": "Donut LoRA Hash Lookup",
    "DonutOpenCivitAI": "Donut Open CivitAI",
}

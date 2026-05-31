"""
CivitAI Download Manager

Handles downloading models from CivitAI with progress tracking
and auto-organization by base model type.
"""

import os
import urllib.request
import urllib.error
import time
import uuid
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime

from .civitai_api import civitai_urlopen

# Try to import folder_paths for proper ComfyUI model directories
try:
    import folder_paths
    HAS_FOLDER_PATHS = True
except ImportError:
    HAS_FOLDER_PATHS = False

# Base model folder mappings
BASE_MODEL_FOLDERS = {
    # SD 1.x
    "SD 1.5": "sd15",
    "SD 1.4": "sd15",
    "SD 1.5 LCM": "sd15",
    "SD 1.5 Hyper": "sd15",
    # SD 2.x
    "SD 2.0": "sd2",
    "SD 2.0 768": "sd2",
    "SD 2.1": "sd2",
    "SD 2.1 768": "sd2",
    "SD 2.1 Unclip": "sd2",
    # SDXL
    "SDXL 1.0": "sdxl",
    "SDXL 0.9": "sdxl",
    "SDXL 1.0 LCM": "sdxl",
    "SDXL Distilled": "sdxl",
    "SDXL Hyper": "sdxl",
    "SDXL Lightning": "sdxl",
    "SDXL Turbo": "sdxl",
    "SDXL Canny": "sdxl",
    "SDXL Depth": "sdxl",
    # Pony
    "Pony": "pony",
    # NoobAI
    "NoobAI": "noobai",
    # SD3
    "SD 3": "sd3",
    "SD 3.5": "sd3",
    "SD 3.5 Large": "sd3",
    "SD 3.5 Large Turbo": "sd3",
    "SD 3.5 Medium": "sd3",
    # Flux
    "Flux.1 D": "flux",
    "Flux.1 S": "flux",
    "Flux.1 Schnell": "flux",
    # Lumina / ZIT
    "Lumina": "zit",
    "Lumina2": "zit",
    "ZImageTurbo": "zit",
    # Hunyuan
    "Hunyuan 1": "hunyuan",
    "Hunyuan Video": "hunyuan",
    # Illustrious
    "Illustrious": "illustrious",
    # AuraFlow
    "AuraFlow": "auraflow",
    # PixArt
    "PixArt a": "pixart",
    "PixArt E": "pixart",
    # Kolors
    "Kolors": "kolors",
    # SVD / Video
    "SVD": "svd",
    "SVD XT": "svd",
    "Mochi": "mochi",
    "LTX Video": "ltxvideo",
    "CogVideoX": "cogvideo",
    # Stable Cascade
    "Stable Cascade": "cascade",
}

# Map CivitAI model types to ComfyUI folder_paths names
MODEL_TYPE_TO_FOLDER = {
    "LORA": "loras",
    "LoCon": "loras",
    "DoRA": "loras",
    "Checkpoint": "checkpoints",
    "TextualInversion": "embeddings",
    "Hypernetwork": "hypernetworks",
    "AestheticGradient": "loras",  # No specific folder, use loras
    "Controlnet": "controlnet",
    "Upscaler": "upscale_models",
    "VAE": "vae",
    "MotionModule": "animatediff_models",
    "Poses": "loras",  # No specific folder
    "Wildcards": "wildcards",
    "Workflows": "workflows",
}


def normalize_base_model(base_model: str) -> str:
    """Convert CivitAI base model name to folder name."""
    if not base_model:
        return ""
    return BASE_MODEL_FOLDERS.get(base_model, "")


def get_model_folder(model_type: str) -> str:
    """
    Get the ComfyUI model folder path for a given model type.

    Uses folder_paths to get the correct directory.
    """
    folder_name = MODEL_TYPE_TO_FOLDER.get(model_type, "loras")

    if HAS_FOLDER_PATHS:
        try:
            paths = folder_paths.get_folder_paths(folder_name)
            if paths:
                return paths[0]  # Return first (primary) path
        except Exception as e:
            print(f"[CivitAI Download] Error getting folder path for {folder_name}: {e}")

    # Fallback to models_dir if folder_paths available
    if HAS_FOLDER_PATHS:
        return os.path.join(folder_paths.models_dir, folder_name)

    # Ultimate fallback
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", folder_name)


def get_download_path(model_type: str, base_model: str, filename: str, use_subfolder: bool = True) -> str:
    """
    Get the full download path for a model.

    Args:
        model_type: CivitAI model type (LORA, Checkpoint, etc.)
        base_model: CivitAI base model (SDXL 1.0, Pony, etc.)
        filename: Original filename from CivitAI
        use_subfolder: Whether to organize into base model subfolders (default True for LoRAs)

    Returns:
        Full path where the file should be saved
    """
    base_dir = get_model_folder(model_type)

    # For LoRAs/LoCon/DoRA, organize by base model subfolder
    # For other types (checkpoints, VAE, etc.), put directly in folder
    if use_subfolder and model_type in ("LORA", "LoCon", "DoRA"):
        subfolder = normalize_base_model(base_model)
        if subfolder:
            full_dir = os.path.join(base_dir, subfolder)
        else:
            full_dir = base_dir
    else:
        full_dir = base_dir

    # Create directory if needed
    Path(full_dir).mkdir(parents=True, exist_ok=True)

    return os.path.join(full_dir, filename)


def invalidate_folder_cache(folder_name: str = None):
    """
    Invalidate the folder_paths cache to make new files visible.

    Args:
        folder_name: Specific folder to invalidate, or None to clear all
    """
    if not HAS_FOLDER_PATHS:
        return

    try:
        # Clear the filename_list_cache to force rescan
        if hasattr(folder_paths, 'filename_list_cache'):
            if folder_name:
                # Clear specific folder cache
                if folder_name in folder_paths.filename_list_cache:
                    del folder_paths.filename_list_cache[folder_name]
            else:
                # Clear entire cache
                folder_paths.filename_list_cache.clear()
            print(f"[CivitAI Download] Invalidated folder cache: {folder_name or 'all'}")
    except Exception as e:
        print(f"[CivitAI Download] Error invalidating cache: {e}")


@dataclass
class DownloadStatus:
    """Status of a download."""
    download_id: str
    url: str
    filepath: str
    filename: str
    model_type: str = ""  # CivitAI model type for cache invalidation
    total_size: int = 0
    downloaded_size: int = 0
    status: str = "pending"  # pending, downloading, completed, error, cancelled
    error: str = ""
    started_at: str = ""
    completed_at: str = ""
    speed_bps: float = 0.0  # bytes per second


class CivitAIDownloader:
    """
    Manages downloads from CivitAI with progress tracking.
    """

    def __init__(self):
        self.downloads: Dict[str, DownloadStatus] = {}
        self._lock = threading.Lock()

    def start_download(
        self,
        download_url: str,
        save_path: str,
        api_key: Optional[str] = None,
        on_progress: Optional[Callable[[DownloadStatus], None]] = None,
        model_type: str = "",
        sha256: Optional[str] = None
    ) -> str:
        """
        Start a download in a background thread.

        Args:
            download_url: URL to download from
            save_path: Where to save the file
            api_key: Optional CivitAI API key for authenticated downloads
            on_progress: Optional callback for progress updates
            model_type: CivitAI model type (for cache invalidation)
            sha256: Pre-known SHA256 hash from CivitAI (saved after download)

        Returns:
            Download ID for tracking
        """
        download_id = str(uuid.uuid4())[:8]
        filename = os.path.basename(save_path)

        status = DownloadStatus(
            download_id=download_id,
            url=download_url,
            filepath=save_path,
            filename=filename,
            model_type=model_type,
            status="pending",
            started_at=datetime.now().isoformat()
        )

        with self._lock:
            self.downloads[download_id] = status

        # Start download in background thread
        thread = threading.Thread(
            target=self._download_thread,
            args=(download_id, download_url, save_path, api_key, on_progress, model_type, sha256),
            daemon=True
        )
        thread.start()

        return download_id

    def _download_thread(
        self,
        download_id: str,
        url: str,
        save_path: str,
        api_key: Optional[str],
        on_progress: Optional[Callable],
        model_type: str = "",
        sha256: Optional[str] = None
    ):
        """Background thread for downloading."""
        status = self.downloads.get(download_id)
        if not status:
            return

        try:
            # Prepare request - CivitAI requires API key as URL parameter for downloads
            download_url = url
            if api_key:
                # Append token to URL as CivitAI expects
                separator = "&" if "?" in url else "?"
                download_url = f"{url}{separator}token={api_key}"
                print(f"[CivitAI Download] Using authenticated URL")
            else:
                print(f"[CivitAI Download] No API key, using unauthenticated URL")

            print(f"[CivitAI Download] Starting download to: {save_path}")

            headers = {
                "User-Agent": "ComfyUI-DonutNodes/1.0"
            }

            request = urllib.request.Request(download_url, headers=headers)

            with self._lock:
                status.status = "downloading"

            # Create directory if needed
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            # Open connection
            with civitai_urlopen(request, timeout=30) as response:
                total_size = int(response.headers.get('Content-Length', 0))

                with self._lock:
                    status.total_size = total_size

                # Download with progress
                downloaded = 0
                chunk_size = 1024 * 1024  # 1MB chunks
                start_time = time.time()

                with open(save_path, 'wb') as f:
                    while True:
                        # Check for cancellation
                        with self._lock:
                            if status.status == "cancelled":
                                break

                        chunk = response.read(chunk_size)
                        if not chunk:
                            break

                        f.write(chunk)
                        downloaded += len(chunk)

                        # Update progress
                        elapsed = time.time() - start_time
                        speed = downloaded / elapsed if elapsed > 0 else 0

                        with self._lock:
                            status.downloaded_size = downloaded
                            status.speed_bps = speed

                        if on_progress:
                            on_progress(status)

                # Check final status
                with self._lock:
                    if status.status == "cancelled":
                        # Clean up partial file
                        if os.path.exists(save_path):
                            os.remove(save_path)
                    else:
                        status.status = "completed"
                        status.completed_at = datetime.now().isoformat()

                        # Save the SHA256 hash to a .hash file if provided
                        if sha256:
                            self._save_hash_file(save_path, sha256)

                        # Invalidate folder cache so the new file is visible in dropdowns
                        folder_name = MODEL_TYPE_TO_FOLDER.get(model_type, "loras")
                        invalidate_folder_cache(folder_name)

        except urllib.error.HTTPError as e:
            error_msg = f"HTTP {e.code}: {e.reason}"
            if e.code == 401:
                error_msg = "Unauthorized - Check your CivitAI API key in settings"
            elif e.code == 403:
                error_msg = "Forbidden - You may need a CivitAI API key or this model requires login"
            elif e.code == 404:
                error_msg = "Not found - The download link may have expired"
            with self._lock:
                status.status = "error"
                status.error = error_msg
            print(f"[CivitAI Download] HTTP Error: {e.code} {e.reason}")

        except urllib.error.URLError as e:
            error_msg = f"Network error: {e.reason}"
            with self._lock:
                status.status = "error"
                status.error = error_msg
            print(f"[CivitAI Download] URL Error: {e.reason}")

        except Exception as e:
            with self._lock:
                status.status = "error"
                status.error = str(e)
            print(f"[CivitAI Download] Error: {e}")

    def _save_hash_file(self, file_path: str, sha256: str):
        """Save SHA256 hash to a .hash cache file alongside the model file."""
        import json
        try:
            hash_file = file_path + ".hash"
            hash_data = {
                "mtime": os.path.getmtime(file_path),
                "file_path": file_path,
                "hashes": {
                    "SHA256": sha256.upper()
                }
            }
            with open(hash_file, 'w') as f:
                json.dump(hash_data, f, indent=2)
            print(f"[CivitAI Download] Saved hash cache: {hash_file}")
        except Exception as e:
            print(f"[CivitAI Download] Error saving hash file: {e}")

    def get_status(self, download_id: str) -> Optional[DownloadStatus]:
        """Get status of a download."""
        with self._lock:
            return self.downloads.get(download_id)

    def cancel_download(self, download_id: str) -> bool:
        """Cancel a download."""
        with self._lock:
            status = self.downloads.get(download_id)
            if status and status.status == "downloading":
                status.status = "cancelled"
                return True
        return False

    def get_all_downloads(self) -> Dict[str, DownloadStatus]:
        """Get all download statuses."""
        with self._lock:
            return dict(self.downloads)

    def clear_completed(self):
        """Remove completed/error/cancelled downloads from tracking."""
        with self._lock:
            to_remove = [
                k for k, v in self.downloads.items()
                if v.status in ("completed", "error", "cancelled")
            ]
            for k in to_remove:
                del self.downloads[k]


# Global downloader instance
_downloader: Optional[CivitAIDownloader] = None


def get_downloader() -> CivitAIDownloader:
    """Get or create the global downloader instance."""
    global _downloader
    if _downloader is None:
        _downloader = CivitAIDownloader()
    return _downloader

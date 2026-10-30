"""
LoRA file hash computation utilities for CivitAI lookups.

Supports multiple hash algorithms:
- SHA256 (most reliable for CivitAI lookups)
- AutoV2 (legacy, first 0x10000 bytes SHA256)
- BLAKE3 (fast, modern)
- CRC32 (legacy)
"""

import hashlib
import os
from typing import Optional, Dict, Tuple
from pathlib import Path

# Try to import blake3 for faster hashing (optional)
try:
    import blake3
    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False


def compute_sha256(file_path: str, chunk_size: int = 1024 * 1024) -> str:
    """
    Compute full SHA256 hash of a file.

    Args:
        file_path: Path to the file
        chunk_size: Size of chunks to read (default 1MB)

    Returns:
        Uppercase hex string of SHA256 hash
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest().upper()


def compute_autov2(file_path: str) -> str:
    """
    Compute AutoV2 hash (SHA256 of first 0x10000 bytes).

    This is a fast hash format that only reads the first 64KB of the file.
    CivitAI accepts this for lookups via /api/v1/model-versions/by-hash/:hash

    Args:
        file_path: Path to the file

    Returns:
        First 10 characters of SHA256 hash of first 0x10000 bytes (uppercase)
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read only first 0x10000 (65536) bytes
        data = f.read(0x10000)
        sha256_hash.update(data)
    # AutoV2 uses first 10 hex characters (matches CivitAI format)
    return sha256_hash.hexdigest()[:10].upper()


def compute_blake3(file_path: str, chunk_size: int = 65536) -> Optional[str]:
    """
    Compute BLAKE3 hash of a file (if blake3 library available).

    BLAKE3 is much faster than SHA256, especially for large files.

    Args:
        file_path: Path to the file
        chunk_size: Size of chunks to read (default 64KB for BLAKE3 efficiency)

    Returns:
        Uppercase hex string of BLAKE3 hash, or None if blake3 not available
    """
    if not HAS_BLAKE3:
        return None

    hasher = blake3.blake3()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest().upper()


def compute_crc32(file_path: str, chunk_size: int = 8192) -> str:
    """
    Compute CRC32 hash of a file.

    Args:
        file_path: Path to the file
        chunk_size: Size of chunks to read

    Returns:
        8-character uppercase hex string of CRC32
    """
    import zlib
    crc = 0
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            crc = zlib.crc32(chunk, crc)
    # Convert to unsigned and format as 8-char hex
    return format(crc & 0xFFFFFFFF, '08X')


def compute_all_hashes(file_path: str) -> Dict[str, str]:
    """
    Compute all supported hashes for a file.

    Args:
        file_path: Path to the file

    Returns:
        Dictionary with hash types as keys and hash values as values
    """
    hashes = {
        "SHA256": compute_sha256(file_path),
        "AutoV2": compute_autov2(file_path),
        "CRC32": compute_crc32(file_path),
    }

    if HAS_BLAKE3:
        hashes["BLAKE3"] = compute_blake3(file_path)

    return hashes


def compute_hash_for_civitai(file_path: str, prefer_fast: bool = True) -> Tuple[str, str]:
    """
    Compute the best hash for CivitAI lookup.

    SHA256 is most reliable for CivitAI lookups, but for very large files
    we can try AutoV2 first (faster) and fall back to SHA256 if needed.

    Args:
        file_path: Path to the file
        prefer_fast: If True, compute AutoV2 first for initial lookup attempt

    Returns:
        Tuple of (hash_type, hash_value) - hash_type is "SHA256" or "AutoV2"
    """
    # Get file size to decide strategy
    file_size = os.path.getsize(file_path)

    # For files under 100MB, just compute SHA256 directly (most reliable)
    if file_size < 100 * 1024 * 1024 or not prefer_fast:
        return ("SHA256", compute_sha256(file_path))

    # For larger files, we return SHA256 anyway since CivitAI primarily uses it
    # AutoV2 is less reliable for lookups
    return ("SHA256", compute_sha256(file_path))


def _get_cache_filename(file_path: str) -> str:
    """Generate a unique cache filename from a file path."""
    import hashlib
    # Use hash of full path to avoid collisions from same-named files in different dirs
    path_hash = hashlib.md5(file_path.encode()).hexdigest()[:12]
    stem = Path(file_path).stem[:30]  # Truncate long filenames
    return f"{stem}_{path_hash}.hash"


def _fallback_cache_dir() -> Optional[str]:
    """Central hash cache used when the sidecar next to a LoRA can't be
    written (read-only or shared model folders). Without it every load
    would re-hash the whole file."""
    try:
        from .civitai_api import get_cache
        return str(get_cache().cache_dir / "hashes")
    except Exception:
        return None


def get_cached_hash(file_path: str, cache_dir: Optional[str] = None) -> Optional[Dict[str, str]]:
    """
    Check if we have cached hashes for this file.

    The cache stores hashes alongside the file's modification time to
    detect if the file has changed.

    Args:
        file_path: Path to the file
        cache_dir: Directory for cache files (default: same as file)

    Returns:
        Dictionary of cached hashes if valid cache exists, None otherwise
    """
    import json

    if cache_dir is None:
        cache_file = Path(file_path).with_suffix(Path(file_path).suffix + ".hash")
    else:
        cache_file = Path(cache_dir) / _get_cache_filename(file_path)

    if not cache_file.exists():
        if cache_dir is None:
            fallback = _fallback_cache_dir()
            if fallback:
                return get_cached_hash(file_path, fallback)
        return None

    try:
        with open(cache_file, "r") as f:
            cache_data = json.load(f)

        # Check if file modification time matches
        current_mtime = os.path.getmtime(file_path)
        if cache_data.get("mtime") != current_mtime:
            return None

        return cache_data.get("hashes")
    except (json.JSONDecodeError, OSError):
        return None


def save_hash_cache(file_path: str, hashes: Dict[str, str], cache_dir: Optional[str] = None) -> bool:
    """
    Save computed hashes to cache file.

    Args:
        file_path: Path to the original file
        hashes: Dictionary of hash types and values
        cache_dir: Directory for cache files (default: same as file)

    Returns:
        True if cache was saved successfully
    """
    import json

    if cache_dir is None:
        cache_file = Path(file_path).with_suffix(Path(file_path).suffix + ".hash")
    else:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cache_file = Path(cache_dir) / _get_cache_filename(file_path)

    try:
        cache_data = {
            "mtime": os.path.getmtime(file_path),
            "file_path": str(file_path),
            "hashes": hashes
        }
        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)
        return True
    except OSError:
        if cache_dir is None:
            fallback = _fallback_cache_dir()
            if fallback:
                try:
                    return save_hash_cache(file_path, hashes, fallback)
                except OSError:
                    return False
        return False


def compute_single_hash(file_path: str, hash_type: str) -> str:
    """
    Compute only the requested hash type.

    Args:
        file_path: Path to the file
        hash_type: Type of hash ("SHA256", "AutoV2", "BLAKE3", "CRC32")

    Returns:
        Hash value as uppercase hex string
    """
    if hash_type == "AutoV2":
        return compute_autov2(file_path)
    elif hash_type == "BLAKE3":
        result = compute_blake3(file_path)
        if result is None:
            # Fall back to SHA256 if BLAKE3 not available
            return compute_sha256(file_path)
        return result
    elif hash_type == "CRC32":
        return compute_crc32(file_path)
    else:  # Default to SHA256
        return compute_sha256(file_path)


def get_or_compute_hash(file_path: str, hash_type: str = "SHA256",
                        use_cache: bool = True, cache_dir: Optional[str] = None,
                        compute: bool = True) -> str:
    """
    Get hash from cache or compute it.

    This is the main function to use - it handles caching automatically.

    Args:
        file_path: Path to the file
        hash_type: Type of hash ("SHA256", "AutoV2", "BLAKE3", "CRC32")
        use_cache: Whether to use/update cache
        cache_dir: Directory for cache files
        compute: If False, only return a cached hash ("" when there is none)

    Returns:
        Hash value as uppercase hex string
    """
    if use_cache:
        cached = get_cached_hash(file_path, cache_dir)
        if cached and hash_type in cached:
            return cached[hash_type]
    if not compute:
        return ""

    # Only compute the requested hash for speed
    hash_value = compute_single_hash(file_path, hash_type)

    # Cache just this hash (don't compute all hashes)
    if use_cache:
        # Get existing cache or create new
        cached = get_cached_hash(file_path, cache_dir) or {}
        cached[hash_type] = hash_value
        save_hash_cache(file_path, cached, cache_dir)

    return hash_value

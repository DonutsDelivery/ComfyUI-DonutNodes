"""
DonutZitHybrid

Nodes to package two differently-LoRA'd ZiT (Z-Image Turbo / Lumina2)
models into a single .safetensors that switches weights mid-sampling at a
configurable sigma threshold.

Nodes:
  - DonutZitHybridSave     : author-side, packs base+early+delta into one file
  - DonutZitHybridLoader   : end-user, loads the packed file → standard MODEL
  - DonutZitHybridApply    : live testing, two MODELs → switching MODEL
  - DonutZitHybridOverride : optional, override baked switch settings at runtime

The saved file contains:
  - model weights with lora_early baked in (the "early" model, stored once)
  - zit_hybrid_delta.<key>.lora_up.weight      delta LoRA A/B matrices
  - zit_hybrid_delta.<key>.lora_down.weight   (late - early, rank 2r, tiny)
  - zit_hybrid_delta.<key>.alpha
  - metadata["zit_hybrid"] JSON : switch settings

File size ≈ base_model + 2× lora_size (not 2× base_model).
"""

import gc
import json
import math
import os

import folder_paths
import torch

import comfy.lora
import comfy.model_detection
import comfy.model_management
import comfy.sd
import comfy.utils
from comfy.cli_args import args as comfy_args

from .DonutModelSave import _materialize_and_cast


DELTA_PREFIX = "zit_hybrid_delta."
METADATA_KEY = "zit_hybrid"
FORMAT_VERSION = 2

ALPHA_EPS = 1e-4
BLEND_CURVES = ["linear", "smoothstep", "smootherstep", "ease_in", "ease_out", "cosine"]
LORA_RANKS = [4, 8, 16, 32, 64, 128, 256, 512]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _detect_unet_prefix(sd):
    return comfy.model_detection.unet_prefix_from_state_dict(sd)


def _ram_free_gb():
    """Return available RAM in GB (Linux)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except Exception:
        return -1


def _bake_model_state_dict(model):
    """Materialize all patches (LoRAs/merges) into a CPU state dict.

    Uses the same state_dict_for_saving + _materialize_and_cast path as
    DonutModelSave, so Apply (Live) results match Save+Load exactly.
    """
    comfy.model_management.load_models_gpu([model])
    sd = model.state_dict_for_saving(None, None, None)
    sd = _materialize_and_cast(sd, "original")
    comfy.model_management.unload_all_models()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return sd


def _apply_curve(t, curve):
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    if curve == "linear":
        return t
    if curve == "smoothstep":
        return t * t * (3.0 - 2.0 * t)
    if curve == "smootherstep":
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    if curve == "ease_in":
        return t * t
    if curve == "ease_out":
        inv = 1.0 - t
        return 1.0 - inv * inv
    if curve == "cosine":
        return 0.5 - 0.5 * math.cos(math.pi * t)
    return t


def _get_module_by_dotted(root, dotted_name):
    obj = root
    parts = dotted_name.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    return obj, parts[-1]


def _compute_alpha(cur_sigma, switch_sigma, start_sigma, end_sigma):
    if start_sigma <= end_sigma:
        return 0.0 if cur_sigma > switch_sigma else 1.0
    if cur_sigma >= start_sigma:
        return 0.0
    if cur_sigma <= end_sigma:
        return 1.0
    return (start_sigma - cur_sigma) / (start_sigma - end_sigma)


# ---------------------------------------------------------------------------
# delta LoRA computation (saver-side)
# ---------------------------------------------------------------------------

def _svd_decompose(delta, rank):
    """
    Decompose a full-rank delta tensor into low-rank LoRA A/B via
    randomized SVD (torch.svd_lowrank). Much faster than full SVD
    when rank << min(m, n).

    Returns (lora_up, lora_down) where:
      lora_up  : [out_dim, rank]  (the B matrix)
      lora_down: [rank, in_dim]   (the A matrix)
    such that lora_up @ lora_down ≈ delta.reshape(out_dim, in_dim)
    """
    mat = delta.float().reshape(delta.shape[0], -1)  # [out_dim, in_dim]
    r = min(rank, min(mat.shape[0], mat.shape[1]))
    U, S, V = torch.svd_lowrank(mat, q=r)
    sqrt_S = S.sqrt()
    lora_up = (U * sqrt_S.unsqueeze(0))        # [out_dim, r]
    lora_down = (sqrt_S.unsqueeze(1) * V.T)    # [r, in_dim]
    return lora_up, lora_down


def _build_delta_lora_from_models(sd_early, sd_late, rank):
    """
    Compute a delta LoRA from two baked state dicts via SVD decomposition.

    For each key that differs between early and late, computes the full-rank
    delta, then decomposes it into rank-r A/B matrices.

    Returns dict of { state_dict_key: (lora_up, lora_down, alpha) }
    """
    delta_dict = {}
    divergent_keys = []
    for k, a in sd_early.items():
        b = sd_late.get(k)
        if b is None or not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            continue
        if a.shape != b.shape:
            continue
        if torch.equal(a, b):
            continue
        divergent_keys.append(k)

    n_total = len(divergent_keys)
    for i, k in enumerate(divergent_keys):
        if (i + 1) % 20 == 0 or i == 0 or i == n_total - 1:
            print(f"[DonutZitHybridSave] SVD {i + 1}/{n_total}...")
        a = sd_early[k]
        b = sd_late[k]
        delta = b.float() - a.float()
        lora_up, lora_down = _svd_decompose(delta, rank)
        r = lora_down.shape[0]
        delta_dict[k] = (lora_up, lora_down, float(r))

    print(f"[DonutZitHybrid] SVD-decomposed {n_total} divergent keys to rank {rank}")
    return delta_dict


# ---------------------------------------------------------------------------
# pre-compute full-rank deltas from embedded LoRA (loader-side)
# ---------------------------------------------------------------------------

def _precompute_deltas(delta_lora_entries, model_sd, unet_prefix):
    """
    From the raw delta LoRA A/B matrices stored in the file, compute
    full-rank delta tensors and snapshot the early (baked) weights.

    Returns (early_snapshot, deltas) where both are dicts keyed by
    model state_dict key, holding CPU tensors.
    """
    early_snapshot = {}
    deltas = {}

    for model_key, (up, down, alpha) in delta_lora_entries.items():
        rank = down.shape[0]
        scale = alpha / rank if alpha else 1.0

        # compute full-rank delta
        delta = torch.mm(up.float().flatten(start_dim=1),
                         down.float().flatten(start_dim=1))

        # find the corresponding weight in the base model to get the shape
        full_key = unet_prefix + model_key if not model_key.startswith(unet_prefix) else model_key
        base_weight = model_sd.get(full_key)
        if base_weight is not None:
            delta = (scale * delta).reshape(base_weight.shape).to(base_weight.dtype)
            early_snapshot[full_key] = base_weight.detach().clone().to("cpu")
            deltas[full_key] = delta.to("cpu")

    return early_snapshot, deltas


# ---------------------------------------------------------------------------
# wrapper (shared by Loader, Apply, Override)
# ---------------------------------------------------------------------------

def _build_delta_wrapper(model_patcher, early_snapshot, deltas,
                         baked_percent, baked_blend_width, baked_curve,
                         unet_prefix):
    """
    Wrapper that applies early + alpha * delta one key at a time.
    No bulk pre-computation — keeps memory bounded during both setup
    and per-step execution.
    """
    diffusion_model = model_patcher.model.diffusion_model

    # Build key → (parent_module, leaf_name) mapping once
    param_map = {}
    skipped = 0
    for full_key in early_snapshot:
        if not full_key.startswith(unet_prefix):
            skipped += 1
            continue
        sub = full_key[len(unet_prefix):]
        try:
            parent, leaf = _get_module_by_dotted(diffusion_model, sub)
            target = getattr(parent, leaf)
            if target is not None:
                param_map[full_key] = (parent, leaf)
        except AttributeError:
            skipped += 1
            continue

    print(f"[ZitHybrid wrapper] param_map: {len(param_map)} keys mapped, "
          f"{skipped} skipped, unet_prefix='{unet_prefix}'")
    if len(param_map) == 0 and len(early_snapshot) > 0:
        sample_key = next(iter(early_snapshot))
        print(f"[ZitHybrid wrapper] WARNING: param_map empty! sample key: '{sample_key}'")

    state = {"current_alpha": None}

    def _apply_alpha(alpha):
        if state["current_alpha"] is not None and abs(alpha - state["current_alpha"]) < ALPHA_EPS:
            return

        for full_key, (parent, leaf) in param_map.items():
            target = getattr(parent, leaf)
            device = target.device
            dtype = target.dtype
            e = early_snapshot[full_key]
            d = deltas.get(full_key)

            if alpha <= 0.0 or d is None:
                src = e.to(device=device, dtype=dtype, non_blocking=True)
            else:
                src = (e.float() + alpha * d.float()).to(dtype=dtype, device=device,
                                                         non_blocking=True)

            with torch.no_grad():
                target.data.copy_(src)
            del src

        state["current_alpha"] = alpha

    def wrapper(apply_model, args):
        c = args["c"]
        timestep = args["timestep"]
        transformer_options = c.get("transformer_options", {}) if isinstance(c, dict) else {}
        percent = float(transformer_options.get("zit_hybrid_switch_percent", baked_percent))
        blend_width = float(transformer_options.get("zit_hybrid_blend_width", baked_blend_width))
        curve = transformer_options.get("zit_hybrid_blend_curve", baked_curve)

        ms = model_patcher.model.model_sampling
        switch_sigma = float(ms.percent_to_sigma(percent))
        cur_sigma = float(timestep[0].item())

        if blend_width <= 0.0:
            alpha = 0.0 if cur_sigma > switch_sigma else 1.0
        else:
            end_percent = max(0.0, min(1.0, percent + blend_width))
            start_sigma = switch_sigma
            end_sigma = float(ms.percent_to_sigma(end_percent))
            alpha_linear = _compute_alpha(cur_sigma, switch_sigma, start_sigma, end_sigma)
            alpha = _apply_curve(alpha_linear, curve)

        print(f"[ZitHybrid] sigma={cur_sigma:.4f} switch_sigma={switch_sigma:.4f} alpha={alpha:.2f}")

        _apply_alpha(alpha)
        return apply_model(args["input"], timestep, **c)

    return wrapper


# ---------------------------------------------------------------------------
# Saver — base MODEL + two LoRA paths → single .safetensors
# ---------------------------------------------------------------------------

class DonutZitHybridSave:
    """
    Pack two models into a single .safetensors: the early model's full
    weights plus a delta LoRA (computed via SVD) that transforms early → late.

    Inputs:
      model_early : base MODEL with early LoRA applied
      model_late  : base MODEL with late LoRA applied
      rank        : LoRA rank for the delta (match your original LoRA rank)

    The delta is computed as SVD(late_weights - early_weights), truncated
    to the specified rank. File size ≈ base_model + delta_lora.
    """

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_early": ("MODEL",),
                "model_late": ("MODEL",),
                "rank": (LORA_RANKS, {
                    "default": 128,
                    "tooltip": "LoRA rank for the delta. Match your source LoRA rank "
                               "for equivalent quality."
                }),
                "filename_prefix": ("STRING", {"default": "diffusion_models/zit_hybrid"}),
                "switch_percent": ("FLOAT", {
                    "default": 0.33, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Sampling percent at which late weights START being applied."
                }),
                "blend_width": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Ramp length past switch_percent. 0 = hard switch."
                }),
                "blend_curve": (BLEND_CURVES, {"default": "linear"}),
            },
            "optional": {
                "settings": ("ZIT_HYBRID_SETTINGS", {
                    "tooltip": "Optional: pipe settings from ZiT Hybrid Apply (Live) "
                               "to copy switch_percent, blend_width, and blend_curve."
                }),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "advanced/model_merging"

    def save(self, model_early, model_late, rank, filename_prefix,
             switch_percent, blend_width, blend_curve, settings=None):
        if settings is not None:
            switch_percent = settings.get("switch_percent", switch_percent)
            blend_width = settings.get("blend_width", blend_width)
            blend_curve = settings.get("blend_curve", blend_curve)
        full_output_folder, filename, counter, _subfolder, _fp = \
            folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        output_path = os.path.join(
            full_output_folder, f"{filename}_{counter:05}_.safetensors"
        )

        # 1. Bake both models
        print("[DonutZitHybridSave] baking model_early...")
        sd_early = _bake_model_state_dict(model_early)
        comfy.model_management.unload_all_models()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[DonutZitHybridSave] baking model_late...")
        sd_late = _bake_model_state_dict(model_late)
        comfy.model_management.unload_all_models()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 2. Compute delta LoRA via SVD
        print(f"[DonutZitHybridSave] computing delta LoRA (rank {rank})...")
        delta_dict = _build_delta_lora_from_models(sd_early, sd_late, rank)
        del sd_late

        # 3. Build output state dict: early weights + delta LoRA
        out_sd = dict(sd_early)
        for state_key, (up, down, alpha) in delta_dict.items():
            # Strip the unet prefix for the delta keys to keep them portable
            out_sd[f"{DELTA_PREFIX}{state_key}.lora_up.weight"] = up.contiguous()
            out_sd[f"{DELTA_PREFIX}{state_key}.lora_down.weight"] = down.contiguous()
            out_sd[f"{DELTA_PREFIX}{state_key}.alpha"] = torch.tensor(alpha)

        # 4. Metadata
        unet_prefix = _detect_unet_prefix(sd_early)
        del sd_early
        metadata = {}
        if not comfy_args.disable_metadata:
            metadata[METADATA_KEY] = json.dumps({
                "version": FORMAT_VERSION,
                "switch_percent": float(switch_percent),
                "blend_width": float(blend_width),
                "blend_curve": blend_curve,
                "unet_prefix": unet_prefix,
                "rank": rank,
                "delta_keys": list(delta_dict.keys()),
            })

        comfy.utils.save_torch_file(out_sd, output_path, metadata=metadata)
        print(f"[DonutZitHybridSave] Saved {output_path}")
        print(f"[DonutZitHybridSave] {len(delta_dict)} delta LoRA keys at rank {rank}, "
              f"switch_percent={switch_percent}, blend_width={blend_width}")
        return {}


# ---------------------------------------------------------------------------
# Loader — reads packed file → MODEL with wrapper
# ---------------------------------------------------------------------------

class DonutZitHybridLoader:
    """Load a ZiT hybrid .safetensors → standard MODEL with switching."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "advanced/loaders"

    def load(self, unet_name):
        path = folder_paths.get_full_path("diffusion_models", unet_name)
        sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)

        meta_blob = None
        if metadata is not None and METADATA_KEY in metadata:
            try:
                meta_blob = json.loads(metadata[METADATA_KEY])
            except Exception as e:
                raise RuntimeError(f"[DonutZitHybridLoader] bad metadata: {e}")
        if meta_blob is None:
            raise RuntimeError(
                f"[DonutZitHybridLoader] {unet_name} is not a ZiT hybrid file"
            )

        baked_percent = float(meta_blob.get("switch_percent", 0.33))
        baked_blend_width = float(meta_blob.get("blend_width", 0.0))
        baked_curve = str(meta_blob.get("blend_curve", "linear"))
        if baked_curve not in BLEND_CURVES:
            baked_curve = "linear"
        unet_prefix = meta_blob.get("unet_prefix", "")

        # Split base model keys from delta LoRA keys
        base_sd = {}
        delta_raw = {}  # model_key → {up, down, alpha}
        for k, v in sd.items():
            if k.startswith(DELTA_PREFIX):
                rest = k[len(DELTA_PREFIX):]
                if rest.endswith(".lora_up.weight"):
                    mk = rest[:-len(".lora_up.weight")]
                    delta_raw.setdefault(mk, {})["up"] = v
                elif rest.endswith(".lora_down.weight"):
                    mk = rest[:-len(".lora_down.weight")]
                    delta_raw.setdefault(mk, {})["down"] = v
                elif rest.endswith(".alpha"):
                    mk = rest[:-len(".alpha")]
                    delta_raw.setdefault(mk, {})["alpha"] = v.item()
            else:
                base_sd[k] = v

        # Build delta entries
        delta_entries = {}
        for mk, parts in delta_raw.items():
            if "up" in parts and "down" in parts:
                delta_entries[mk] = (parts["up"], parts["down"], parts.get("alpha"))

        # Load base model
        print(f"[DonutZitHybridLoader] loading base model...")
        model_patcher = comfy.sd.load_diffusion_model_state_dict(base_sd)
        if model_patcher is None:
            raise RuntimeError(f"[DonutZitHybridLoader] failed to load model from {unet_name}")

        if not unet_prefix:
            unet_prefix = _detect_unet_prefix(base_sd)

        # Pre-compute full-rank deltas
        print(f"[DonutZitHybridLoader] pre-computing {len(delta_entries)} deltas...")
        early_snapshot, deltas = _precompute_deltas(delta_entries, base_sd, unet_prefix)

        wrapper = _build_delta_wrapper(model_patcher, early_snapshot, deltas,
                                       baked_percent, baked_blend_width, baked_curve,
                                       unet_prefix)
        model_patcher.set_model_unet_function_wrapper(wrapper)

        print(f"[DonutZitHybridLoader] Loaded {unet_name}: {len(deltas)} delta keys")
        return (model_patcher,)


def _build_apply_wrapper(model_patcher, early_div, late_div,
                         baked_percent, baked_blend_width, baked_curve,
                         unet_prefix):
    """
    Wrapper for Apply (Live) using two pre-baked snapshot dicts.
    Per-step: copies the right snapshot into live params. No float32
    intermediates — tensors stay in original dtype.
    """
    diffusion_model = model_patcher.model.diffusion_model

    # Build key → (parent, leaf) mapping once
    param_map = {}
    skipped = 0
    for full_key in early_div:
        if not full_key.startswith(unet_prefix):
            skipped += 1
            continue
        sub = full_key[len(unet_prefix):]
        try:
            parent, leaf = _get_module_by_dotted(diffusion_model, sub)
            if getattr(parent, leaf) is not None:
                param_map[full_key] = (parent, leaf)
        except AttributeError:
            skipped += 1
            continue

    print(f"[ZitHybrid apply wrapper] param_map: {len(param_map)} keys mapped, "
          f"{skipped} skipped, unet_prefix='{unet_prefix}'")
    if len(param_map) == 0 and len(early_div) > 0:
        sample_key = next(iter(early_div))
        print(f"[ZitHybrid apply wrapper] WARNING: param_map empty! sample key: '{sample_key}'")

    state = {"current_alpha": None}

    def _apply_alpha(alpha):
        if state["current_alpha"] is not None and abs(alpha - state["current_alpha"]) < ALPHA_EPS:
            return

        for full_key, (parent, leaf) in param_map.items():
            target = getattr(parent, leaf)
            device = target.device
            dtype = target.dtype
            e = early_div[full_key]
            l = late_div[full_key]

            if alpha <= 0.0:
                src = e
            elif alpha >= 1.0:
                src = l
            else:
                src = torch.lerp(e.to(torch.float32), l.to(torch.float32), alpha).to(dtype)

            with torch.no_grad():
                target.data.copy_(src.to(device=device, non_blocking=True))
            del src

        state["current_alpha"] = alpha

    def wrapper(apply_model, args):
        c = args["c"]
        timestep = args["timestep"]
        transformer_options = c.get("transformer_options", {}) if isinstance(c, dict) else {}
        percent = float(transformer_options.get("zit_hybrid_switch_percent", baked_percent))
        blend_width = float(transformer_options.get("zit_hybrid_blend_width", baked_blend_width))
        curve = transformer_options.get("zit_hybrid_blend_curve", baked_curve)

        ms = model_patcher.model.model_sampling
        switch_sigma = float(ms.percent_to_sigma(percent))
        cur_sigma = float(timestep[0].item())

        if blend_width <= 0.0:
            alpha = 0.0 if cur_sigma > switch_sigma else 1.0
        else:
            end_percent = max(0.0, min(1.0, percent + blend_width))
            start_sigma = switch_sigma
            end_sigma = float(ms.percent_to_sigma(end_percent))
            alpha_linear = _compute_alpha(cur_sigma, switch_sigma, start_sigma, end_sigma)
            alpha = _apply_curve(alpha_linear, curve)

        print(f"[ZitHybrid apply] sigma={cur_sigma:.4f} switch_sigma={switch_sigma:.4f} alpha={alpha:.2f}")

        _apply_alpha(alpha)
        return apply_model(args["input"], timestep, **c)

    return wrapper


# ---------------------------------------------------------------------------
# Apply (live) — two MODELs → switching MODEL (for testing, no disk write)
# ---------------------------------------------------------------------------

class DonutZitHybridApply:
    """
    Live-apply: takes two MODELs (base+lora_early, base+lora_late), computes
    full-rank deltas in memory, and returns a MODEL with the switching wrapper.
    Uses cache keyed on patches_uuid so tweaking switch settings is instant.
    """

    _cache_key = None
    _cache_early_div = None
    _cache_late_div = None
    _cache_n = 0
    _cache_unet_prefix = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_early": ("MODEL",),
                "model_late": ("MODEL",),
                "switch_percent": ("FLOAT", {
                    "default": 0.33, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "blend_width": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "blend_curve": (BLEND_CURVES, {"default": "linear"}),
            }
        }

    RETURN_TYPES = ("MODEL", "ZIT_HYBRID_SETTINGS")
    RETURN_NAMES = ("model", "settings")
    FUNCTION = "apply"
    CATEGORY = "advanced/model"

    def apply(self, model_early, model_late, switch_percent, blend_width, blend_curve):
        cls = type(self)
        cache_key = (model_early.patches_uuid, model_late.patches_uuid)

        if cls._cache_key == cache_key and cls._cache_early_div is not None:
            early_div = cls._cache_early_div
            late_div = cls._cache_late_div
            unet_prefix = cls._cache_unet_prefix
            print(f"[DonutZitHybridApply] cache hit ({cls._cache_n} divergent keys)")
        else:
            # Clear old cache to free memory before allocating
            cls._cache_key = None
            cls._cache_early_div = None
            cls._cache_late_div = None
            cls._cache_unet_prefix = None
            gc.collect()
            print(f"[DonutZitHybridApply] RAM free: {_ram_free_gb():.1f}GB")

            print("[DonutZitHybridApply] baking model_early...")
            sd_early = _bake_model_state_dict(model_early)
            print(f"[DonutZitHybridApply] RAM free: {_ram_free_gb():.1f}GB")

            print("[DonutZitHybridApply] baking model_late...")
            sd_late = _bake_model_state_dict(model_late)
            print(f"[DonutZitHybridApply] RAM free: {_ram_free_gb():.1f}GB")

            unet_prefix = _detect_unet_prefix(sd_early)

            # Diff — keep only divergent keys in original dtype, free the rest
            print("[DonutZitHybridApply] diffing...")
            early_div = {}
            late_div = {}
            for k in list(sd_early.keys()):
                a = sd_early.pop(k)
                b = sd_late.pop(k, None)
                if b is None or not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
                    continue
                if a.shape != b.shape or torch.equal(a, b):
                    continue
                early_div[k] = a
                late_div[k] = b

            del sd_early, sd_late
            gc.collect()
            print(f"[DonutZitHybridApply] RAM free after diff: {_ram_free_gb():.1f}GB")

            cls._cache_key = cache_key
            cls._cache_early_div = early_div
            cls._cache_late_div = late_div
            cls._cache_n = len(early_div)
            cls._cache_unet_prefix = unet_prefix
            print(f"[DonutZitHybridApply] cached {len(early_div)} divergent keys")

        m = model_early.clone()
        wrapper = _build_apply_wrapper(m, early_div, late_div,
                                       float(switch_percent), float(blend_width),
                                       blend_curve, unet_prefix)
        m.set_model_unet_function_wrapper(wrapper)

        settings = {
            "switch_percent": float(switch_percent),
            "blend_width": float(blend_width),
            "blend_curve": blend_curve,
        }
        print(f"[DonutZitHybridApply] switch_percent={switch_percent}, "
              f"blend_width={blend_width}, curve={blend_curve}")
        return (m, settings)


# ---------------------------------------------------------------------------
# Override — change baked settings at runtime
# ---------------------------------------------------------------------------

class DonutZitHybridOverride:
    """Override baked switch settings via transformer_options."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "switch_percent": ("FLOAT", {
                    "default": 0.33, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "blend_width": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "blend_curve": (BLEND_CURVES, {"default": "linear"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "advanced/model"

    def apply(self, model, switch_percent, blend_width, blend_curve):
        m = model.clone()
        to = dict(m.model_options.get("transformer_options", {}))
        to["zit_hybrid_switch_percent"] = float(switch_percent)
        to["zit_hybrid_blend_width"] = float(blend_width)
        to["zit_hybrid_blend_curve"] = blend_curve
        m.model_options["transformer_options"] = to
        return (m,)


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "DonutZitHybridSave": DonutZitHybridSave,
    "DonutZitHybridLoader": DonutZitHybridLoader,
    "DonutZitHybridApply": DonutZitHybridApply,
    "DonutZitHybridOverride": DonutZitHybridOverride,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutZitHybridSave": "ZiT Hybrid Save",
    "DonutZitHybridLoader": "ZiT Hybrid Loader",
    "DonutZitHybridApply": "ZiT Hybrid Apply (Live)",
    "DonutZitHybridOverride": "ZiT Hybrid Switch Override",
}

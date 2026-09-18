"""Native SeedVR2 still-image engine for Donut's existing upscale stages.

Keep all Comfy/model imports behind the selected engine. In particular, an older
ComfyUI without native SeedVR2 must still load and run the normal Donut workflow.
The recipe follows Comfy-Org's utility_seedvr2_3b_int8_upscale_image template:
resize -> preprocess -> SeedVR2 VAE -> conditioning -> KSampler -> decode -> post.
"""
import math
from pathlib import Path


MODEL_3B = "seedvr2_3b_int8_convrot.safetensors"
MODEL_7B = "seedvr2_7b_int8_convrot.safetensors"
VAE_NAME = "seedvr2_ema_vae_fp16.safetensors"
NATIVE_NODES = ("SeedVR2Preprocess", "SeedVR2Conditioning", "SeedVR2PostProcessing")
SHARED_INPUTS = frozenset(("seed", "rescale_factor", "resampling_method"))
DEFAULTS = {
    "seedvr2_model_name": MODEL_3B,
    "seedvr2_vae_name": VAE_NAME,
    "seedvr2_steps": 1,
    "seedvr2_denoise": 1.0,
    "seedvr2_color_correction": "none",
    "seedvr2_vae_tile_size": 512,
}


def _choices(folder, defaults):
    import folder_paths
    try:
        installed = folder_paths.get_filename_list(folder)
    except KeyError:  # Older cores still need to register the disabled engine.
        installed = []
    return list(dict.fromkeys([*defaults, *sorted(installed)]))


def input_types():
    return {
        "upscale_engine": (["Donut", "SeedVR2"], {
            "default": "Donut",
            "tooltip": "Donut keeps the existing upscale + diffusion pass. SeedVR2 uses its own native model/VAE; only the stage seed, scale and resize filter are shared.",
        }),
        "seedvr2_model_name": (_choices("diffusion_models", [MODEL_3B, MODEL_7B]), {"default": MODEL_3B, "lazy": True}),
        "seedvr2_vae_name": (_choices("vae", [VAE_NAME]), {"default": VAE_NAME, "lazy": True}),
        "seedvr2_steps": ("INT", {"default": 1, "min": 1, "max": 100, "lazy": True}),
        "seedvr2_denoise": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01, "lazy": True}),
        "seedvr2_color_correction": (["none", "lab", "wavelet", "adain"], {"default": "none", "lazy": True}),
        "seedvr2_vae_tile_size": ("INT", {"default": 512, "min": 128, "max": 4096, "step": 64, "lazy": True,
            "tooltip": "Native VAE encode/decode tiling only, NOT diffusion tiling. SeedVR2 diffusion still needs enough memory for the full output canvas."}),
    }


def outputs(value):
    """Normalize the native V3 NodeOutput and legacy tuple result contracts."""
    if isinstance(value, dict):
        value = value["result"]
    elif hasattr(value, "result"):
        value = value.result
    if not isinstance(value, (tuple, list)):
        raise TypeError("Unexpected result from a native ComfyUI node.")
    return value


def call_node(registry, node_id, **kwargs):
    cls = registry[node_id]
    instance = cls()
    return outputs(getattr(instance, getattr(cls, "FUNCTION", "execute"))(**kwargs))


def require_native(registry):
    missing = [name for name in NATIVE_NODES if name not in registry]
    if missing:
        raise RuntimeError(
            "SeedVR2 needs a ComfyUI build with native SeedVR2 support. Update "
            "ComfyUI and restart, or select the Donut engine. Missing core nodes: "
            + ", ".join(missing)
        )


def _model_file(folder, name):
    import folder_paths
    path = folder_paths.get_full_path(folder, name)
    if not path or not Path(path).is_file():
        raise FileNotFoundError(
            f"SeedVR2: install {name!r} in models/{folder}/ or choose an installed file. "
            "Use the native Comfy-Org SeedVR2 weights, not a custom-node GGUF checkpoint. "
            "No files are downloaded automatically."
        )
    stat = Path(path).stat()
    return str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns


def dimensions(width, height, scale):
    if not math.isfinite(float(scale)) or not 1.0 <= float(scale) <= 8.0:
        raise ValueError("SeedVR2 scale must be finite and between 1 and 8.")
    # Native post-processing returns even image dimensions. Choose these before
    # resizing, rather than silently cutting a row/column off the intended image.
    return tuple(max(2, math.floor(edge * float(scale) / 2 + 0.5) * 2) for edge in (width, height))


def upscale(owner, image, *, seed=0, rescale_factor=2.0, resampling_method="lanczos", **options):
    import torch
    import nodes

    require_native(nodes.NODE_CLASS_MAPPINGS)
    if not torch.is_tensor(image) or image.ndim != 4 or image.shape[0] < 1 or image.shape[-1] not in (3, 4):
        raise ValueError("SeedVR2 expects a non-empty IMAGE batch shaped B,H,W,3 or B,H,W,4.")
    if min(image.shape[1:3]) < 2 or not image.is_floating_point() or not bool(torch.isfinite(image).all()):
        raise ValueError("SeedVR2 input must contain finite floating-point pixels and be at least 2x2.")
    unknown = set(options) - DEFAULTS.keys()
    if unknown:
        raise TypeError("Unknown SeedVR2 settings: " + ", ".join(sorted(unknown)))
    settings = {**DEFAULTS, **options}
    steps, denoise = settings["seedvr2_steps"], settings["seedvr2_denoise"]
    tile = settings["seedvr2_vae_tile_size"]
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 100:
        raise ValueError("SeedVR2 steps must be an integer between 1 and 100.")
    if not math.isfinite(float(denoise)) or not 0 < float(denoise) <= 1:
        raise ValueError("SeedVR2 denoise must be finite, greater than zero and at most one.")
    if isinstance(tile, bool) or not isinstance(tile, int) or tile < 128 or tile > 4096 or tile % 64:
        raise ValueError("SeedVR2 VAE tile size must be a multiple of 64 between 128 and 4096.")
    if settings["seedvr2_color_correction"] not in ("none", "lab", "wavelet", "adain"):
        raise ValueError("Unknown SeedVR2 color correction method.")
    if resampling_method not in ("lanczos", "nearest", "bilinear", "bicubic"):
        raise ValueError("Unsupported SeedVR2 resize filter.")
    width, height = dimensions(image.shape[2], image.shape[1], rescale_factor)
    key = (_model_file("diffusion_models", settings["seedvr2_model_name"]),
           _model_file("vae", settings["seedvr2_vae_name"]))
    cached = getattr(owner, "_seedvr2_resources", None)
    if cached is None or cached[0] != key:
        # Do not publish a partly loaded pair after an exception.
        owner._seedvr2_resources = None
        model = nodes.UNETLoader().load_unet(settings["seedvr2_model_name"], "default")[0]
        vae = nodes.VAELoader().load_vae(settings["seedvr2_vae_name"])[0]
        owner._seedvr2_resources = (key, model, vae)
    _, model, vae = owner._seedvr2_resources
    registry = nodes.NODE_CLASS_MAPPINGS
    results = []
    with torch.inference_mode():
        # A normal IMAGE batch is a set of unrelated stills. Native preprocessing
        # treats a 4-D batch as video frames, so send one still at a time.
        for index in range(image.shape[0]):
            resized = nodes.ImageScale().upscale(
                image[index:index + 1], "nearest-exact" if resampling_method == "nearest" else resampling_method,
                width, height, "disabled",
            )[0]
            padded = call_node(registry, "SeedVR2Preprocess", resized_images=resized)[0]
            latent = nodes.VAEEncodeTiled().encode(
                vae, padded, tile, tile // 4, temporal_size=4096, temporal_overlap=8,
            )[0]
            positive, negative = call_node(registry, "SeedVR2Conditioning", model=model, vae_conditioning=latent)
            sampled = nodes.KSampler().sample(
                model, (int(seed) + index) % (2 ** 64), steps, 1.0, "euler", "simple",
                positive, negative, latent, denoise=float(denoise),
            )[0]
            decoded = nodes.VAEDecodeTiled().decode(
                vae, sampled, tile, tile // 4, temporal_size=4096, temporal_overlap=8,
            )[0]
            result = call_node(registry, "SeedVR2PostProcessing", images=decoded,
                original_resized_images=resized, color_correction_method=settings["seedvr2_color_correction"])[0]
            if result.ndim == 5 and tuple(result.shape[:2]) == (1, 1):
                result = result[0]
            if tuple(result.shape) != (1, height, width, image.shape[-1]):
                raise RuntimeError(f"Native SeedVR2 returned unexpected still-image shape {tuple(result.shape)}.")
            results.append(result)
    return torch.cat(results, dim=0)

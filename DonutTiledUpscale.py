"""
DonutTiledUpscale - A simple tiled upscale node designed to work with any model type.
Uses the same sampling approach as the core KSampler to ensure compatibility with
Z-Image/Lumina2 and other modern architectures.
"""

import torch
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
from comfy import model_management
from nodes import VAEEncode, VAEDecode, VAEDecodeTiled
from PIL import Image, ImageDraw, ImageFont
import numpy as np

try:
    from .krea2_edit_integration import prepare_krea2_edit
except ImportError:
    from krea2_edit_integration import prepare_krea2_edit


def upscale_with_model(upscale_model, image):
    """Upscale image using a model (adapted from ComfyUI's ImageUpscaleWithModel)"""
    device = model_management.get_torch_device()

    memory_required = model_management.module_size(upscale_model.model)
    memory_required += (512 * 512 * 3) * image.element_size() * max(upscale_model.scale, 1.0) * 384.0
    memory_required += image.nelement() * image.element_size()
    model_management.free_memory(memory_required, device)

    upscale_model.to(device)
    in_img = image.movedim(-1, -3).to(device)

    tile = 512
    overlap = 32

    oom = True
    while oom:
        try:
            steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
                in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap
            )
            pbar = comfy.utils.ProgressBar(steps)
            s = comfy.utils.tiled_scale(
                in_img,
                lambda a: upscale_model(a),
                tile_x=tile, tile_y=tile,
                overlap=overlap,
                upscale_amount=upscale_model.scale,
                pbar=pbar
            )
            oom = False
        except model_management.OOM_EXCEPTION as e:
            tile //= 2
            if tile < 128:
                raise e

    upscale_model.to("cpu")
    s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
    return s


def find_all_valid_configs(input_width, input_height, feather_percent, max_tiles=6):
    """
    Find ALL valid tiling configurations with <0.5% aspect error.

    For each grid (nx x ny), search all ~1MP tile sizes to find configs where:
    - Tile is ~1MP (0.95-1.1 MP)
    - Tile aspect is within 2:1 (0.5 to 2.0)
    - Tile dimensions are multiples of 64
    - Overlap is 5-40% of tile dimension
    - Output aspect matches input aspect (<0.5% error)

    Args:
        input_width, input_height: Original image dimensions
        feather_percent: Target feather percentage (for preference)
        max_tiles: Maximum tiles per axis to search

    Returns:
        List of config dicts sorted by scale
    """
    import math

    TARGET_PIXELS = 1048576
    input_aspect = input_width / input_height

    configs = []

    # 1x1 = original size (no tiling)
    configs.append({
        'tile_width': input_width,
        'tile_height': input_height,
        'overlap_x': 0,
        'overlap_y': 0,
        'step_x': input_width,
        'step_y': input_height,
        'output_width': input_width,
        'output_height': input_height,
        'scale': 1.0,
        'scale_x': 1.0,
        'scale_y': 1.0,
        'nx': 1,
        'ny': 1,
        'feather': 0.0,
        'feather_x': 0.0,
        'feather_y': 0.0,
        'aspect_error': 0.0
    })

    # Search all grid configurations
    for nx in range(1, max_tiles + 1):
        for ny in range(1, max_tiles + 1):
            if nx == 1 and ny == 1:
                continue

            # Search all ~1MP tile sizes (multiples of 64)
            for tile_h in range(512, 1600, 64):
                for tile_w in range(512, 1600, 64):
                    tile_mp = tile_w * tile_h / 1e6

                    # Must be ~1MP (0.95-1.1)
                    if tile_mp < 0.95 or tile_mp > 1.1:
                        continue

                    # Tile aspect must be within 2:1
                    tile_aspect = tile_w / tile_h
                    if tile_aspect < 0.5 or tile_aspect > 2.0:
                        continue

                    # Try overlap_x values (5-40% of tile_w)
                    min_ovl_x = max(8, int(tile_w * 0.05 / 8) * 8)
                    max_ovl_x = int(tile_w * 0.4 / 8) * 8

                    for overlap_x in range(min_ovl_x, max_ovl_x + 1, 8):
                        step_x = tile_w - overlap_x
                        if step_x <= 0:
                            continue

                        output_w = tile_w + (nx - 1) * step_x
                        target_output_h = output_w * input_height / input_width

                        if ny == 1:
                            # Single row: output_h = tile_h
                            output_h = tile_h
                            overlap_y = 0
                            if abs(tile_h - target_output_h) / target_output_h > 0.005:
                                continue
                        else:
                            # Multiple rows: solve for overlap_y
                            step_y_needed = (target_output_h - tile_h) / (ny - 1)
                            if step_y_needed <= 0 or step_y_needed >= tile_h:
                                continue

                            overlap_y = tile_h - step_y_needed
                            overlap_y = round(overlap_y / 8) * 8

                            # Check overlap_y bounds (5-40%)
                            if overlap_y < tile_h * 0.05 or overlap_y > tile_h * 0.4:
                                continue

                            step_y = tile_h - overlap_y
                            output_h = tile_h + (ny - 1) * step_y

                        # Check aspect error
                        output_aspect = output_w / output_h
                        aspect_error = abs(output_aspect - input_aspect) / input_aspect * 100

                        if aspect_error > 0.5:
                            continue

                        step_y = tile_h - overlap_y
                        scale = output_w / input_width

                        configs.append({
                            'tile_width': tile_w,
                            'tile_height': tile_h,
                            'overlap_x': overlap_x,
                            'overlap_y': overlap_y,
                            'step_x': step_x,
                            'step_y': step_y,
                            'output_width': output_w,
                            'output_height': int(output_h),
                            'scale': scale,
                            'scale_x': scale,
                            'scale_y': output_h / input_height,
                            'nx': nx,
                            'ny': ny,
                            'feather': (overlap_x / tile_w + overlap_y / tile_h) / 2 * 100,
                            'feather_x': overlap_x / tile_w * 100,
                            'feather_y': overlap_y / tile_h * 100,
                            'aspect_error': aspect_error
                        })

    # Sort by scale
    configs.sort(key=lambda c: c['scale'])
    return configs


def find_best_tiling(input_width, input_height, target_scale, feather_percent, max_tiles=6):
    """
    Find the best grid configuration that achieves a scale closest to target.

    Searches ALL valid configurations with 0% aspect error and returns the one
    closest to the target scale, preferring configurations with feather closer
    to the requested percentage.

    Args:
        input_width, input_height: Original image dimensions
        target_scale: User's requested scale factor
        feather_percent: Target feather percentage
        max_tiles: Maximum tiles per axis to search

    Returns:
        dict with: nx, ny, tile_width, tile_height, output_width, output_height, scale, etc.
    """
    all_configs = find_all_valid_configs(input_width, input_height, feather_percent, max_tiles)

    if not all_configs:
        raise ValueError("No valid tiling configuration found")

    # Find configs closest to target scale
    for config in all_configs:
        config['scale_err'] = abs(config['scale'] - target_scale)
        config['feather_err'] = abs(config['feather'] - feather_percent)
        config['total_tiles'] = config['nx'] * config['ny']

    # Sort by: scale error first, then feather error, then fewer tiles
    all_configs.sort(key=lambda c: (c['scale_err'], c['feather_err'], c['total_tiles']))

    return all_configs[0]


def snap_scaled_dimensions(width, height, scale, multiple=32):
    """Scale a canvas and snap each edge to the nearest latent-safe multiple."""
    output_width = max(multiple, round(width * scale / multiple) * multiple)
    output_height = max(multiple, round(height * scale / multiple) * multiple)
    return output_width, output_height


def tensor_to_pil(tensor, batch_index=0):
    """Convert a tensor [B,H,W,C] to PIL Image"""
    img = tensor[batch_index].cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)


def pil_to_tensor(image):
    """Convert PIL Image to tensor [1,H,W,C]"""
    arr = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def create_debug_image(output_width, output_height, tile_width, tile_height, overlap_x, overlap_y, num_tiles_x, num_tiles_y,
                       original_width, original_height, actual_scale, target_scale):
    """
    Create a debug visualization showing tile layout and dimensions.
    Shows actual tile boundaries (all same size) with overlap zones highlighted.
    """
    step_x = tile_width - overlap_x
    step_y = tile_height - overlap_y

    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", 11)
        font_large = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
            font_small = font
            font_large = font

    # Color scheme
    text_color = (255, 255, 255)
    dim_text_color = (180, 180, 180)

    # Create base canvas
    debug_img = Image.new('RGB', (output_width, output_height), (40, 40, 40))
    draw = ImageDraw.Draw(debug_img)

    # Draw actual tile rectangles (all same size: tile_width x tile_height)
    # Each tile is offset by step_x/step_y from the previous
    tile_colors = [
        (60, 80, 100),   # Blue-ish
        (80, 60, 100),   # Purple-ish
        (60, 100, 80),   # Green-ish
        (100, 80, 60),   # Orange-ish
        (80, 100, 60),   # Lime-ish
        (100, 60, 80),   # Pink-ish
    ]

    for ty in range(num_tiles_y):
        for tx in range(num_tiles_x):
            # Actual tile position
            x1 = tx * step_x
            y1 = ty * step_y
            x2 = x1 + tile_width
            y2 = y1 + tile_height

            # Color based on tile index for variety
            tile_idx = ty * num_tiles_x + tx
            color = tile_colors[tile_idx % len(tile_colors)]

            # Draw filled rectangle with border
            draw.rectangle([x1, y1, x2, y2], fill=color, outline=(150, 150, 150), width=2)

    # Draw overlap zones on top (semi-transparent effect via darker color)
    # Horizontal seams (between rows)
    if overlap_y > 0:
        for ty in range(1, num_tiles_y):
            seam_y = ty * step_y
            y1 = seam_y
            y2 = seam_y + overlap_y
            draw.rectangle([0, y1, output_width, y2], fill=(120, 60, 60))

    # Vertical seams (between columns)
    if overlap_x > 0:
        for tx in range(1, num_tiles_x):
            seam_x = tx * step_x
            x1 = seam_x
            x2 = seam_x + overlap_x
            draw.rectangle([x1, 0, x2, output_height], fill=(120, 60, 60))

    # Corner overlaps
    if overlap_x > 0 and overlap_y > 0:
        for ty in range(1, num_tiles_y):
            for tx in range(1, num_tiles_x):
                seam_x = tx * step_x
                seam_y = ty * step_y
                draw.rectangle([seam_x, seam_y, seam_x + overlap_x, seam_y + overlap_y],
                             fill=(150, 70, 70))

    # Redraw tile borders on top so they're visible
    for ty in range(num_tiles_y):
        for tx in range(num_tiles_x):
            x1 = tx * step_x
            y1 = ty * step_y
            x2 = x1 + tile_width
            y2 = y1 + tile_height
            draw.rectangle([x1, y1, x2, y2], outline=(200, 200, 200), width=2)

    # Draw tile labels
    for ty in range(num_tiles_y):
        for tx in range(num_tiles_x):
            x1 = tx * step_x
            y1 = ty * step_y
            x2 = x1 + tile_width
            y2 = y1 + tile_height

            tile_num = ty * num_tiles_x + tx + 1
            tile_label = f"Tile {tile_num}"
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            bbox = draw.textbbox((0, 0), tile_label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            draw.text((center_x - text_w // 2, center_y - text_h // 2), tile_label, fill=text_color, font=font)

            dim_label = f"{tile_width}x{tile_height}"
            bbox = draw.textbbox((0, 0), dim_label, font=font_small)
            dim_w = bbox[2] - bbox[0]
            draw.text((center_x - dim_w // 2, center_y + text_h // 2 + 5),
                      dim_label, fill=dim_text_color, font=font_small)

    # Total dimensions label (top-right corner)
    total_label = f"Output: {output_width}x{output_height}"
    bbox = draw.textbbox((0, 0), total_label, font=font_large)
    text_w = bbox[2] - bbox[0]
    draw.rectangle([output_width - text_w - 20, 0, output_width, 25], fill=(60, 60, 60))
    draw.text((output_width - text_w - 10, 5), total_label, fill=text_color, font=font_large)

    # Info box in bottom-left
    info_lines = [
        f"Original: {original_width}x{original_height}",
        f"Target: {target_scale:.2f}x -> Actual: {actual_scale:.2f}x",
        f"Tiles: {num_tiles_x}x{num_tiles_y} = {num_tiles_x * num_tiles_y}",
        f"Tile size: {tile_width}x{tile_height}",
        f"Overlap: {overlap_x}x{overlap_y}px",
        f"Step: {step_x}x{step_y}px",
    ]

    box_height = len(info_lines) * 18 + 10
    box_width = 220
    box_x = 5
    box_y = output_height - box_height - 5

    draw.rectangle([box_x, box_y, box_x + box_width, box_y + box_height],
                   fill=(20, 20, 20, 200), outline=(100, 100, 100))

    for i, line in enumerate(info_lines):
        draw.text((box_x + 8, box_y + 5 + i * 18), line, fill=text_color, font=font_small)

    # Legend in bottom-right
    legend_x = output_width - 150
    legend_y = output_height - 60
    draw.rectangle([legend_x, legend_y, output_width - 5, output_height - 5],
                   fill=(20, 20, 20, 200), outline=(100, 100, 100))
    draw.text((legend_x + 5, legend_y + 5), "Legend:", fill=text_color, font=font_small)
    draw.rectangle([legend_x + 5, legend_y + 22, legend_x + 20, legend_y + 32], outline=(200, 200, 200), width=2)
    draw.text((legend_x + 25, legend_y + 20), "Tile boundary", fill=dim_text_color, font=font_small)
    draw.rectangle([legend_x + 5, legend_y + 37, legend_x + 20, legend_y + 47], fill=(120, 60, 60))
    draw.text((legend_x + 25, legend_y + 35), "Overlap zone", fill=dim_text_color, font=font_small)

    return debug_img


class DonutTiledUpscale:
    """
    Tiled upscale node that processes an image in tiles using img2img.
    Designed to work with any model type including Z-Image/Lumina2.
    """

    @classmethod
    def INPUT_TYPES(cls):
        resampling_methods = ["lanczos", "nearest", "bilinear", "bicubic"]
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_model": ("UPSCALE_MODEL",),
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "rescale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.5, "tooltip": "Upscale factor. Regular mode selects ~1MP tiles; edit mode snaps the full-frame target to a 32-pixel grid."}),
                "resampling_method": (resampling_methods, {"default": "lanczos"}),
                "feather": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 50.0, "step": 1.0, "tooltip": "Regular tiled mode only. Feather/blend zone as percentage of tile size."}),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Regular mode only. Edit mode forces one full-frame regular VAE decode."}),
            },
            "optional": {
                "edit_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Run one full-frame Krea2 img2img upscale against the full reference at the configured denoise; no diffusion or VAE tiling.",
                }),
                "clip": ("CLIP", {"tooltip": "Required when edit_mode is enabled."}),
                "edit_model": ("MODEL", {
                    "tooltip": "Optional Krea2 model with the Identity Edit LoRA already applied. Falls back to model.",
                }),
                "edit_source_image": ("IMAGE", {
                    "tooltip": "Full identity reference used for edit-mode source patching and grounded encoding.",
                }),
                "edit_prompt": ("STRING", {"forceInput": True}),
                "edit_negative_prompt": ("STRING", {"forceInput": True}),
                "grounding_px": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("image", "debug_image",)
    FUNCTION = "upscale"
    CATEGORY = "donut/upscale"

    def upscale(self, image, upscale_model, model, positive, negative, vae, seed,
                steps, cfg, sampler_name, scheduler, denoise,
                rescale_factor, resampling_method, feather, tiled_vae,
                edit_mode=False, clip=None, edit_prompt="Enhance fine details while preserving the source image.",
                edit_negative_prompt="", grounding_px=768, edit_model=None, edit_source_image=None):

        # Resampling filter mapping
        resample_filters = {
            'nearest': Image.NEAREST,
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
            'lanczos': Image.LANCZOS
        }
        resample_filter = resample_filters.get(resampling_method, Image.LANCZOS)

        # Get image dimensions (B, H, W, C)
        batch_size, img_height, img_width, channels = image.shape
        if edit_mode and edit_source_image is None:
            raise ValueError("DonutTiledUpscale edit_mode requires edit_source_image.")

        if edit_mode:
            # Identity edits may move subjects across the frame. Process the full
            # generated canvas against the full reference so identity remains
            # available regardless of its original coordinates.
            output_width, output_height = snap_scaled_dimensions(
                img_width, img_height, rescale_factor,
            )
            config = {
                'nx': 1,
                'ny': 1,
                'tile_width': output_width,
                'tile_height': output_height,
                'overlap_x': 0,
                'overlap_y': 0,
                'output_width': output_width,
                'output_height': output_height,
                'scale': output_width / img_width,
                'step_x': output_width,
                'step_y': output_height,
            }
        else:
            # Regular mode retains the existing approximately 1-MP tile solver.
            config = find_best_tiling(
                img_width, img_height, rescale_factor, feather,
            )

        num_tiles_x = config['nx']
        num_tiles_y = config['ny']
        tile_width = config['tile_width']
        tile_height = config['tile_height']
        overlap_x = config['overlap_x']
        overlap_y = config['overlap_y']
        output_width = config['output_width']
        output_height = config['output_height']
        actual_scale = config['scale']
        step_x = config['step_x']
        step_y = config['step_y']

        print(f"[DonutTiledUpscale] Input: {img_width}x{img_height} (aspect {img_width/img_height:.4f})")
        print(f"[DonutTiledUpscale] Requested: {rescale_factor:.2f}x -> Actual: {actual_scale:.2f}x")
        print(f"[DonutTiledUpscale] Tile: {tile_width}x{tile_height}, {num_tiles_x}x{num_tiles_y} grid")
        print(f"[DonutTiledUpscale] Overlap: {overlap_x}x{overlap_y}px ({feather}%), Step: {step_x}x{step_y}px")
        print(f"[DonutTiledUpscale] Output: {output_width}x{output_height} (aspect {output_width/output_height:.4f})")

        # Get upscale factor from the model
        model_scale = upscale_model.scale
        print(f"[DonutTiledUpscale] Upscale model scale: {model_scale}x")

        # First, upscale with the model
        print(f"[DonutTiledUpscale] Upscaling with model...")
        upscaled_image = upscale_with_model(upscale_model, image)
        _, up_height, up_width, _ = upscaled_image.shape
        print(f"[DonutTiledUpscale] After model upscale: {up_width}x{up_height}")

        total_tiles = num_tiles_x * num_tiles_y
        print(f"[DonutTiledUpscale] Processing {total_tiles} total tiles")

        # VAE encoder/decoder
        vae_encoder = VAEEncode()
        vae_decoder = VAEDecode() if edit_mode or not tiled_vae else VAEDecodeTiled()

        # Process each batch item
        result_images = []

        for b in range(batch_size):
            # Convert model-upscaled image to PIL and resize to output dimensions
            pil_image = tensor_to_pil(upscaled_image, b)
            pil_image = pil_image.resize((output_width, output_height), resample_filter)

            # Create result image at output dimensions
            result_image = Image.new('RGB', (output_width, output_height))

            # Create weight map for blending
            weight_map = Image.new('L', (output_width, output_height), 0)

            tile_idx = 0
            pbar = comfy.utils.ProgressBar(total_tiles)

            for ty in range(num_tiles_y):
                for tx in range(num_tiles_x):
                    tile_idx += 1

                    # Calculate tile coordinates
                    x1 = tx * step_x
                    y1 = ty * step_y
                    x2 = x1 + tile_width
                    y2 = y1 + tile_height

                    # Crop tile from scaled image
                    tile = pil_image.crop((x1, y1, x2, y2))

                    # Convert to tensor for VAE encoding
                    tile_tensor = pil_to_tensor(tile)

                    sampling_model = model
                    sampling_positive = positive
                    sampling_negative = negative
                    if edit_mode:
                        assert edit_source_image is not None
                        reference_index = min(b, len(edit_source_image) - 1)
                        reference_tile = edit_source_image[reference_index].unsqueeze(0)
                        sampling_model, sampling_positive, sampling_negative, _source_latent, _conditioning_image = prepare_krea2_edit(
                            edit_model if edit_model is not None else model,
                            clip, vae, reference_tile, edit_prompt,
                            edit_negative_prompt, grounding_px,
                            tile_width, tile_height,
                        )
                        latent = vae_encoder.encode(vae, tile_tensor)[0]
                    else:
                        # Encode to latent for regular img2img sampling.
                        latent = vae_encoder.encode(vae, tile_tensor)[0]

                    # Sample using the same approach as core KSampler
                    latent_image = latent["samples"]
                    latent_image = comfy.sample.fix_empty_latent_channels(sampling_model, latent_image)

                    # Prepare noise
                    noise = comfy.sample.prepare_noise(latent_image, seed + tile_idx)

                    # Sample
                    callback = latent_preview.prepare_callback(sampling_model, steps)
                    samples = comfy.sample.sample(
                        sampling_model, noise, steps, cfg, sampler_name, scheduler,
                        sampling_positive, sampling_negative, latent_image,
                        denoise=denoise,
                        disable_noise=False,
                        start_step=None,
                        last_step=None,
                        force_full_denoise=False,
                        noise_mask=None,
                        callback=callback,
                        disable_pbar=True,
                        seed=seed + tile_idx
                    )

                    # Decode
                    sampled_latent = {"samples": samples}
                    if tiled_vae and not edit_mode:
                        decoded = vae_decoder.decode(vae, sampled_latent, 512)[0]
                    else:
                        decoded = vae_decoder.decode(vae, sampled_latent)[0]

                    # Convert back to PIL
                    sampled_tile = tensor_to_pil(decoded, 0)

                    # Blend tile into result with feathered edges
                    tile_mask = self._create_blend_mask(tile_width, tile_height, overlap_x, overlap_y,
                                                        tx == 0, ty == 0,
                                                        tx == num_tiles_x - 1, ty == num_tiles_y - 1)
                    result_image = self._blend_tile(result_image, sampled_tile, x1, y1, tile_mask, weight_map)

                    pbar.update(1)

            # Convert result back to tensor
            result_tensor = pil_to_tensor(result_image)
            result_images.append(result_tensor)

        # Combine batch
        output = torch.cat(result_images, dim=0)

        # Generate debug image
        debug_pil = create_debug_image(
            output_width, output_height,
            tile_width, tile_height,
            overlap_x, overlap_y,
            num_tiles_x, num_tiles_y,
            img_width, img_height,
            actual_scale, rescale_factor
        )
        debug_tensor = pil_to_tensor(debug_pil)

        return (output, debug_tensor)

    def _create_blend_mask(self, width, height, overlap_x, overlap_y, is_left, is_top, is_right, is_bottom):
        """Create a feathered blend mask for seamless tiling with separate x/y overlaps"""
        mask = Image.new('L', (width, height), 255)

        if overlap_x <= 0 and overlap_y <= 0:
            return mask

        mask_array = np.array(mask, dtype=np.float32)

        # Feather left edge
        if not is_left and overlap_x > 0:
            for x in range(min(overlap_x, width)):
                mask_array[:, x] *= x / overlap_x

        # Feather right edge
        if not is_right and overlap_x > 0:
            for x in range(min(overlap_x, width)):
                mask_array[:, -(x+1)] *= x / overlap_x

        # Feather top edge
        if not is_top and overlap_y > 0:
            for y in range(min(overlap_y, height)):
                mask_array[y, :] *= y / overlap_y

        # Feather bottom edge
        if not is_bottom and overlap_y > 0:
            for y in range(min(overlap_y, height)):
                mask_array[-(y+1), :] *= y / overlap_y

        return Image.fromarray(mask_array.astype(np.uint8))

    def _blend_tile(self, result, tile, x, y, tile_mask, weight_map):
        """Blend a tile into the result image using weighted averaging"""
        # Get the region from result
        region = result.crop((x, y, x + tile.width, y + tile.height))
        existing_weight = weight_map.crop((x, y, x + tile.width, y + tile.height))

        # Convert to numpy for blending
        region_arr = np.array(region, dtype=np.float32)
        tile_arr = np.array(tile, dtype=np.float32)
        mask_arr = np.array(tile_mask, dtype=np.float32) / 255.0
        existing_arr = np.array(existing_weight, dtype=np.float32)

        # Weighted blend
        new_weight = existing_arr + mask_arr * 255.0

        # Avoid division by zero
        blend_factor = np.where(new_weight > 0,
                                (mask_arr * 255.0) / np.maximum(new_weight, 1.0),
                                1.0)

        # Blend
        blended = region_arr * (1 - blend_factor[:, :, np.newaxis]) + tile_arr * blend_factor[:, :, np.newaxis]
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        # Update result and weight map
        result.paste(Image.fromarray(blended), (x, y))
        weight_map.paste(Image.fromarray(np.clip(new_weight, 0, 255).astype(np.uint8)), (x, y))

        return result


NODE_CLASS_MAPPINGS = {
    "DonutTiledUpscale": DonutTiledUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutTiledUpscale": "Donut Tiled Upscale",
}

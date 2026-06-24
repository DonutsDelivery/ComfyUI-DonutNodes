"""
donut_detailer_core - Shared detailer helpers extracted from DonutFaceDetailer
and DonutUniversalDetailer.

This module removes duplicated logic between the two detailer nodes WITHOUT
changing behavior. The math/pipeline below is copied verbatim from the original
nodes. Where the two original implementations diverged, both behaviors are kept
selectable via parameters rather than picking one.
"""

import math


def filter_segs_by_area(segs, max_count):
    """
    Filter SEGS to keep only the N largest by bounding box area.
    segs format: (shape, list_of_seg)
    """
    shape, seg_list = segs

    if not seg_list or max_count <= 0 or len(seg_list) <= max_count:
        return segs

    def get_area(seg):
        bbox = seg.bbox
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return width * height

    sorted_segs = sorted(seg_list, key=get_area, reverse=True)
    return (shape, sorted_segs[:max_count])


def scale_to_megapixels(w, h, target_pixels, max_resolution=0):
    """
    Calculate new dimensions that maintain aspect ratio and hit target pixel count.
    If max_resolution > 0, clamp so neither dimension exceeds it.
    Returns (new_w, new_h) rounded to nearest 8 pixels.
    """
    current_pixels = w * h
    if current_pixels <= 0:
        return w, h

    scale = math.sqrt(target_pixels / current_pixels)
    new_w = int(round(w * scale / 8) * 8)
    new_h = int(round(h * scale / 8) * 8)

    # Clamp to max_resolution if specified
    if max_resolution > 0 and (new_w > max_resolution or new_h > max_resolution):
        clamp_scale = max_resolution / max(new_w, new_h)
        new_w = int(round(new_w * clamp_scale / 8) * 8)
        new_h = int(round(new_h * clamp_scale / 8) * 8)

    # Ensure minimum size
    new_w = max(64, new_w)
    new_h = max(64, new_h)

    return new_w, new_h


def sample_and_decode(model, seed, steps, cfg, sampler_name, scheduler,
                      positive, negative, latent_image, denoise,
                      vae, impact_sampling, scheduler_func=None):
    """
    Run impact_sampling.ksampler_wrapper then VAE-decode, collapsing the frame
    dimension of video VAEs (5D -> 4D NHWC).

    This is the shared per-seg "ksampler_wrapper -> VAE decode -> 5D->4D reshape"
    step used by both detailers. It is copied verbatim from DonutFaceDetailer
    (which contains the video-VAE reshape). DonutUniversalDetailer previously
    lacked the reshape, but the reshape is a no-op for non-video (4D) VAE output,
    so this preserves both nodes' behavior while adding video-VAE support to the
    universal detailer.

    impact_sampling is passed in by the caller so this module does not import the
    optional Impact Pack at module level.
    """
    # Sample
    refined_latent = impact_sampling.ksampler_wrapper(model, seed, steps, cfg, sampler_name, scheduler,
                                            positive, negative, latent_image, denoise,
                                            scheduler_func=scheduler_func)

    # Decode
    refined_image = vae.decode(refined_latent['samples'])

    # Video VAEs (e.g. WanVAE) return a 5D tensor (batch, frames, H, W, C).
    # Collapse the frame dimension so downstream image ops get 4D NHWC.
    if refined_image.ndim == 5:
        refined_image = refined_image.reshape((-1,) + tuple(refined_image.shape[-3:]))

    return refined_image

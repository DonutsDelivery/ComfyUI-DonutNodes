"""Use the selected 2x VAE at the workflow's configured RGB resolution."""
from copy import copy
from types import MethodType

import nodes
from torch.nn import functional as F


def _is_upscale_vae(vae):
    # Wan's 12-channel head packs four RGB pixels into each decoder position.
    return (getattr(vae, "latent_dim", None) == 3
            and getattr(vae, "latent_channels", None) == 16
            and getattr(vae, "conv_out_channels", None) == 12
            and getattr(vae, "output_channels", None) == 3)


def _original_size(vae, samples, image):
    compression = vae.spacial_compression_decode()
    height, width = samples.shape[-2] * compression, samples.shape[-1] * compression
    if image.ndim != 4 or image.shape[1:] != (height * 2, width * 2, 3):
        raise ValueError("The selected 2x VAE did not return the expected RGB image. Update ComfyUI and ComfyUI-VAE-Utils.")
    # Spacepxl recommends filtering and downsampling the 2x RGB when retaining
    # original resolution. Antialiased reduction supplies the low-pass filter.
    return F.interpolate(image.movedim(-1, 1).float(), size=(height, width),
                         mode="bilinear", align_corners=False, antialias=True).movedim(1, -1).to(image.dtype)


def _decode(vae, samples, vae_options=None):
    image = vae._donut_full_size_decode(vae, samples, {} if vae_options is None else vae_options)
    return _original_size(vae, samples, image)


def _decode_tiled(vae, samples, tile_x=None, tile_y=None, overlap=None, tile_t=None, overlap_t=None):
    image = vae._donut_full_size_decode_tiled(
        vae, samples, tile_x=tile_x, tile_y=tile_y, overlap=overlap, tile_t=tile_t, overlap_t=overlap_t,
    )
    return _original_size(vae, samples, image)


def prepare_vae(vae):
    if not _is_upscale_vae(vae) or getattr(vae, "_donut_vae_original_size", False):
        return vae
    if getattr(vae, "_vae_utils_wan_upscale_patch", False):
        prepared = copy(vae)
    else:
        patch = nodes.NODE_CLASS_MAPPINGS.get("VAEUtils_PatchWanUpscaleVAE")
        if patch is None:
            raise RuntimeError(
                "The selected 2x VAE requires ComfyUI-VAE-Utils by spacepxl. "
                "Install/update that node pack and restart ComfyUI."
            )
        prepared, = patch().patch(vae)
    # Adapt only this VAE's public RGB output. Its encoder, latent layout,
    # decoder weights and managed model patcher remain owned by the VAE.
    prepared._donut_full_size_decode = prepared.decode.__func__
    prepared._donut_full_size_decode_tiled = prepared.decode_tiled.__func__
    prepared.decode = MethodType(_decode, prepared)
    prepared.decode_tiled = MethodType(_decode_tiled, prepared)
    prepared._donut_vae_original_size = True
    return prepared


class DonutVAELoader(nodes.VAELoader):
    CATEGORY = "donut/model"
    DESCRIPTION = "Load the selected VAE's encoder and decoder. The Wan2.1/Qwen 2x VAE decodes internally at 2x, then filters back to the configured image size."

    def load_vae(self, vae_name):
        vae, = super().load_vae(vae_name)
        return (prepare_vae(vae),)


NODE_CLASS_MAPPINGS = {"DonutVAELoader": DonutVAELoader}
NODE_DISPLAY_NAME_MAPPINGS = {"DonutVAELoader": "Donut Load VAE"}

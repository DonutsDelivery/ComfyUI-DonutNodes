"""Run-local NAG preparations for scheduled grounding.

Use the installed NAG node's public patch method, not its closure internals.
Each resolution gets its own immutable negative context. Only the selected
NAG diffusion wrapper is substituted in per-prediction options; the sampler,
model weights, reference geometry and unrelated wrappers are left alone.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


_PREPARATIONS = ContextVar("donut_grounding_nag_preparations", default=None)
_EDIT_KEY = "krea2_edit_normalized_attention_guidance"
_NAG_KEY = "krea2_normalized_attention_guidance"


@contextmanager
def capture_nag_preparations(enabled=True):
    """Keep preparation recipes only for this call, including failure cleanup."""
    token = _PREPARATIONS.set({} if enabled else None)
    try:
        yield
    finally:
        _PREPARATIONS.reset(token)


class _ReferenceVAECache:
    """Share deterministic reference encodes across the NAG wrapper variants.

    The upstream node primes its own pixel cache in patch(). Compare pixel
    CONTENT, not just shape or storage identity: A and B can have equal sizes.
    Cache only its plain encode(pixels) calls, with a small run-local bound.
    Other methods/encode options retain the original VAE behavior.
    """
    def __init__(self, vae):
        self.vae = vae
        self.entries = []

    def __getattr__(self, name):
        return getattr(self.vae, name)

    def encode(self, pixels, *args, **kwargs):
        import torch
        if args or kwargs or not torch.is_tensor(pixels):
            return self.vae.encode(pixels, *args, **kwargs)
        for index, (saved, latent) in enumerate(self.entries):
            if (saved.shape == pixels.shape and saved.dtype == pixels.dtype
                    and saved.device == pixels.device and torch.equal(saved, pixels)):
                self.entries.append(self.entries.pop(index))
                return latent
        # Never retain an entry if the VAE raises, and protect the key from
        # a caller changing the input tensor after encoding.
        saved = pixels.detach().clone()
        latent = self.vae.encode(pixels)
        self.entries.append((saved, latent))
        self.entries = self.entries[-4:]
        return latent


def prepare_nag_arguments(arguments, explicit_negative=False):
    """Only scheduled runs wrap the VAE; constant NAG is an exact pass-through."""
    if (_PREPARATIONS.get() is None or arguments.get("vae") is None
            or explicit_negative or float(arguments.get("phi", 0.0)) == 0.0
            or float(arguments.get("alpha", 0.0)) == 0.0):
        return arguments
    arguments = dict(arguments)
    arguments["vae"] = _ReferenceVAECache(arguments["vae"])
    return arguments


@dataclass(frozen=True)
class NAGPreparation:
    patched: object
    node_class: object
    arguments: dict
    explicit_negative: bool

    @property
    def changes_negative(self):
        # Zero guidance still needs the combined edit forward for fit/ref_boost.
        # Do not disable NAG or replace that forward with Donut's legacy patch.
        return (not self.explicit_negative
                and float(self.arguments["phi"]) != 0.0
                and float(self.arguments["alpha"]) != 0.0)

    def wrappers_for(self, negative=None):
        """Return only NAG's wrapper, preserving the original reference inputs."""
        import comfy.patcher_extension
        try:
            from .krea2_nag_integration import prepare_nag_conditioning
        except ImportError:
            from krea2_nag_integration import prepare_nag_conditioning

        patched = self.patched
        if negative is not None:
            arguments = dict(self.arguments)
            arguments["nag_negative"] = prepare_nag_conditioning(arguments["model"], negative)
            patched = self.node_class().patch(**arguments)[0]
        key = _EDIT_KEY if "source_latent" in self.arguments else _NAG_KEY
        table = getattr(patched, "wrappers", {}).get(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, {},
        )
        wrappers = table.get(key)
        if not wrappers:
            raise RuntimeError(
                f"The installed Krea2 NAG node did not register {key!r}. "
                "Update krea2-nag; scheduled grounding cannot select its negative context."
            )
        return {key: tuple(wrappers)}


def record_nag_preparation(patched, node_class, arguments, explicit_negative):
    """Called by the normal NAG bridge; constant/non-edit runs retain no recipe."""
    preparations = _PREPARATIONS.get()
    if preparations is not None:
        preparations[id(patched)] = NAGPreparation(
            patched, node_class, dict(arguments), explicit_negative,
        )


def get_nag_preparation(model):
    preparations = _PREPARATIONS.get()
    if preparations is None:
        return None
    preparation = preparations.get(id(model))
    return preparation if preparation is not None and preparation.patched is model else None


def select_nag_options(model_options, replacements):
    """Copy just the option dictionaries we change. Never mutate a shared model.

    ComfyUI merges ModelPatcher wrappers into transformer_options before the
    sampler runs. Replace only the registered NAG key there; keep all other
    wrappers, patches, sigma information and upstream sampler options intact.
    """
    import comfy.patcher_extension

    kind = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    options = dict(model_options or {})
    transformer = dict(options.get("transformer_options", {}))
    wrappers = dict(transformer.get("wrappers", {}))
    diffusion = dict(wrappers.get(kind, {}))
    for key, selected in replacements.items():
        if key not in diffusion:
            raise RuntimeError(
                f"Scheduled NAG wrapper {key!r} is missing from sampling options. "
                "Update ComfyUI and krea2-nag; the negative was not silently left static."
            )
        diffusion[key] = list(selected)
    wrappers[kind] = diffusion
    transformer["wrappers"] = wrappers
    options["transformer_options"] = transformer
    return options

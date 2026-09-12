"""Bound Krea2 feed-forward intermediates without changing attention geometry.

``DONUT_KREA2_REUSE_CHUNK_WEIGHTS=1`` keeps an MLP's dynamic-VRAM projections
prefetched while its token chunks run. The opt-in switch limits the memory
trade-off to callers that have measured a benefit on their hardware.
"""
import os
import torch


class TokenChunkedMLP:
    def __init__(self, forward, chunk_size=1024, reuse_weights=None):
        self.forward = forward
        self.chunk_size = chunk_size
        self.module = getattr(forward, "__self__", None)
        if reuse_weights is None:
            reuse_weights = os.environ.get("DONUT_KREA2_REUSE_CHUNK_WEIGHTS", "") == "1"
        self.reuse_weights = bool(reuse_weights)

    def _stage_weights(self, x):
        """Keep dynamic-vram MLP weights prefetched until all chunks finish."""
        if self.module is None:
            return None
        modules = []
        for name in ("gate", "up", "down"):
            module = getattr(self.module, name, None)
            if module is not None and hasattr(module, "_v"):
                modules.append(module)
        if not modules:
            return None
        try:
            import comfy.model_management
            import comfy.ops
            stream = comfy.ops.cast_modules_with_vbar(
                modules, None, x.device, None, True
            )
            comfy.model_management.sync_stream(x.device, stream)
            return modules
        except (ImportError, AttributeError, RuntimeError):
            # Unit tests and non-Comfy callers should retain the old behavior.
            # A partial cast can leave _prefetch markers behind, so clean those
            # markers before falling back to per-chunk casting.
            self._release_weights(modules)
            return None

    def _release_weights(self, modules):
        if not modules or self.module is None:
            return
        try:
            import comfy.model_prefetch
            comfy.model_prefetch.cleanup_prefetched_modules(self.module, modules)
        except (ImportError, AttributeError, RuntimeError):
            # Best effort cleanup; Comfy's normal model cleanup remains authoritative.
            for module in modules:
                module.__dict__.pop("_prefetch", None)

    def __call__(self, x):
        # Krea2 SwiGLU acts independently on each token. Keep training/autograd
        # untouched; this optimization is for inference's large B,T,C tensors.
        if torch.is_grad_enabled() or x.ndim != 3 or x.shape[1] <= self.chunk_size:
            return self.forward(x)
        staged = self._stage_weights(x) if self.reuse_weights else None
        try:
            output = None
            for start in range(0, x.shape[1], self.chunk_size):
                end = min(start + self.chunk_size, x.shape[1])
                chunk = self.forward(x[:, start:end])
                if output is None:
                    output = chunk.new_empty((chunk.shape[0], x.shape[1], chunk.shape[2]))
                output[:, start:end].copy_(chunk)
                del chunk
            return output
        finally:
            self._release_weights(staged)


def patch_krea2_upscale_memory(model):
    """Clone-persistent object patches leave linear-layer bypass hooks active."""
    root = getattr(getattr(model, "model", None), "diffusion_model", None)
    if root is None or not all(hasattr(root, name) for name in ("blocks", "txtfusion", "tproj")):
        return model
    patched = model.clone()
    count = 0
    for index, block in enumerate(root.blocks):
        mlp = getattr(block, "mlp", None)
        if mlp is None or not all(hasattr(mlp, name) for name in ("gate", "up", "down")):
            continue
        path = f"diffusion_model.blocks.{index}.mlp.forward"
        forward = patched.get_model_object(path)
        if not isinstance(forward, TokenChunkedMLP):
            patched.add_object_patch(path, TokenChunkedMLP(forward))
            count += 1
    if count:
        reuse_note = " with bounded MLP-weight reuse" if os.environ.get("DONUT_KREA2_REUSE_CHUNK_WEIGHTS", "") == "1" else ""
        print(f"[DonutTiledUpscale] Full-frame edit: chunking {count} MLPs at 1024 tokens{reuse_note}; attention and references remain full-frame")
    return patched

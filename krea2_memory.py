"""Bound Krea2 feed-forward intermediates without changing attention geometry."""
import torch


class TokenChunkedMLP:
    def __init__(self, forward, chunk_size=1024):
        self.forward = forward
        self.chunk_size = chunk_size

    def __call__(self, x):
        # Krea2 SwiGLU acts independently on each token. Keep training/autograd
        # untouched; this optimization is for inference's large B,T,C tensors.
        if torch.is_grad_enabled() or x.ndim != 3 or x.shape[1] <= self.chunk_size:
            return self.forward(x)
        output = None
        for start in range(0, x.shape[1], self.chunk_size):
            end = min(start + self.chunk_size, x.shape[1])
            chunk = self.forward(x[:, start:end])
            if output is None:
                output = chunk.new_empty((chunk.shape[0], x.shape[1], chunk.shape[2]))
            output[:, start:end].copy_(chunk)
            del chunk
        return output


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
        print(f"[DonutTiledUpscale] Full-frame edit: chunking {count} MLPs at 1024 tokens; attention and references remain full-frame")
    return patched

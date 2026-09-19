"""Bound Krea2 feed-forward intermediates without changing attention geometry."""
import torch


class TokenChunkedMLP:
    def __init__(self, forward, chunk_size=1024, axis=1):
        self.forward = forward
        self.chunk_size = chunk_size
        self.axis = axis

    def __call__(self, x):
        # Krea2 SwiGLU acts independently on each token. Keep training/autograd
        # untouched; this optimization is for inference's large B,T,C tensors.
        if torch.is_grad_enabled() or x.ndim < 3 or x.shape[self.axis] <= self.chunk_size:
            return self.forward(x)
        output = None
        for start in range(0, x.shape[self.axis], self.chunk_size):
            end = min(start + self.chunk_size, x.shape[self.axis])
            index = [slice(None)] * x.ndim
            index[self.axis] = slice(start, end)
            index = tuple(index)
            chunk = self.forward(x[index])
            if output is None:
                shape = list(chunk.shape)
                shape[self.axis] = x.shape[self.axis]
                output = chunk.new_empty(shape)
            output[index].copy_(chunk)
            del chunk
        return output


def patch_krea2_upscale_memory(model):
    """Clone-persistent object patches leave linear-layer bypass hooks active."""
    options = getattr(model, "model_options", {})
    chunk_mlp = options.get("donut_chunk_edit_mlp", False)
    chunk_norm = options.get("donut_chunk_edit_norm", False)
    if not (chunk_mlp or chunk_norm):
        return model
    root = getattr(getattr(model, "model", None), "diffusion_model", None)
    if root is None or not all(hasattr(root, name) for name in ("blocks", "txtfusion", "tproj")):
        return model
    patched = model.clone()
    count = 0
    for index, block in enumerate(root.blocks):
        # RMSNorm reduces only the final feature axis. Its FP32 temporaries
        # can likewise be bounded without splitting the attention operation.
        qknorm = getattr(getattr(block, "attn", None), "qknorm", None)
        for name in ("qnorm", "knorm"):
            if chunk_norm and getattr(qknorm, name, None) is not None:
                path = f"diffusion_model.blocks.{index}.attn.qknorm.{name}.forward"
                forward = patched.get_model_object(path)
                if not isinstance(forward, TokenChunkedMLP):
                    patched.add_object_patch(path, TokenChunkedMLP(forward, axis=-2))
        mlp = getattr(block, "mlp", None)
        if not chunk_mlp or mlp is None or not all(hasattr(mlp, name) for name in ("gate", "up", "down")):
            continue
        path = f"diffusion_model.blocks.{index}.mlp.forward"
        forward = patched.get_model_object(path)
        if not isinstance(forward, TokenChunkedMLP):
            patched.add_object_patch(path, TokenChunkedMLP(forward))
            count += 1
    if count:
        print(f"[Donut Krea2] Full-frame edit: chunking {count} MLPs at 1024 tokens; attention and references remain full-frame")
    return patched

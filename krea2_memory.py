"""Bound Krea2 feed-forward intermediates without changing attention geometry."""
import torch


class TokenChunkedMLP:
    def __init__(self, forward, chunk_size=1024, axis=1):
        self.forward = forward
        self.chunk_size = chunk_size
        self.axis = axis

    def __call__(self, x):
        # SwiGLU and RMSNorm act independently on each token. Axis 0 also
        # handles txtfusion's flattened (B*T),L,C layout. Keep autograd
        # untouched; only inference uses the preallocated chunk output.
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


_EDIT_WRAPPER_KEYS = ("donut_krea2_edit", "krea2_edit")


def _has_edit_wrapper(model, options):
    """Recognize active edit forwards, not an unused Edit Studio side branch."""
    transformer_options = options.get("transformer_options", {})
    owners = [getattr(model, "wrappers", None)]
    if isinstance(transformer_options, dict):
        owners.append(transformer_options.get("wrappers"))
    for wrappers in owners:
        if not isinstance(wrappers, dict):
            continue
        for keyed in wrappers.values():
            if isinstance(keyed, dict) and any(keyed.get(key) for key in _EDIT_WRAPPER_KEYS):
                return True
    return False


def _block_memory_targets(block, prefix, token_axis=1, qk_axis=-2, chunk_size=1024):
    """Yield pointwise forwards only; never split an attention operation."""
    mlp = getattr(block, "mlp", None)
    if mlp is not None and all(hasattr(mlp, name) for name in ("gate", "up", "down")):
        yield f"{prefix}.mlp.forward", "mlp", token_axis, chunk_size
    for name in ("prenorm", "postnorm"):
        if callable(getattr(getattr(block, name, None), "forward", None)):
            yield f"{prefix}.{name}.forward", "norm", token_axis, chunk_size
    qknorm = getattr(getattr(block, "attn", None), "qknorm", None)
    for name in ("qnorm", "knorm"):
        if callable(getattr(getattr(qknorm, name, None), "forward", None)):
            yield f"{prefix}.attn.qknorm.{name}.forward", "norm", qk_axis, chunk_size


def _memory_targets(root):
    for index, block in enumerate(root.blocks):
        yield from _block_memory_targets(block, f"diffusion_model.blocks.{index}")
    for group in ("layerwise_blocks", "refiner_blocks"):
        blocks = getattr(root.txtfusion, group, ())
        for index, block in enumerate(blocks):
            prefix = f"diffusion_model.txtfusion.{group}.{index}"
            if group == "layerwise_blocks":
                # Upstream flattens B*T into the batch axis here. The sequence
                # axis contains encoder layers, not image/text positions. At
                # 12 encoder layers, 64 rows bound MLP work to 768 layer tokens.
                # Q/K have shape (B*T),heads,layers,head_dim: chunk axis 0 too.
                yield from _block_memory_targets(block, prefix, 0, 0, 64)
            else:
                yield from _block_memory_targets(block, prefix)


def patch_krea2_upscale_memory(model):
    """Bound active edit forwards with clone-local, hook-preserving patches.

    Existing edit sampler/upscale/detailer call sites share this helper. Active
    edit wrappers opt in automatically; non-edit models remain unchanged unless
    explicitly opted in. The existing MLP/norm options remain independent and
    explicit False disables that family, including on an already-patched clone.
    """
    options = getattr(model, "model_options", {})
    if not isinstance(options, dict):
        options = {}
    root = getattr(getattr(model, "model", None), "diffusion_model", None)
    if root is None or not all(hasattr(root, name) for name in ("blocks", "txtfusion", "tproj")):
        return model
    active_edit = _has_edit_wrapper(model, options)
    enabled = {
        "mlp": options.get("donut_chunk_edit_mlp", active_edit),
        "norm": options.get("donut_chunk_edit_norm", active_edit),
    }
    patched = model
    counts = {"mlp": 0, "norm": 0}
    for path, family, axis, chunk_size in _memory_targets(root):
        forward = patched.get_model_object(path)
        if enabled[family]:
            if isinstance(forward, TokenChunkedMLP):
                continue
            replacement = TokenChunkedMLP(forward, chunk_size, axis)
            counts[family] += 1
        elif isinstance(forward, TokenChunkedMLP):
            # Restore only our own wrapper, retaining any upstream forward.
            replacement = forward.forward
        else:
            continue
        if patched is model:
            patched = model.clone()
        patched.add_object_patch(path, replacement)
    if any(counts.values()):
        print(
            f"[Donut Krea2] Edit memory: chunking {counts['mlp']} MLPs and "
            f"{counts['norm']} norms (1024 tokens; txtfusion layerwise 64 rows); "
            "attention and references remain full-frame"
        )
    return patched

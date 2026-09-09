"""Shared variance controls for full/face prompts and regenerated edit prompts."""

VARIANCE_KEY = "donut_krea2_seed_variance"


def variance_input_types():
    return {
        "variance_enabled": ("BOOLEAN", {"default": False}),
        "variance_randomize_percent": ("FLOAT", {"default": 50., "min": 1., "max": 100., "step": 1.}),
        "variance_auto_strength_factor": ("FLOAT", {"default": 1., "min": 0., "max": 100., "step": .05}),
        "variance_strength": ("FLOAT", {"default": 20., "min": -0xFFFFFFFF, "max": 0xFFFFFFFF, "step": .00001}),
        "variance_noise_insert": (["noise on beginning steps", "noise on ending steps", "noise on all steps", "disabled"],),
        "variance_steps_switchover_percent": ("FLOAT", {"default": 25., "min": 1., "max": 99., "step": 1.}),
        "variance_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                                  "tooltip": "Connect the sampling seed for a new variation each run. Face conditioning uses seed + 1."}),
        "variance_mask_starts_at": (["beginning", "end"],),
        "variance_mask_percent": ("FLOAT", {"default": 0., "min": 0., "max": 99., "step": 1.}),
        "variance_log_to_console": ("BOOLEAN", {"default": False}),
        "variance_noise_distribution": (["uniform", "gaussian"],),
        "variance_granularity": (["values", "tokens"],),
    }


def apply_seed_variance(conditioning, settings):
    # Resolve the optional pack only when enabled, after custom-node registration.
    import nodes
    node_class = nodes.NODE_CLASS_MAPPINGS.get("KreaSeedVarianceEnhancer")
    if node_class is None:
        raise RuntimeError("Seed variance requires krea-seed-variance-enhancer. Install/enable it and restart ComfyUI.")
    enhanced, _diagnostics = node_class().randomize_conditioning(conditioning, **settings)
    # Carry the recipe with positive conditioning so edit-mode re-encoding can
    # apply it to fresh grounded embeddings, without adding more workflow wires.
    return [[tensor, {**metadata, VARIANCE_KEY: dict(settings)}] for tensor, metadata in enhanced]


def enhance_prompt_pair(positive, face_positive, *, variance_enabled=False,
                        variance_randomize_percent=50., variance_auto_strength_factor=1.,
                        variance_strength=20., variance_noise_insert="noise on beginning steps",
                        variance_steps_switchover_percent=25., variance_seed=0,
                        variance_mask_starts_at="beginning", variance_mask_percent=0.,
                        variance_log_to_console=False, variance_noise_distribution="uniform",
                        variance_granularity="values"):
    if not variance_enabled:
        return positive, face_positive
    settings = dict(randomize_percent=variance_randomize_percent,
                    auto_strength_factor=variance_auto_strength_factor,
                    strength=variance_strength, noise_insert=variance_noise_insert,
                    steps_switchover_percent=variance_steps_switchover_percent,
                    seed=variance_seed, mask_starts_at=variance_mask_starts_at,
                    mask_percent=variance_mask_percent, log_to_console=variance_log_to_console,
                    noise_distribution=variance_noise_distribution, granularity=variance_granularity)
    general = apply_seed_variance(positive, settings)
    face = apply_seed_variance(face_positive, {**settings, "seed": (variance_seed + 1) % (1 << 64)})
    return general, face


def reapply_edit_variance(grounded_positive, original_positive):
    for _tensor, metadata in original_positive:
        settings = metadata.get(VARIANCE_KEY)
        if settings is not None:
            return apply_seed_variance(grounded_positive, settings)
    return grounded_positive

"""Explicit seed domains: one text seed, distinct reproducible sampling stages."""
MAX_SEED = 2**53 - 1  # Lossless across Python, JSON and JavaScript.


def stage_seeds(seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be an integer between 0 and {MAX_SEED}")
    # Keep the old base/upscale offsets; fix the old upscale-2/face collision.
    return tuple((seed + offset) % (MAX_SEED + 1) for offset in (0, 2, 3, 4))


class DonutSeedPlan:
    @classmethod
    def INPUT_TYPES(cls):
        seed = lambda: ("INT", {"default": 0, "min": 0, "max": MAX_SEED, "control_after_generate": True})
        return {"required": {"text_seed": seed(), "sampler_seed": seed(), "filename_seed": seed()}}
    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("text", "base", "upscale_1", "upscale_2", "face", "filename")
    CATEGORY = "donut/control"
    FUNCTION = "generate"

    def generate(self, text_seed, sampler_seed, filename_seed):
        stage_seeds(text_seed)
        stage_seeds(filename_seed)
        stages = stage_seeds(sampler_seed)
        return (text_seed, *stages, str(filename_seed))


NODE_CLASS_MAPPINGS = {"DonutSeedPlan": DonutSeedPlan}
NODE_DISPLAY_NAME_MAPPINGS = {"DonutSeedPlan": "Donut Seeds · Text / Samplers / Filename"}

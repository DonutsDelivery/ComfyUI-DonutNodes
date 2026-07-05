"""
DonutWeightVectorScale

Scales a comma/semicolon-separated weight vector around a 1.0 baseline,
instead of multiplying the raw values directly. Each entry's deviation from
1.0 is multiplied by `scale`: scale=1.0 reproduces the input unchanged,
scale=0.0 collapses every entry back to 1.0 (neutral/no-op), and scale>1
amplifies the existing deviations.

Originally written to fix the (third-party) ConditioningKrea2Rebalance node,
whose own "multiplier" scales every per-layer weight raw - including entries
left at 1.0 - so multiplier=0 zeroes even the "neutral" layers instead of
leaving them untouched. Feed this node's output into per_layer_weights and
leave that node's own multiplier at 1.0.

Also applies to LoRA block-weight vectors (e.g. DonutLoRAStack's
block_vector fields), which use the same 1.0 = full-strength / neutral
convention.

The optional `rebalance` toggle renormalizes the scaled vector so its mean
is exactly 1.0 (sum / n = 1). That makes it an actual rebalance rather than
a pure gain: raising one layer's weight proportionally lowers the others,
holding the overall conditioning strength (and therefore effective CFG)
constant instead of letting it drift with the sum of the weights.
"""


class DonutWeightVectorScale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "weights": ("STRING", {
                    "default": "1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0",
                    "multiline": False,
                }),
                "scale": ("FLOAT", {
                    "default": 1.0, "min": -1000.0, "max": 1000.0, "step": 0.01,
                    "tooltip": "Scales each weight's deviation from 1.0. "
                               "1.0 = unchanged, 0.0 = all weights become 1.0 (neutral).",
                }),
                "rebalance": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Renormalize so the weights' mean is 1.0 - raising one "
                               "layer proportionally lowers the others, keeping overall "
                               "conditioning strength (and effective CFG) constant.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("weights",)
    FUNCTION = "scale_weights"
    CATEGORY = "Donut/conditioning"

    @staticmethod
    def _parse(s):
        parts = [p.strip() for chunk in s.split(";") for p in chunk.split(",")]
        return [float(p) for p in parts if p]

    def scale_weights(self, weights, scale, rebalance=False):
        vals = self._parse(weights)
        scaled = [1.0 + scale * (v - 1.0) for v in vals]

        if rebalance and scaled:
            mean = sum(scaled) / len(scaled)
            if abs(mean) > 1e-9:
                scaled = [v / mean for v in scaled]

        out = ",".join(f"{v:g}" for v in scaled)
        return (out,)


NODE_CLASS_MAPPINGS = {
    "DonutWeightVectorScale": DonutWeightVectorScale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutWeightVectorScale": "Donut Weight-Vector Scale",
}

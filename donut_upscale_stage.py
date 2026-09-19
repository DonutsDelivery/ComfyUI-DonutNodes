"""Lazy engine selection for both existing Donut upscale/finishing stages."""
from copy import deepcopy
from . import donut_seedvr2
from .DonutTiledUpscale import NODE_CLASS_MAPPINGS as _nodes
_Base = _nodes["DonutTiledUpscale"]


class DonutTiledUpscaleStage(_Base):
    @classmethod
    def INPUT_TYPES(cls):
        result = deepcopy(_Base.INPUT_TYPES())
        for section in ("required", "optional"):
            for name, spec in list(result.get(section, {}).items()):
                if name != "image":
                    result[section][name] = (spec[0], {**(spec[1] if len(spec) > 1 else {}), "lazy": True})
        result.setdefault("optional", {})["enabled"] = ("BOOLEAN", {"default": True})
        # Append after enabled: old positional widget values retain their meaning.
        result["optional"].update(donut_seedvr2.input_types())
        return result
    FUNCTION = "run_stage"

    def check_lazy_status(self, image, enabled=True, upscale_engine="Donut", **kwargs):
        if not enabled:
            return []
        if upscale_engine == "SeedVR2":
            required = donut_seedvr2.SHARED_INPUTS | donut_seedvr2.DEFAULTS.keys()
            return [key for key, value in kwargs.items() if key in required and value is None]
        if upscale_engine != "Donut":
            raise ValueError(f"Unknown upscale engine: {upscale_engine}")
        return [key for key, value in kwargs.items() if key not in donut_seedvr2.DEFAULTS and value is None]

    def run_stage(self, image, enabled=True, upscale_engine="Donut", **kwargs):
        if not enabled:
            return (image, image)
        if upscale_engine == "SeedVR2":
            selected = donut_seedvr2.SHARED_INPUTS | donut_seedvr2.DEFAULTS.keys()
            result = donut_seedvr2.upscale(self, image, **{key: value for key, value in kwargs.items() if key in selected})
            # Preserve the stage's two-IMAGE contract and all existing downstream
            # inpaint restoration, preview, detailer and save connections.
            return (result, result)
        if upscale_engine != "Donut":
            raise ValueError(f"Unknown upscale engine: {upscale_engine}")
        self._seedvr2_resources = None
        legacy = {key: value for key, value in kwargs.items() if key not in donut_seedvr2.DEFAULTS}
        return getattr(_Base, _Base.FUNCTION)(self, image=image, **legacy)


NODE_CLASS_MAPPINGS = {"DonutTiledUpscale": DonutTiledUpscaleStage}
NODE_DISPLAY_NAME_MAPPINGS = {"DonutTiledUpscale": "Donut Tiled Upscale"}

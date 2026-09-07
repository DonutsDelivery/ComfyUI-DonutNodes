"""Add a real, lazy enable switch without replacing the upscale algorithm."""
from copy import deepcopy
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
        return result
    FUNCTION = "run_stage"

    def check_lazy_status(self, image, enabled=True, **kwargs):
        return [key for key, value in kwargs.items() if value is None] if enabled else []

    def run_stage(self, image, enabled=True, **kwargs):
        if not enabled:
            return (image, image)
        return getattr(_Base, _Base.FUNCTION)(self, image=image, **kwargs)


NODE_CLASS_MAPPINGS = {"DonutTiledUpscale": DonutTiledUpscaleStage}
NODE_DISPLAY_NAME_MAPPINGS = {"DonutTiledUpscale": "Donut Tiled Upscale"}

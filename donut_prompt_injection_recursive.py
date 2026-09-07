"""Keep all existing style controls/UI; resolve their output with the text seed."""
from copy import deepcopy
from .DonutPromptInjection import NODE_CLASS_MAPPINGS as _nodes
from .donut_prompt import expand_text, directory_fingerprint
_Base = _nodes["DonutPromptInjection"]


class DonutPromptInjectionRecursive(_Base):
    @classmethod
    def INPUT_TYPES(cls):
        result = deepcopy(_Base.INPUT_TYPES())
        result.setdefault("optional", {}).update({
            "wildcard_depth": ("INT", {"default": 128, "min": 1, "max": 1024}),
            "missing_wildcard": (["error", "keep", "empty"],),
        })
        # Disable client-side dynamic prompt expansion; Python owns the seed.
        spec = result["required"]["prompt"]
        result["required"]["prompt"] = (spec[0], {**(spec[1] if len(spec) > 1 else {}), "dynamicPrompts": False})
        return result
    FUNCTION = "process_recursive"

    def process_recursive(self, wildcard_depth=128, missing_wildcard="error", **kwargs):
        result = getattr(_Base, _Base.FUNCTION)(self, **kwargs)
        values = list(result["result"] if isinstance(result, dict) else result)
        values[0] = expand_text(values[0], kwargs.get("seed", 0), wildcard_depth, missing_wildcard)
        if isinstance(result, dict):
            return {**result, "result": tuple(values)}
        return tuple(values)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return directory_fingerprint()


NODE_CLASS_MAPPINGS = {"DonutPromptInjection": DonutPromptInjectionRecursive}
NODE_DISPLAY_NAME_MAPPINGS = {"DonutPromptInjection": "Donut Prompt Injection"}

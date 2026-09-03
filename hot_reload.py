"""Disabled DonutNodes hot-reload control and experimental-node aggregation.

Hot reloading remains intentionally disabled because replacing ComfyUI node
classes in a live process can invalidate existing workflows. This module is
already imported by the package root, so the experimental DLSS 5 branch also
merges its node mappings here without changing stable node identifiers.
"""

from __future__ import annotations


class DonutHotReload:
    """Compatibility facade for the existing, intentionally disabled feature."""

    def __init__(self) -> None:
        self.watching = False

    def reload_modules(self) -> bool:
        print(
            "[DonutHotReload] Module reload disabled to prevent interference "
            "with other extensions"
        )
        return False

    def start_watching(self) -> bool:
        print(
            "[DonutHotReload] Hot reload watching disabled to prevent "
            "interference with other extensions"
        )
        self.watching = False
        return False

    def stop_watching(self) -> None:
        self.watching = False


hot_reload = DonutHotReload()
hot_reload.stop_watching()
print("[DonutHotReload] Stopped any existing file watching on import")


class DonutHotReloadNode:
    """ComfyUI node preserving the original DonutHotReload workflow ID."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (
                    ["start_watching", "stop_watching", "reload_now"],
                    {"default": "start_watching"},
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "execute"
    CATEGORY = "donut/dev"
    OUTPUT_NODE = True

    def execute(self, action):
        if action == "start_watching":
            hot_reload.start_watching()
            status = "Hot reload watching started"
        elif action == "stop_watching":
            hot_reload.stop_watching()
            status = "Hot reload watching stopped"
        elif action == "reload_now":
            status = "Reload successful" if hot_reload.reload_modules() else "Reload failed"
        else:
            status = "Unknown action"
        print(f"[DonutHotReload] {status}")
        return (status,)


from .DonutDLSS5Linux import (  # noqa: E402
    NODE_CLASS_MAPPINGS as _DLSS5_NODE_CLASS_MAPPINGS,
)
from .DonutDLSS5Linux import (  # noqa: E402
    NODE_DISPLAY_NAME_MAPPINGS as _DLSS5_NODE_DISPLAY_NAME_MAPPINGS,
)

NODE_CLASS_MAPPINGS = {
    "DonutHotReload": DonutHotReloadNode,
    **_DLSS5_NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutHotReload": "DonutHotReload",
    **_DLSS5_NODE_DISPLAY_NAME_MAPPINGS,
}

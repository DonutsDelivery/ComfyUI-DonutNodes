# existing nodes’ class mappings
from .DonutDetailer         import NODE_CLASS_MAPPINGS as m1
from .DonutDetailer2        import NODE_CLASS_MAPPINGS as m2
from .DonutDetailer4        import NODE_CLASS_MAPPINGS as m3
from .DonutDetailer5        import NODE_CLASS_MAPPINGS as m4
from .DonutDetailerXLBlocks import NODE_CLASS_MAPPINGS as m5
from .DonutClipEncode       import NODE_CLASS_MAPPINGS as m6
from .DonutWidenMerge       import NODE_CLASS_MAPPINGS as m7

# new LoRA nodes (include display names)
from .donut_lora_nodes      import NODE_CLASS_MAPPINGS        as m_lora
from .donut_lora_nodes      import NODE_DISPLAY_NAME_MAPPINGS as d_lora

# hot reload functionality
from .hot_reload            import NODE_CLASS_MAPPINGS        as m_reload
from .hot_reload            import NODE_DISPLAY_NAME_MAPPINGS as d_reload

# build globals
NODE_CLASS_MAPPINGS = {
    **m1, **m2, **m3, **m4, **m5,
    **m6, **m7,
    **m_lora,
    **m_reload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **d_lora,
    **d_reload,
}

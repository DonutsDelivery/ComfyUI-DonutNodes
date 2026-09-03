import os
import sys
import importlib
import threading
import time
from pathlib import Path
from typing import Dict, Any

class DonutHotReload:
    """Hot reload functionality for DonutNodes custom node"""
    
    def __init__(self):
        self.watching = False
        self.watch_thread = None
        self.node_modules = [
            'DonutDetailer',
            'DonutDetailer2', 
            'DonutDetailer4',
            'DonutDetailer5',
            'DonutDetailerXLBlocks',
            'DonutClipEncode',
            'DonutWidenMerge',
            'donut_lora_nodes',
            'lora_block_weight',
            'merging_methods',
        ]
        self.base_path = Path(__file__).parent
        self.file_timestamps = {}
        self._update_timestamps()
    
    def _update_timestamps(self):
        """Update file modification timestamps"""
        for module_name in self.node_modules:
            file_path = self.base_path / f"{module_name}.py"
            if file_path.exists():
                self.file_timestamps[str(file_path)] = file_path.stat().st_mtime
    
    def _has_file_changed(self) -> bool:
        """Check if any tracked files have changed"""
        for file_path, old_time in self.file_timestamps.items():
            if os.path.exists(file_path):
                new_time = os.path.getmtime(file_path)
                if new_time > old_time:
                    print(f"[DonutHotReload] Detected change in {file_path}")
                    return True
        return False
    
    def reload_modules(self) -> bool:
        """Reload all DonutNodes modules - DISABLED to prevent interference"""
        print("[DonutHotReload] Module reload disabled to prevent interference with other extensions")
        return False
        
        # DISABLED CODE BELOW
        """
        try:
            print("[DonutHotReload] Starting module reload...")
            
            # Get the main ComfyUI nodes module to access NODE_CLASS_MAPPINGS
            import nodes
            
            # Store old node names for cleanup (but keep them for now to avoid breaking workflows)
            old_donut_nodes = [name for name in nodes.NODE_CLASS_MAPPINGS.keys() 
                             if any(x in name for x in ['Donut'])]
            
            print(f"[DonutHotReload] Found {len(old_donut_nodes)} existing Donut nodes")
            
            # Reload each module individually (skip __init__ for now)
            reloaded_modules = {}
            for module_name in self.node_modules:
                full_module_name = f"custom_nodes.ComfyUI-DonutNodes.{module_name}"
                if full_module_name in sys.modules:
                    print(f"[DonutHotReload] Reloading module: {module_name}")
                    reloaded_modules[module_name] = importlib.reload(sys.modules[full_module_name])
            
            # Now reload the main __init__ module
            init_module_name = "custom_nodes.ComfyUI-DonutNodes"
            if init_module_name in sys.modules:
                print("[DonutHotReload] Reloading main __init__ module...")
                reloaded_init = importlib.reload(sys.modules[init_module_name])
                
                # Skip removing old nodes to avoid interfering with other extensions
                # for node_name in old_donut_nodes:
                #     if node_name in nodes.NODE_CLASS_MAPPINGS:
                #         del nodes.NODE_CLASS_MAPPINGS[node_name]
                #         print(f"[DonutHotReload] Removed old node: {node_name}")
                
                # Re-register nodes from the reloaded module
                if hasattr(reloaded_init, 'NODE_CLASS_MAPPINGS'):
                    new_nodes = reloaded_init.NODE_CLASS_MAPPINGS
                    nodes.NODE_CLASS_MAPPINGS.update(new_nodes)
                    print(f"[DonutHotReload] Re-registered {len(new_nodes)} nodes: {list(new_nodes.keys())}")
                
                if hasattr(reloaded_init, 'NODE_DISPLAY_NAME_MAPPINGS'):
                    if not hasattr(nodes, 'NODE_DISPLAY_NAME_MAPPINGS'):
                        nodes.NODE_DISPLAY_NAME_MAPPINGS = {}
                    nodes.NODE_DISPLAY_NAME_MAPPINGS.update(reloaded_init.NODE_DISPLAY_NAME_MAPPINGS)
            
            # Update timestamps
            self._update_timestamps()
            
            print("[DonutHotReload] ✓ Hot reload completed successfully!")
            return True
            
        except Exception as e:
            print(f"[DonutHotReload] ✗ Reload failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        """
    
    def _watch_loop(self):
        """Background thread that watches for file changes"""
        print("[DonutHotReload] File watching started")
        while self.watching:
            try:
                if self._has_file_changed():
                    time.sleep(0.5)  # Brief delay to ensure file write is complete
                    self.reload_modules()
                time.sleep(1)  # Check every second
            except Exception as e:
                print(f"[DonutHotReload] Watch error: {e}")
                time.sleep(5)  # Wait longer on error
    
    def start_watching(self):
        """Start watching for file changes - DISABLED to prevent interference with other extensions"""
        print("[DonutHotReload] Hot reload watching disabled to prevent interference with other extensions")
        return False
    
    def stop_watching(self):
        """Stop watching for file changes"""
        if self.watching:
            self.watching = False
            if self.watch_thread:
                self.watch_thread.join(timeout=2)
            print("[DonutHotReload] Hot reload watching disabled")

# Global instance
hot_reload = DonutHotReload()

# Stop any existing watching to prevent interference
try:
    hot_reload.stop_watching()
    print("[DonutHotReload] Stopped any existing file watching on import")
except:
    pass

class DonutHotReloadNode:
    """ComfyUI node for controlling hot reload functionality"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["start_watching", "stop_watching", "reload_now"], {"default": "start_watching"}),
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
            success = hot_reload.reload_modules()
            status = "Reload successful" if success else "Reload failed"
        else:
            status = "Unknown action"
        
        print(f"[DonutHotReload] {status}")
        return (status,)

# Export the node
NODE_CLASS_MAPPINGS = {
    "DonutHotReload": DonutHotReloadNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutHotReload": "DonutHotReload",
}
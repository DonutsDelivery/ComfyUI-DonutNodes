"""Cancelling a stage must not leave ComfyUI a dead loaded-model entry.

A dead entry makes every later model load run a full gc.collect() until restart.
"""
import gc
from pathlib import Path
import sys
import unittest
import weakref

import torch

# Append, so ComfyUI's own `utils` package wins over this repo's.
sys.path.append(str(Path(__file__).resolve().parents[1]))

import comfy.model_management as mm
import comfy.model_patcher

from krea2_nag_integration import anchor_stage_patcher


def _stage_entry(anchor):
    """Loader patcher -> temporary clone -> stage clone, loaded, then freed by
    the cycle collector as an interrupt traceback does."""
    module = torch.nn.Linear(2, 2)
    root = comfy.model_patcher.ModelPatcher(module, torch.device("cpu"), torch.device("cpu"))
    temporary = root.clone()
    stage = temporary.clone()
    if anchor:
        anchor_stage_patcher(stage)
    entry = mm.LoadedModel(stage)
    entry.real_model = weakref.ref(module)
    cycle = [temporary, stage]
    cycle.append(cycle)
    del temporary, stage, cycle
    gc.collect()
    return root, entry


class StagePatcherAnchorTests(unittest.TestCase):
    def test_unanchored_stage_leaves_a_dead_entry(self):
        _root, entry = _stage_entry(anchor=False)
        self.assertTrue(entry.is_dead())

    def test_anchored_stage_falls_back_to_the_live_root(self):
        root, entry = _stage_entry(anchor=True)
        self.assertFalse(entry.is_dead())
        self.assertIs(entry.model, root)

    def test_root_and_unpatched_models_are_returned_unchanged(self):
        root = comfy.model_patcher.ModelPatcher(torch.nn.Linear(2, 2), torch.device("cpu"), torch.device("cpu"))
        self.assertIs(anchor_stage_patcher(root), root)
        self.assertIsNone(root.parent)
        sentinel = object()
        self.assertIs(anchor_stage_patcher(sentinel), sentinel)


if __name__ == "__main__":
    unittest.main()

"""Vendored RL inference code (unmodified copies; see VENDOR.md).

The files under sim2real/ and model/ are verbatim copies from the team's RL
repository (Pequi-Mecanico-SSL/RL) and use their original top-level import paths (``from sim2real.utils
import ...``). This shim puts this directory on sys.path so those imports
resolve without editing the vendored files.

Nothing outside rl_strategy.adapter may import from this package.
"""
import os
import sys

_VENDORED_DIR = os.path.dirname(os.path.abspath(__file__))
if _VENDORED_DIR not in sys.path:
    sys.path.insert(0, _VENDORED_DIR)

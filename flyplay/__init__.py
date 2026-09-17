"""Helpers for driving NeuroMechFly v2 (FlyGym 2.x) on this machine.

Modules
-------
build          : construct fly + world + Simulation for a named terrain
control        : decimated hybrid walking controller + wall-clock pacing
env            : Gymnasium environment for goal-directed navigation (RL)
odor           : olfaction, which FlyGym 2.x does not ship -- ported here
vision         : analytic bilateral features from the 1442 ommatidia
arena          : pillars the fly can see and hit, markers it cannot
env_multimodal : find food by smell while dodging obstacles by sight
"""

import os

# Render on the discrete NVIDIA GPU of a hybrid-graphics Windows laptop, for this
# process and the ones it starts only (FlyGym's eyes and the viewer's video are
# OpenGL through MuJoCo). The NVIDIA Optimus driver reads SHIM_MCCOMPAT when an
# OpenGL context is created, so this must run before the first renderer: it does
# (measured working even when set after mujoco and flygym were imported). Without
# it this laptop drew on "Intel(R) Graphics": a 512x450 eye render and read-back
# 10.9-16.5 ms, against 0.9-1.9 ms on the RTX 5060. The GPU changes pixels at
# colour edges: over 114 fixed views, 7.5% of eye views got a visual Kenyon-cell
# code differing in 1 of its 5 active cells, so results are not bit-identical
# to Intel-rendered ones. Set SHIM_MCCOMPAT=0x800000000 to render on the
# integrated GPU; other drivers and platforms ignore the variable.
os.environ.setdefault("SHIM_MCCOMPAT", "0x800000001")

from flyplay.build import TERRAINS, FlySim, build  # noqa: E402

__all__ = ["TERRAINS", "FlySim", "build"]

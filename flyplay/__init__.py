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

from flyplay.build import TERRAINS, FlySim, build

__all__ = ["TERRAINS", "FlySim", "build"]

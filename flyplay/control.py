"""Walking control at a sane rate, plus wall-clock pacing for live viewing.

NeuroMechFly's physics runs at 10 kHz (dt = 1e-4 s). Re-evaluating the hybrid
controller at every one of those steps is what makes the simulation feel slow:
measured on this machine, raw physics runs at ~1.85x real time, but calling
`HybridControllerObservation.from_sim` every step drops it to ~0.17x. The
controller only needs a few hundred Hz to reproduce the same gait, so
`Walker` decimates it. At `decimation=20` (500 Hz control) the fly walks an
identical distance and the loop runs at ~1.1x real time.
"""

from __future__ import annotations

import time

import numpy as np
from flygym_demo.complex_terrain import (
    HybridControllerObservation,
    HybridTurningController,
    LocomotionAction,
    PreprogrammedSteps,
    apply_locomotion_action,
)

from flyplay.build import FlySim

#: Physics steps per controller update. 20 -> 500 Hz control at dt = 1e-4 s.
DEFAULT_DECIMATION = 20


class Walker:
    """Hybrid CPG turning controller driven at a decimated rate.

    The two-element ``descending_signal`` is the high-level command: it scales
    the CPG amplitude of the left and right tripods, so ``[1.0, 1.0]`` walks
    straight, ``[1.2, 0.4]`` turns right, and ``[0.4, 1.2]`` turns left. This
    is the same interface the brain-to-VNC descending neurons provide in the
    NeuroMechFly papers, and the action space the RL policy learns to drive.

    Args:
        flysim: The simulation to control.
        decimation: Physics steps between controller updates.
        seed: Seed for the CPG's initial phases.
    """

    def __init__(
        self,
        flysim: FlySim,
        *,
        decimation: int = DEFAULT_DECIMATION,
        seed: int = 0,
        profile: bool = False,
    ):
        self.fs = flysim
        self.decimation = int(decimation)
        self.seed = seed
        # Simulation.print_performance_report() only has data if the profiled
        # stepping variants were used, and raises otherwise.
        self.profile = profile
        self.steps = PreprogrammedSteps()
        self.controller = HybridTurningController(
            timestep=flysim.sim.timestep * self.decimation,
            preprogrammed_steps=self.steps,
            output_dof_order=flysim.dof_order,
        )
        self._phys_counter = 0
        self.descending_signal = np.array([1.0, 1.0], dtype=float)

    # --- lifecycle ------------------------------------------------------

    def reset(self, *, seed: int | None = None, warmup_s: float = 0.05) -> None:
        """Reset physics and controller, then let the fly settle onto the ground."""
        self.fs.sim.reset()
        self.controller.reset(seed=self.seed if seed is None else seed)
        self._phys_counter = 0
        self.descending_signal = np.array([1.0, 1.0], dtype=float)
        apply_locomotion_action(
            self.fs.sim,
            self.fs.name,
            LocomotionAction(
                joint_angles=self.steps.default_pose_by_dof_order(self.fs.dof_order),
                adhesion_onoff=np.ones(6, dtype=bool),
            ),
        )
        self.fs.sim.warmup(warmup_s)

    def physics_step(self) -> None:
        """Advance physics by one step, refreshing the controller when due."""
        if self._phys_counter % self.decimation == 0:
            obs = HybridControllerObservation.from_sim(self.fs.sim, self.fs.name)
            action = self.controller.step(self.descending_signal, obs)
            apply_locomotion_action(self.fs.sim, self.fs.name, action)
        if self.profile:
            self.fs.sim.step_with_profile()
        else:
            self.fs.sim.step()
        self._phys_counter += 1

    def run_for(self, duration_s: float) -> int:
        """Step physics for `duration_s` of simulated time. Returns steps taken."""
        n = int(round(duration_s / self.fs.sim.timestep))
        for _ in range(n):
            self.physics_step()
        return n

    # --- CPG state, useful as an RL observation -------------------------

    @property
    def cpg_phases(self) -> np.ndarray:
        return self.controller.cpg_network.curr_phases

    @property
    def cpg_magnitudes(self) -> np.ndarray:
        return self.controller.cpg_network.curr_magnitudes


class RealtimePacer:
    """Throttle a simulation loop so it plays back at a chosen wall-clock speed.

    Args:
        timestep: Physics timestep in seconds.
        speed: Target playback speed. 1.0 tracks real time; 0.2 is slow motion,
            which is what you want to actually see the legs.
    """

    def __init__(self, timestep: float, speed: float = 1.0):
        self.timestep = timestep
        self.speed = max(speed, 1e-6)
        self.reset()

    def reset(self) -> None:
        self._wall_start = time.perf_counter()
        self._sim_start: float | None = None
        self._lag_s = 0.0

    def wait(self, sim_time: float) -> None:
        """Sleep until wall-clock time catches up with `sim_time`."""
        if self._sim_start is None:
            self._sim_start = sim_time
        target = (sim_time - self._sim_start) / self.speed
        actual = time.perf_counter() - self._wall_start
        remaining = target - actual
        if remaining > 0:
            time.sleep(remaining)
            self._lag_s = 0.0
        else:
            self._lag_s = -remaining

    @property
    def lag_s(self) -> float:
        """How far the last `wait` call was behind schedule, in seconds."""
        return self._lag_s

    def report(self, sim_time: float) -> str:
        wall = max(time.perf_counter() - self._wall_start, 1e-9)
        elapsed_sim = sim_time - (self._sim_start or sim_time)
        return (
            f"sim {elapsed_sim:6.2f}s | wall {wall:6.2f}s | "
            f"speed {elapsed_sim / wall:5.2f}x (target {self.speed:.2f}x)"
        )

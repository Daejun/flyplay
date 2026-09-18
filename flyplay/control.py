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

import mujoco
import numpy as np
from flygym.anatomy import LEGS
from flygym.compose import ActuatorType
from scipy.interpolate import PPoly
from flygym_demo.complex_terrain import (
    HybridControllerObservation,
    HybridTurningController,
    LocomotionAction,
    PreprogrammedSteps,
    apply_locomotion_action,
)
from flygym_demo.complex_terrain.common import dof_spec_to_jointdof, get_default_locomotion_dof_order
from flygym_demo.complex_terrain.hybrid_controller import (
    _CORRECTION_VECTORS,
    _DETECTED_STUMBLING_LINKS,
    _RIGHT_LEG_CORRECTION_SIGN,
)

from flyplay.build import ContactForces, FlySim

#: Physics steps per controller update. 20 -> 500 Hz control at dt = 1e-4 s.
DEFAULT_DECIMATION = 20

TWO_PI = 2 * np.pi


class FastTurningController(HybridTurningController):
    """FlyGym's hybrid turning controller with its per-call bookkeeping done once.

    Same arithmetic in the same order, so joint targets and adhesion flags are
    bit-identical to `HybridTurningController.step`. Measured on a sandbox fly
    under cProfile, the stock step took 366 ms of each simulated second at
    500 Hz, most of it building and hashing 42 `JointDOF` objects per call to
    order its output and rebuilding the phase-gain breakpoints. `last_info`,
    which nothing in flyplay reads, is left empty.

    The per-leg numpy calls then dominated what was left, so `step` evaluates
    all six leg splines in one `PPoly` call and does the rest of the per-leg
    work in Python floats: 88.7 us a call against 51.3, or 79.1 ms of each
    simulated second against 54.2. Checked over 5000 observations recorded
    from a real run (13 stumbling corrections among them): joint angles,
    adhesion flags, both correction arrays and the CPG phases matched exactly
    on every call.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        steps = self.preprogrammed_steps
        order = self.output_dof_order or get_default_locomotion_dof_order()
        position = {dof: i for i, dof in enumerate(order)}
        self._out_index = [np.array([position[dof_spec_to_jointdof(leg, spec)] for spec in steps.dofs_per_leg])
                           for leg in self.legs]
        self._n_out = len(order)
        self._psi = [steps._psi_funcs[leg] for leg in self.legs]
        self._neutral = [steps.neutral_pos[leg] for leg in self.legs]
        self._correction = [_CORRECTION_VECTORS[leg[1]] * _RIGHT_LEG_CORRECTION_SIGN if leg.startswith("r")
                            else _CORRECTION_VECTORS[leg[1]] for leg in self.legs]
        self._gain_points, self._adhesion_window = [], []
        for leg in self.legs:
            swing_start, swing_end = steps.swing_period[leg]
            # As `_step_phase_gain` and `_get_adhesion_onoff` compute them per call.
            self._gain_points.append(np.array([swing_start, np.mean([swing_start, swing_end]),
                                               swing_end + self.swing_extension,
                                               np.mean([swing_end, 2 * np.pi]), 2 * np.pi]))
            self._adhesion_window.append((swing_start, swing_end + self.swing_extension))
        self._gain_values = np.array([0.0, 0.8, 0.0, -0.1, 0.0])
        # One spline for all six legs. The per-leg coefficients are stacked on
        # a new last axis, so evaluating at the six phases gives (6, 7, 6) and
        # leg i reads [i, :, i] -- each element the same sum of the same terms.
        # The legs' splines are all `CubicSpline(..., bc_type="periodic")` on
        # the same knots, which is where `extrapolate="periodic"` comes from.
        first = self._psi[0]
        if not all(np.array_equal(p.x, first.x) for p in self._psi):
            raise ValueError("leg splines no longer share their knots; see __post_init__")
        self._psi_all = PPoly(np.stack([p.c for p in self._psi], axis=-1), first.x, extrapolate="periodic")
        self._neutral_all = np.stack([n[:, 0] for n in self._neutral])  # (6, 7)
        self._leg_index = np.arange(len(self.legs))
        # Python floats for the scalar interpolation below, which is `np.interp`
        # written out: the call itself cost 0.71 us of the 88.7.
        self._gain_x = [tuple(float(v) for v in points) for points in self._gain_points]
        self._gain_y = tuple(float(v) for v in self._gain_values)

    def step(self, descending_signal: np.ndarray, obs: HybridControllerObservation) -> LocomotionAction:
        descending_signal = np.asarray(descending_signal, dtype=float)
        if descending_signal.shape != (2,):
            raise ValueError("descending_signal must have shape (2,).")
        cpg = self.cpg_network
        cpg.intrinsic_amps = np.repeat(np.abs(descending_signal[:, np.newaxis]), 3, axis=1).ravel()
        intrinsic_freqs = self._base_intrinsic_freqs.copy()
        intrinsic_freqs[:3] *= 1 if descending_signal[0] >= 0 else -1
        intrinsic_freqs[3:] *= 1 if descending_signal[1] >= 0 else -1
        cpg.intrinsic_freqs = intrinsic_freqs

        leg_to_correct = self._select_retraction_leg(obs)
        if leg_to_correct is not None:
            if self.retraction_correction[leg_to_correct] > self.retraction_persistence_initiation_threshold:
                self.retraction_persistence_counter[leg_to_correct] = 1
        self._update_persistence_counter()
        stumbling_mask = self._get_stumbling_mask(obs)
        cpg.step()

        phases, magnitudes = cpg.curr_phases, cpg.curr_magnitudes
        own = self._psi_all(phases)[self._leg_index, :, self._leg_index]  # (6, 7)
        neutral = self._neutral_all
        leg_angles_all = neutral + magnitudes[:, np.newaxis] * (own - neutral)

        retraction, stumbling = self.retraction_correction, self.stumbling_correction
        persistence = self.retraction_persistence_counter
        # `_update_retraction_correction` and `_update_stumbling_correction`
        # written out, with their per-call products hoisted.
        retraction_up = self.retraction_rates[0] * self.timestep
        retraction_down = self.retraction_rates[1] * self.timestep
        stumbling_up = self.stumbling_rates[0] * self.timestep
        stumbling_down = self.stumbling_rates[1] * self.timestep
        joint_angles = np.empty(self._n_out, dtype=float)
        adhesion = np.zeros(len(self.legs), dtype=bool)
        for leg_idx in range(len(self.legs)):
            if leg_idx == leg_to_correct or persistence[leg_idx] > 0:
                retraction[leg_idx] += retraction_up
            else:
                retraction[leg_idx] = max(0, retraction[leg_idx] - retraction_down)
            if stumbling_mask[leg_idx]:
                stumbling[leg_idx] += stumbling_up
            else:
                stumbling[leg_idx] = max(0, stumbling[leg_idx] - stumbling_down)
            if retraction[leg_idx] > 0:
                net_correction = retraction[leg_idx]
                stumbling[leg_idx] = 0
            else:
                net_correction = stumbling[leg_idx]
            net_correction = np.clip(net_correction, 0, self.max_correction)
            phase = phases[leg_idx] % TWO_PI
            phase_gain = self._phase_gain(leg_idx, phase)
            joint_angles[self._out_index[leg_idx]] = (
                leg_angles_all[leg_idx] + net_correction * phase_gain * self._correction[leg_idx])
            if self.enable_adhesion:
                swing_start, swing_end = self._adhesion_window[leg_idx]
                adhesion[leg_idx] = not (swing_start < phase < swing_end)
        return LocomotionAction(joint_angles=joint_angles, adhesion_onoff=adhesion)

    def _phase_gain(self, leg_idx: int, phase: float) -> float:
        """`np.interp(phase, self._gain_points[leg_idx], self._gain_values)`,
        term for term, on Python floats: the same five breakpoints, the same
        slope expression, the same behaviour off either end and on a knot."""
        points, values = self._gain_x[leg_idx], self._gain_y
        if phase <= points[0]:
            return values[0]
        if phase >= points[-1]:
            return values[-1]
        i = 0
        while not (points[i] <= phase < points[i + 1]):
            i += 1
        if phase == points[i]:
            return values[i]
        slope = (values[i + 1] - values[i]) / (points[i + 1] - points[i])
        return slope * (phase - points[i]) + values[i]


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
        self.controller = FastTurningController(
            timestep=flysim.sim.timestep * self.decimation,
            preprogrammed_steps=self.steps,
            output_dof_order=flysim.dof_order,
        )
        self._phys_counter = 0
        self.descending_signal = np.array([1.0, 1.0], dtype=float)
        # What `HybridControllerObservation.from_sim` and `apply_locomotion_action`
        # look up on every call, looked up once.
        sim, name = flysim.sim, flysim.name
        fly = sim.world.fly_lookup[name]
        segment = type(fly).BODY_SEGMENT_CLASS
        order = fly.get_bodysegs_order()
        body_ids = sim._internal_bodyids_by_fly[name]
        self._thorax_body = int(body_ids[order.index(segment("c_thorax"))])
        self._tarsus_bodies = [int(body_ids[order.index(segment(f"{leg}_tarsus5"))]) for leg in LEGS]
        self._stumbling_forces = ContactForces(
            sim, name, [segment(f"{leg}_{link}") for leg in LEGS for link in _DETECTED_STUMBLING_LINKS])
        self._position_ids = sim._intern_actuatorids_by_type_by_fly[ActuatorType.POSITION][name]
        self._adhesion_ids = sim._intern_adhesionactuatorids_by_fly[name]

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

    def _observe(self) -> HybridControllerObservation:
        """`HybridControllerObservation.from_sim`, value for value."""
        data = self.fs.sim.mj_data
        xpos = data.xpos
        return HybridControllerObservation(
            thorax_z=float(xpos[self._thorax_body, 2]),
            tarsus5_z=np.array([xpos[i, 2] for i in self._tarsus_bodies], dtype=float),
            stumbling_contact_forces=self._stumbling_forces().reshape(len(LEGS), len(_DETECTED_STUMBLING_LINKS), 3),
            fly_heading=data.xmat[self._thorax_body].reshape(3, 3)[:, 0].copy(),
        )

    def _control(self) -> None:
        action = self.controller.step(self.descending_signal, self._observe())
        ctrl = self.fs.sim.mj_data.ctrl
        # What `apply_locomotion_action` writes through `Simulation.set_*`.
        ctrl[self._position_ids] = action.joint_angles
        ctrl[self._adhesion_ids] = action.adhesion_onoff

    def physics_step(self) -> None:
        """Advance physics by one step, refreshing the controller when due."""
        if self._phys_counter % self.decimation == 0:
            self._control()
        if self.profile:
            self.fs.sim.step_with_profile()
        else:
            self.fs.sim.step()
        self._phys_counter += 1

    def advance(self, steps: int) -> None:
        """`physics_step` `steps` times. Between controller updates the physics
        runs as one `mj_step(..., nstep)` call, which loops in C: the same steps
        in the same order, without a Python call per step."""
        sim = self.fs.sim
        remaining = int(steps)
        while remaining > 0:
            if self._phys_counter % self.decimation == 0:
                self._control()
            block = min(remaining, self.decimation - self._phys_counter % self.decimation)
            if self.profile:
                for _ in range(block):
                    sim.step_with_profile()
            else:
                mujoco.mj_step(sim.mj_model, sim.mj_data, nstep=block)
            self._phys_counter += block
            remaining -= block

    def run_for(self, duration_s: float) -> int:
        """Step physics for `duration_s` of simulated time. Returns steps taken."""
        n = int(round(duration_s / self.fs.sim.timestep))
        self.advance(n)
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

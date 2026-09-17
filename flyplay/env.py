"""Gymnasium environment: goal-directed navigation for NeuroMechFly v2.

What the fly actually learns here
---------------------------------
The policy does **not** learn to move individual legs. Walking itself comes
from the hybrid CPG controller, which is a fixed, biologically-motivated
spinal-cord model. What the policy learns is the *descending command*: the
two-element signal a real fly's brain sends down to its ventral nerve cord to
modulate the left and right tripods.

So the learned quantity is a steering law:

    (where the goal is, how I am moving, what my legs are doing)
        -> (left tripod drive, right tripod drive)

That is a small, fast-converging problem, and it is directly interpretable --
`scripts/07_analyze_policy.py` sweeps the goal bearing and plots the resulting
turn command, which is the learned control law drawn as a curve.

Because the same policy interface works on every terrain in
`flyplay.build.TERRAINS`, training the same task on "flat" vs "gapped" vs
"blocks" shows how the learned law changes when the ground stops cooperating.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from flyplay.build import TERRAINS, build
from flyplay.control import DEFAULT_DECIMATION, Walker

# Descending-signal range, measured on this machine (see README "Steering
# response"). Over [0.4, 1.6] both speed and turn rate respond monotonically:
# symmetric drive gives 5.6 mm/s at 0.4 and 23.8 mm/s at 1.6, and the turn rate
# runs smoothly from -166 deg/s (left tripod stronger) to +150 deg/s. Pushing a
# tripod below ~0.3 stalls it, the fly starts pivoting about the stalled side,
# and the turn direction flips -- so the action space stops short of that.
SIGNAL_LOW = 0.4
SIGNAL_HIGH = 1.6


@dataclass
class NavConfig:
    """Task and reward settings for `FlyNavEnv`.

    Attributes:
        terrain: Key into `flyplay.build.TERRAINS`.
        goal_distance: (min, max) spawn distance of the goal, in mm.
        goal_radius: Distance at which the goal counts as reached, in mm.
        episode_seconds: Time limit for one episode, in simulated seconds.
        action_hz: How often the policy emits a descending command.
        decimation: Physics steps per hybrid-controller update.
        w_progress: Reward per mm of distance closed on the goal.
        w_heading: Reward per second for facing the goal.
        w_smooth: Penalty on the squared change in action between steps.
        r_goal: One-off bonus for reaching the goal.
        r_fall: One-off penalty for falling over or leaving the terrain.
        fall_upright: Terminate when the thorax up-axis z drops below this.
            0.2 is roughly 78 degrees of roll -- past recovering.
        fall_height: Terminate when the thorax sits this far, in mm, below the
            settled reference height. Must clear the gait's own bobbing, which
            is about +-0.2 mm on flat ground; 0.8 mm means it genuinely dropped
            off something (the gaps are 2 mm deep).
        warmup_seconds: Settling time before the episode clock starts. The fly
            spawns in the air and needs ~0.2 s to stop bouncing; measuring the
            reference height any earlier catches the transient and every
            episode terminates immediately as a false fall.
        settle_seconds: Window used to measure the reference ground height.
    """

    terrain: str = "flat"
    goal_distance: tuple[float, float] = (10.0, 18.0)
    goal_radius: float = 2.0
    episode_seconds: float = 5.0
    action_hz: float = 100.0
    decimation: int = DEFAULT_DECIMATION
    w_progress: float = 8.0
    w_heading: float = 1.0
    w_smooth: float = 0.05
    r_goal: float = 50.0
    r_fall: float = -20.0
    fall_upright: float = 0.2
    fall_height: float = 0.8
    warmup_seconds: float = 0.2
    settle_seconds: float = 0.1
    #: Filled in from the terrain spec unless overridden.
    goal_bearing_deg: float | None = None
    #: Tracking cameras to attach. Empty for training -- cameras cost nothing
    #: until something renders, but the web viewer needs them on the model
    #: before it is compiled.
    cameras: tuple[str, ...] = ()


OBS_LAYOUT: tuple[tuple[str, int], ...] = (
    ("goal_cos_sin", 2),  # goal bearing in the fly's own frame
    ("goal_distance", 1),  # tanh-squashed distance to goal
    ("velocity", 2),  # forward / lateral speed, body frame
    ("yaw_rate", 1),
    ("height", 1),  # thorax height above spawn ground
    ("upright", 1),  # z of the thorax up-axis
    ("cpg_phase", 12),  # sin/cos of the six CPG phases
    ("cpg_magnitude", 6),
    ("leg_contact", 6),
    ("prev_action", 2),
)
OBS_DIM = sum(n for _, n in OBS_LAYOUT)


class FlyNavEnv(gym.Env):
    """Steer NeuroMechFly to a goal point on a chosen terrain.

    **Action** — ``Box(-1, 1, shape=(2,))``, linearly mapped onto the left and
    right descending signals in ``[SIGNAL_LOW, SIGNAL_HIGH]``.

    The fly always spawns at the origin facing +x; it is the *goal* that is
    placed at a random bearing. The world is compiled once per env, so the
    spawn pose is fixed, and for a steering task only the relative bearing
    matters anyway.

    **Observation** — ``Box(shape=(34,))``; see `OBS_LAYOUT` for the field
    order. All of it is information a real fly plausibly has: an egocentric
    goal direction, self-motion, posture, its own CPG state, and leg load.

    **Reward** — progress toward the goal dominates; a heading term breaks the
    symmetry early in training, a smoothness term discourages twitching, and
    reaching the goal or falling over ends the episode with a bonus/penalty.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: NavConfig | None = None, *, seed: int | None = None):
        super().__init__()
        self.config = config or NavConfig()
        if self.config.terrain not in TERRAINS:
            raise ValueError(
                f"Unknown terrain {self.config.terrain!r}. "
                f"Choose from {sorted(TERRAINS)}."
            )
        self._build_seed = 0 if seed is None else int(seed)

        self.fs = build(
            self.config.terrain,
            cameras=self.config.cameras,
            seed=self._build_seed,
        )
        self.walker = Walker(self.fs, decimation=self.config.decimation)

        physics_dt = self.fs.sim.timestep
        steps_per_action = (1.0 / self.config.action_hz) / physics_dt
        if abs(steps_per_action - round(steps_per_action)) > 1e-6:
            raise ValueError(
                f"action_hz={self.config.action_hz} does not divide the "
                f"physics rate {1 / physics_dt:.0f} Hz evenly."
            )
        self.steps_per_action = int(round(steps_per_action))
        self.max_steps = int(round(self.config.episode_seconds * self.config.action_hz))

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        bearing = self.config.goal_bearing_deg
        self._goal_bearing_rad = np.deg2rad(
            self.fs.spec.goal_bearing_deg if bearing is None else bearing
        )

        self.goal = np.zeros(2, dtype=float)
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._prev_distance = 0.0
        self._step_count = 0
        self._ground_z = 0.0
        self._path: list[np.ndarray] = []

    # --- gymnasium API --------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        rng = self.np_random
        opts = options or {}

        self.walker.reset(
            seed=int(rng.integers(0, 2**31 - 1)),
            warmup_s=self.config.warmup_seconds,
        )

        # Reference ground height, taken as the median over a settled window.
        # A single sample lands wherever the gait happens to be in its cycle,
        # which is up to 0.2 mm off and pushes every later height comparison
        # around by that much.
        samples = []
        for _ in range(10):
            self.walker.run_for(self.config.settle_seconds / 10)
            samples.append(self.fs.thorax_pos()[2])
        self._ground_z = float(np.median(samples))

        start = self.fs.thorax_pos()

        if "goal" in opts:
            self.goal = np.asarray(opts["goal"], dtype=float)[:2]
        else:
            bearing = opts.get("bearing")
            if bearing is None:
                bearing = float(
                    rng.uniform(-self._goal_bearing_rad, self._goal_bearing_rad)
                )
            distance = opts.get(
                "distance", float(rng.uniform(*self.config.goal_distance))
            )
            yaw = self.fs.yaw() + bearing
            self.goal = start[:2] + distance * np.array([np.cos(yaw), np.sin(yaw)])

        self._prev_action[:] = 0.0
        self._prev_distance = float(np.linalg.norm(self.goal - start[:2]))
        self._step_count = 0
        self._path = [start[:2].copy()]

        return self._observe(), {"goal": self.goal.copy(), "start": start.copy()}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.walker.descending_signal = self.action_to_signal(action)

        for _ in range(self.steps_per_action):
            self.walker.physics_step()

        self._step_count += 1
        pos = self.fs.thorax_pos()
        self._path.append(pos[:2].copy())

        distance = float(np.linalg.norm(self.goal - pos[:2]))
        dt = 1.0 / self.config.action_hz

        progress = self._prev_distance - distance
        reward = self.config.w_progress * progress
        reward += self.config.w_heading * np.cos(self.goal_bearing()) * dt
        reward -= self.config.w_smooth * float(
            np.sum((action - self._prev_action) ** 2)
        )

        terminated = False
        outcome = "running"

        if distance <= self.config.goal_radius:
            reward += self.config.r_goal
            terminated = True
            outcome = "goal"
        elif self._has_fallen(pos):
            reward += self.config.r_fall
            terminated = True
            outcome = "fell"
        elif self.fs.out_of_bounds():
            reward += self.config.r_fall
            terminated = True
            outcome = "out_of_bounds"

        truncated = (not terminated) and self._step_count >= self.max_steps
        if truncated:
            outcome = "timeout"

        self._prev_distance = distance
        self._prev_action = action.copy()

        info = {
            "distance": distance,
            "progress": progress,
            "bearing": self.goal_bearing(),
            "outcome": outcome,
            "sim_time": self.fs.sim.time,
        }
        if terminated or truncated:
            info["is_success"] = outcome == "goal"
            info["path"] = np.asarray(self._path)
            info["goal"] = self.goal.copy()

        return self._observe(), float(reward), terminated, truncated, info

    def close(self) -> None:
        self.fs.close()

    # --- helpers --------------------------------------------------------

    @staticmethod
    def action_to_signal(action: np.ndarray) -> np.ndarray:
        """Map an action in [-1, 1]^2 onto left/right descending signals."""
        mid = 0.5 * (SIGNAL_HIGH + SIGNAL_LOW)
        half = 0.5 * (SIGNAL_HIGH - SIGNAL_LOW)
        return mid + half * np.asarray(action, dtype=float)

    def goal_bearing(self) -> float:
        """Angle to the goal in the fly's own frame; 0 ahead, + to the left.

        Wrapped to (-pi, pi]. The observation and reward only use the cosine
        and sine so they do not care, but an unwrapped value plots as a jump
        from -180 to +180 mid-turn.
        """
        delta = self.goal - self.fs.thorax_pos()[:2]
        bearing = np.arctan2(delta[1], delta[0]) - self.fs.yaw()
        return float(np.arctan2(np.sin(bearing), np.cos(bearing)))

    @property
    def step_count(self) -> int:
        """Policy steps taken in the current episode."""
        return self._step_count

    @property
    def elapsed(self) -> float:
        """Episode time so far, in simulated seconds."""
        return self._step_count / self.config.action_hz

    @property
    def max_goal_bearing(self) -> float:
        """Widest goal bearing this terrain samples, in radians."""
        return self._goal_bearing_rad

    def _has_fallen(self, pos: np.ndarray) -> bool:
        if self.fs.upright() < self.config.fall_upright:
            return True
        return bool(pos[2] - self._ground_z < -self.config.fall_height)

    def _observe(self) -> np.ndarray:
        bearing = self.goal_bearing()
        distance = float(np.linalg.norm(self.goal - self.fs.thorax_pos()[:2]))
        vel = self.fs.body_velocity()
        phases = self.walker.cpg_phases

        obs = np.concatenate(
            [
                [np.cos(bearing), np.sin(bearing)],
                [np.tanh(distance / 10.0)],
                vel[:2] / 20.0,
                [self.fs.yaw_rate() / 10.0],
                [(self.fs.thorax_pos()[2] - self._ground_z) / 2.0],
                [self.fs.upright()],
                np.sin(phases),
                np.cos(phases),
                self.walker.cpg_magnitudes,
                self.fs.leg_contacts().astype(float),
                self._prev_action,
            ]
        )
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def make_env(terrain: str = "flat", seed: int = 0, **overrides: Any) -> FlyNavEnv:
    """Convenience constructor used by the training and analysis scripts."""
    config = NavConfig(terrain=terrain, **overrides)
    env = FlyNavEnv(config, seed=seed)
    env.reset(seed=seed)
    return env

"""Find food by smell while dodging obstacles by sight.

This is the task from Figure 5 of the NeuroMechFly v2 paper, rebuilt on the
2.x API. What makes it a *multimodal* task rather than two tasks bolted
together is that neither sense can do the other's job:

- **Odour says where, not how.** The intensity field is radial, so it points
  straight at the food and knows nothing about what is in the way.
- **Vision says what is in the way, not where to go.** The pillars are dark and
  obvious; the food has no visual marker at all (its marker geom sits in group
  2, which the eye cameras ignore -- verified, the fly's visual features do not
  move at all when a source is placed in front of it).

So the policy has to hold a heading from one sense while detouring on the
other. Unlike `flyplay.env.FlyNavEnv`, the observation contains **no goal
bearing and no goal distance**: everything the fly knows about the food comes
through its four odour sensors.

Reward still uses true distance-to-source, which is privileged information. That
is deliberate and standard -- it is a dense training signal, not an input to the
policy. Only the *observation* has to be biologically honest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import mujoco as mj
import numpy as np
from gymnasium import spaces

from flyplay.arena import move_geom
from flyplay.build import TERRAINS, build
from flyplay.control import DEFAULT_DECIMATION, Walker
from flyplay.env import SIGNAL_HIGH, SIGNAL_LOW
from flyplay.odor import OdorField, OdorSource
from flyplay.vision import N_FEATURES as N_VISION_FEATURES
from flyplay.vision import RetinaFeatures

#: Odour asymmetry between sensors 0.3 mm apart, with the source 8-25 mm away,
#: lands in about +/-0.04 (measured). Scale it into the same range as the rest
#: of the observation or the policy will never notice it.
ASYMMETRY_GAIN = 20.0
#: Visual mass features run 0 to ~0.2 for a nearby pillar (measured).
VISION_GAIN = 5.0
#: Scale for the log-intensity time derivative. Walking at 14 mm/s straight at
#: the source gives d(ln I)/dt = 2 v / d, which is 1.3 /s at 22 mm and 7 /s at
#: 4 mm; dividing by 4 before the squash keeps that whole range resolvable.
#: (Using the raw derivative instead saturates by 8 mm and throws away exactly
#: the part of the approach where steering still matters.)
ODOR_CHANGE_SCALE = 4.0
#: Where pillars are parked while the horizon is calibrated on an empty scene.
PARKING_SPOT = (500.0, 500.0, 2.0)


@dataclass
class ForageConfig:
    """Task settings for `ForageEnv`.

    Attributes:
        terrain: Key into `flyplay.build.TERRAINS`.
        n_pillars: Obstacles allocated in the scene. They are repositioned each
            episode, so one compiled model serves the whole run.
        source_distance: (min, max) distance of the food from the fly, in mm.
        source_bearing_deg: Max |bearing| of the food from the spawn heading.
        reach_radius: Distance at which the food counts as found, in mm.
        episode_seconds: Time limit.
        action_hz: Policy decision rate.
        vision_hz: How often the eyes are actually rendered. Rendering costs
            25 ms against 0.106 ms for a physics step, so refreshing vision at
            the policy rate would drop the simulation to 0.28x real time; at
            20 Hz it stays at 0.64x. In between, the last features are held.
        decimation: Physics steps per hybrid-controller update.
        w_progress: Reward per mm of distance closed on the food.
        w_collision: Penalty per second spent touching a pillar.
        w_smooth: Penalty on squared change in action.
        r_found: Bonus for reaching the food.
        r_fall: Penalty for falling or leaving the terrain.
        fall_upright / fall_height: Failure thresholds, as in `flyplay.env`.
        deaf_to_odor / blind: Zero out one sense. Used for the ablation study.
    """

    terrain: str = "flat"
    n_pillars: int = 2
    source_distance: tuple[float, float] = (18.0, 26.0)
    source_bearing_deg: float = 50.0
    reach_radius: float = 2.5
    episode_seconds: float = 8.0
    action_hz: float = 100.0
    vision_hz: float = 20.0
    decimation: int = DEFAULT_DECIMATION
    w_progress: float = 8.0
    w_collision: float = 6.0
    w_smooth: float = 0.05
    r_found: float = 60.0
    r_fall: float = -20.0
    fall_upright: float = 0.2
    fall_height: float = 0.8
    warmup_seconds: float = 0.2
    settle_seconds: float = 0.1
    deaf_to_odor: bool = False
    blind: bool = False


OBS_LAYOUT: tuple[tuple[str, int], ...] = (
    ("odor_level", 1),  # log-compressed mean intensity
    ("odor_asymmetry", 1),  # (left - right) / mean, scaled
    ("odor_change", 1),  # d(log intensity)/dt -- scale-free approach signal
    ("vision", N_VISION_FEATURES),  # bilateral dark-mass features
    ("velocity", 2),  # forward / lateral speed, body frame
    ("yaw_rate", 1),
    ("height", 1),
    ("upright", 1),
    ("cpg_phase", 12),
    ("cpg_magnitude", 6),
    ("leg_contact", 6),
    ("prev_action", 2),
)
OBS_DIM = sum(n for _, n in OBS_LAYOUT)


def _build_slices() -> dict[str, slice]:
    out, start = {}, 0
    for name, width in OBS_LAYOUT:
        out[name] = slice(start, start + width)
        start += width
    return out


#: Named slices into an observation, so scripts never hard-code indices.
OBS_SLICES = _build_slices()


def obs_field(obs: np.ndarray, name: str) -> np.ndarray:
    """Pull one named block out of an observation (or a batch of them)."""
    return np.asarray(obs)[..., OBS_SLICES[name]]


class ForageEnv(gym.Env):
    """Odour-guided foraging with visual obstacle avoidance.

    **Action** — ``Box(-1, 1, shape=(2,))``, the left/right descending signal,
    exactly as in `flyplay.env.FlyNavEnv`, so a policy trained there is a
    sensible initialisation.

    **Observation** — see `OBS_LAYOUT`. Sensory and proprioceptive only.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: ForageConfig | None = None, *, seed: int | None = None):
        super().__init__()
        self.config = config or ForageConfig()
        if self.config.terrain not in TERRAINS:
            raise ValueError(
                f"Unknown terrain {self.config.terrain!r}. "
                f"Choose from {sorted(TERRAINS)}."
            )
        build_seed = 0 if seed is None else int(seed)

        # One attractive source; its position is rewritten every episode.
        self.odor_field = OdorField(
            sources=[OdorSource(pos=(20.0, 0.0, 1.5), peak=(1.0,))]
        )
        self.fs = build(
            self.config.terrain,
            vision=True,
            odor_field=self.odor_field,
            pillars=tuple(PARKING_SPOT[:2] for _ in range(self.config.n_pillars)),
            seed=build_seed,
        )
        self.walker = Walker(self.fs, decimation=self.config.decimation)

        physics_dt = self.fs.sim.timestep
        self.steps_per_action = _exact_ratio(
            1.0 / self.config.action_hz, physics_dt, "action_hz"
        )
        self.actions_per_look = _exact_ratio(
            1.0 / self.config.vision_hz, 1.0 / self.config.action_hz, "vision_hz"
        )
        self.max_steps = int(round(self.config.episode_seconds * self.config.action_hz))

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        self.retina = RetinaFeatures()
        #: Most recent raw ommatidia readouts, for visualisation.
        self.last_readouts: np.ndarray | None = None
        self._calibrate_horizon()

        self._prev_action = np.zeros(2, dtype=np.float32)
        self._prev_distance = 0.0
        self._prev_odor = 0.0
        self._vision = np.zeros(N_VISION_FEATURES, dtype=np.float32)
        self._step_count = 0
        self._ground_z = 0.0
        self._collisions = 0
        self._path: list[np.ndarray] = []

    # --- setup ----------------------------------------------------------

    def _calibrate_horizon(self) -> None:
        """Pin the visual horizon using a scene with nothing in it.

        The floor already darkens roughly half the ommatidia, so the features
        only look above a horizon row. Calibrating with a pillar in view would
        push that row above the pillar and make obstacles invisible -- so the
        pillars are parked far away for this one reading.
        """
        for gid in self.fs.pillar_geom_ids:
            move_geom(self.fs.sim, int(gid), PARKING_SPOT)
        self.fs.sim.reset()
        self.fs.sim.warmup(0.3)
        self.retina.calibrate_horizon(
            self.fs.sim.get_ommatidia_readouts(self.fs.name)
        )

    def _layout_episode(self, rng) -> None:
        """Place the food and scatter the pillars along the path to it."""
        start = self.fs.thorax_pos()
        heading = self.fs.yaw()

        distance = float(rng.uniform(*self.config.source_distance))
        bearing = float(
            np.deg2rad(
                rng.uniform(-self.config.source_bearing_deg, self.config.source_bearing_deg)
            )
        )
        angle = heading + bearing
        source_xy = start[:2] + distance * np.array([np.cos(angle), np.sin(angle)])
        self.odor_field.sources[0].pos = (float(source_xy[0]), float(source_xy[1]), 1.5)
        for gid in self.fs.marker_geom_ids:
            move_geom(self.fs.sim, int(gid), self.odor_field.sources[0].pos)

        # Pillars sit along the straight line to the food, nudged sideways so
        # some episodes need a detour and others are nearly clear.
        along = np.array([np.cos(angle), np.sin(angle)])
        across = np.array([-np.sin(angle), np.cos(angle)])
        fractions = np.linspace(0.35, 0.7, max(len(self.fs.pillar_geom_ids), 1))
        for gid, frac in zip(self.fs.pillar_geom_ids, fractions):
            offset = float(rng.uniform(-1.5, 1.5))
            xy = start[:2] + frac * distance * along + offset * across
            move_geom(self.fs.sim, int(gid), (float(xy[0]), float(xy[1]), 2.0))

        mj.mj_forward(self.fs.sim.mj_model, self.fs.sim.mj_data)

    # --- gymnasium API --------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        rng = self.np_random

        self.walker.reset(
            seed=int(rng.integers(0, 2**31 - 1)),
            warmup_s=self.config.warmup_seconds,
        )
        samples = []
        for _ in range(10):
            self.walker.run_for(self.config.settle_seconds / 10)
            samples.append(self.fs.thorax_pos()[2])
        self._ground_z = float(np.median(samples))

        self._layout_episode(rng)

        start = self.fs.thorax_pos()
        self._prev_distance = self._distance()
        self._prev_action[:] = 0.0
        self._prev_odor = self._odor_log_level()
        self._vision = self._look()
        self._step_count = 0
        self._collisions = 0
        self._path = [start[:2].copy()]

        return self._observe(0.0), {
            "source": np.array(self.odor_field.sources[0].pos[:2]),
            "start": start.copy(),
            "distance": self._prev_distance,
        }

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        mid = 0.5 * (SIGNAL_HIGH + SIGNAL_LOW)
        half = 0.5 * (SIGNAL_HIGH - SIGNAL_LOW)
        self.walker.descending_signal = mid + half * action

        touched = False
        for _ in range(self.steps_per_action):
            self.walker.physics_step()
            touched = touched or self.fs.touching_pillar()

        self._step_count += 1
        if self._step_count % self.actions_per_look == 0:
            self._vision = self._look()

        pos = self.fs.thorax_pos()
        self._path.append(pos[:2].copy())
        distance = self._distance()
        dt = 1.0 / self.config.action_hz

        reward = self.config.w_progress * (self._prev_distance - distance)
        if touched:
            reward -= self.config.w_collision * dt
            self._collisions += 1
        reward -= self.config.w_smooth * float(
            np.sum((action - self._prev_action) ** 2)
        )

        terminated = False
        outcome = "running"
        if distance <= self.config.reach_radius:
            reward += self.config.r_found
            terminated, outcome = True, "found"
        elif self._has_fallen(pos):
            reward += self.config.r_fall
            terminated, outcome = True, "fell"
        elif self.fs.out_of_bounds():
            reward += self.config.r_fall
            terminated, outcome = True, "out_of_bounds"

        truncated = (not terminated) and self._step_count >= self.max_steps
        if truncated:
            outcome = "timeout"

        odor_now = self._odor_log_level()
        obs = self._observe((odor_now - self._prev_odor) / dt)
        self._prev_odor = odor_now
        self._prev_distance = distance
        self._prev_action = action.copy()

        info = {
            "distance": distance,
            "outcome": outcome,
            "collisions": self._collisions,
            "touching": touched,
        }
        if terminated or truncated:
            info["is_success"] = outcome == "found"
            info["path"] = np.asarray(self._path)
            info["source"] = np.array(self.odor_field.sources[0].pos[:2])
            info["pillars"] = self.pillar_positions()
        return obs, float(reward), terminated, truncated, info

    def close(self) -> None:
        self.fs.close()

    # --- helpers --------------------------------------------------------

    def pillar_positions(self) -> np.ndarray:
        """Current ``(n_pillars, 2)`` obstacle positions."""
        if self.fs.pillar_geom_ids.size == 0:
            return np.zeros((0, 2))
        return self.fs.sim.mj_model.geom_pos[self.fs.pillar_geom_ids][:, :2].copy()

    @property
    def elapsed(self) -> float:
        return self._step_count / self.config.action_hz

    @property
    def step_count(self) -> int:
        return self._step_count

    def _distance(self) -> float:
        source = np.array(self.odor_field.sources[0].pos[:2])
        return float(np.linalg.norm(source - self.fs.thorax_pos()[:2]))

    def _odor_log_level(self) -> float:
        """Natural log of mean odour intensity.

        The temporal cue is taken in the log domain because ``d(ln I)/dt`` is
        ``2 v / d`` for an inverse-square field -- it depends only on how fast
        the fly closes on the source relative to how far away it is, so it
        stays informative from 25 mm all the way in. The raw derivative spans
        three orders of magnitude over the same approach and saturates long
        before the fly arrives.
        """
        if self.config.deaf_to_odor:
            return 0.0
        intensity = float(self.odor_field.read(self.fs.sim)[:, 0].mean())
        return float(np.log(intensity + 1e-9))

    def _look(self) -> np.ndarray:
        if self.config.blind:
            return np.zeros(N_VISION_FEATURES, dtype=np.float32)
        readouts = self.fs.sim.get_ommatidia_readouts(self.fs.name)
        # Kept so a viewer can show what the fly sees without paying for a
        # second eye render (they cost 25 ms each).
        self.last_readouts = readouts
        return self.retina.extract(readouts)

    def _has_fallen(self, pos: np.ndarray) -> bool:
        if self.fs.upright() < self.config.fall_upright:
            return True
        return bool(pos[2] - self._ground_z < -self.config.fall_height)

    def _observe(self, odor_change: float) -> np.ndarray:
        if self.config.deaf_to_odor:
            level, asym = 0.0, 0.0
            odor_change = 0.0
        else:
            intensities = self.odor_field.read(self.fs.sim)
            level = float(np.log10(intensities[:, 0].mean() + 1e-6) + 6.0) / 6.0
            asym = float(OdorField.asymmetry(intensities)[0]) * ASYMMETRY_GAIN

        vel = self.fs.body_velocity()
        phases = self.walker.cpg_phases
        obs = np.concatenate(
            [
                [level, asym, np.tanh(odor_change / ODOR_CHANGE_SCALE)],
                self._vision * VISION_GAIN,
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


def _exact_ratio(numerator: float, denominator: float, label: str) -> int:
    ratio = numerator / denominator
    if ratio < 1 or abs(ratio - round(ratio)) > 1e-6:
        raise ValueError(
            f"{label} must divide the faster rate evenly; got a ratio of {ratio}."
        )
    return int(round(ratio))

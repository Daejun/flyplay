"""The central complex: a compass, a path integrator and a remembered goal.

The mushroom body learns what a smell or a colour predicts; it cannot learn a
*place*. Flies can -- they find a cool spot on a hot floor faster trial after
trial by the panorama around them, and need their ellipsoid-body ring neurons
for it, not the mushroom body (Ofstad et al. 2011). This module is that other
memory, at the level of what its parts compute rather than of neurons:

    compass      a bump of activity over 16 wedges (EPG neurons) that moves with
                 the fly's own turning, and drifts: heading error diffuses at
                 1.82e-3 rad^2/s, about 27 degrees s.d. after a minute
                 (Turner-Evans et al. 2017's model; bump 82 degrees wide at
                 half height, Seelig & Jayaraman 2015)
    landmarks    ring neurons read the panorama's darkness round the fly. Where
                 a ring neuron and a wedge are active together their inhibitory
                 synapse weakens (Kim et al. 2019; Fisher et al. 2019), so a
                 scene seen again pulls the bump to where it sat before. Kept
                 here as its simplest sufficient form: where the panorama is
                 darkest, as a bearing in compass coordinates, learned slowly
                 while compass and scene agree. A single dark bar anchors the
                 real bump the same way (Seelig & Jayaraman 2015). A new fly
                 has learned no bearing and is pulled nowhere.
    path         the fly's own motion, turned by the *estimated* heading and
    integrator   summed: an estimate of where it is since it was put down, with
                 the compass's errors in it (a vector version of Stone et al.
                 2017's CPU4 memory)
    goal         a remembered position in that frame (FC2), steered toward as
                 the sine of the bearing error (PFL3; Westeinde et al. 2024,
                 Mussells Pires et al. 2024)

Being picked up and put down loses the compass's alignment: `put_down` gives the
bump a random heading and zeroes the integrator. Only a learned panorama brings
the old alignment back, which is why a place remembered in one trial is found
again in the next with landmarks and not without them.

**What was tried first, and failed** (heat maze, one fly, trials of 40 s):
Hebbian ring-to-wedge weights updated with the bump they were pulling swung the
heading estimate 100-200 degrees within seconds and sent the path integrator
500 mm across a 100 mm room; a whole-panorama template matched by correlation
held steady but re-anchored only three put-downs in six, because the room's
walls fill much of the view near them and correlate with any turn.

Which ommatidium looks where was measured by rendering (`scripts/
19_eye_azimuths.py`, `flyplay/data/ommatidia_azimuth.npz`).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from flyplay.visual_pathway import eye_layout, photoreceptors

N_WEDGES = 16
#: Width of the bump at half height, degrees (Seelig & Jayaraman 2015: 82 +/- 12
#: with a stripe).
BUMP_FWHM_DEG = 82.0
#: The von Mises concentration with that width: cos(FWHM/2) = 1 + ln(0.5)/kappa.
BUMP_KAPPA = float(np.log(2.0) / (1.0 - np.cos(np.deg2rad(BUMP_FWHM_DEG / 2))))
#: Heading error diffusion, rad^2/s (Turner-Evans et al. 2017).
HEADING_DIFFUSION = 1.82e-3
#: The ring neurons read the ommatidia that see the upper half of the drum from
#: the room's centre, and only dark features there: readings under
#: `DARK_BELOW` (the drum's stripes read about 0.04, the room's grey walls 0.4
#: and more, the sky 0.99). Counting all darkness, the walls -- which fill much
#: of the view near them -- made the bearing point at the nearest wall with as
#: much strength (0.19-0.33) as the drum's dark sector gave from the middle.
#: Measured over 8 places x 4 headings with dark features only: the room
#: without the drum reads 0.00-0.09, with it 0.05-0.98 (median 0.54), mostly
#: pointing into the drum's 90-degree dark sector.
RING_UPPER_FRACTION = 0.5
DARK_BELOW = 0.3
#: How fast a clear landmark pulls the bump, 1/s at full trust. Measured in the
#: heat maze with `VIEW_ASYMMETRY` below (4 flies, 8 trials of 45 s, each put
#: down at the centre with a random heading; scratch `tune_calibrate.py`), time
#: to reach the cool spot in the last two trials, landmarks against none: at 0.4,
#: 14.3 s against 27.5 s, every pair faster, 9 of 32 trials never reaching it
#: against 13 -- but the landmark flies were already faster in the first two
#: trials (25.4 s against 34.1 s), why was not looked into. At 1.0, 39.2 s and 15
#: of 32 never, although the heading offset agreed across put-downs as well
#: (resultant over trials 2-8: 0.72-0.96 at 1.0, 0.41-0.96 at 0.4, 0.32-0.70
#: without landmarks).
VISUAL_GAIN = 0.4
#: Landmark bearing learning rate, 1/s. Remapping takes minutes of training
#: (Fisher et al. 2019; Kim et al. 2019); a model choice.
LANDMARK_RATE = 1.0 / 30.0
#: Seconds of viewing before the learned bearing is used, and the largest
#: mismatch at which it keeps learning: a bearing learned while the compass is
#: off would blur. Model choices.
LANDMARK_MATURE_S = 20.0
LEARN_WITHIN_DEG = 30.0
#: How clear the view must be (`landmark_view` length) to trust it, not at all
#: and fully; and how consistent the learned bearing (its resultant). Model
#: choices, set against the values measured for `DARK_BELOW` (a room without the
#: drum reads up to 0.09).
VIEW_ASYMMETRY = (0.3, 0.7)
LEARNED_CONSISTENCY = (0.2, 0.5)
#: How long a remembered goal lasts, s: place memory held at least 2 h in the
#: heat maze (Ofstad et al. 2011).
GOAL_TAU = 7200.0

DATA = Path(__file__).with_name("data")


@lru_cache(maxsize=1)
def _ring_layout() -> tuple[np.ndarray, np.ndarray]:
    """(azimuth in radians, mask) of the ommatidia the ring neurons read, (2, 721)."""
    with np.load(DATA / "ommatidia_azimuth.npz") as data:
        azimuth = data["azimuth_deg"].astype(float)
    _, _, row, _ = eye_layout()
    seen = np.isfinite(azimuth)
    use = np.zeros_like(seen)
    for eye in (0, 1):
        cut = np.quantile(row[seen[eye]], RING_UPPER_FRACTION)
        use[eye] = seen[eye] & (row <= cut)
    return np.deg2rad(np.where(use, azimuth, 0.0)), use


def landmark_view(readouts: np.ndarray) -> complex:
    """Where the dark features of the panorama are, egocentric: their mean
    bearing as a complex number, angle counterclockwise from straight ahead,
    length how clear (resultant length, scaled down when few ommatidia see
    anything dark)."""
    azimuth, use = _ring_layout()
    light = photoreceptors(np.asarray(readouts, dtype=float))
    dark = np.clip((DARK_BELOW - light) / DARK_BELOW, 0.0, 1.0)[use]
    total = float(dark.sum())
    if total <= 0.0:
        return 0j
    return complex(np.sum(dark * np.exp(1j * azimuth[use])) / total) * min(1.0, 10.0 * float(dark.mean()))


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _ramp(value: float, bounds: tuple[float, float]) -> float:
    return float(np.clip((value - bounds[0]) / (bounds[1] - bounds[0]), 0.0, 1.0))


class CentralComplex:
    """One fly's compass, integrator, landmark memory and goals. See the module
    docstring. `rng` supplies heading noise and the heading after `put_down`."""

    WEDGE_ANGLES = np.linspace(0.0, 2.0 * np.pi, N_WEDGES, endpoint=False)

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.reset()

    def reset(self) -> None:
        """A new fly: no landmark learned, no goal."""
        #: Learned landmark bearing in compass coordinates (angle), and how
        #: consistently and clearly it was seen there (length).
        self.landmark = 0j
        self.viewed = 0.0
        self.goal: np.ndarray | None = None
        self.goal_strength = 0.0
        self.put_down()

    def put_down(self) -> None:
        """Picked up and set down: the compass loses its alignment, the
        integrator starts again, the last meal's place is forgotten."""
        self.heading = float(self.rng.uniform(-np.pi, np.pi))
        self.position = np.zeros(2)
        self.food: np.ndarray | None = None

    def bump(self) -> np.ndarray:
        """Wedge activity, peak 1."""
        return np.exp(BUMP_KAPPA * (np.cos(self.WEDGE_ANGLES - self.heading) - 1.0))

    def step(self, turned: float, moved: np.ndarray, dt: float, view: complex | None = None,
             view_dt: float = 0.0) -> None:
        """Advance by one action step.

        Args:
            turned: The fly's own rotation this step, rad, counterclockwise.
            moved: Its displacement this step in its own body frame, mm.
            dt: Step length, s.
            view: `landmark_view` if the eyes were read this step.
            view_dt: Time since the eyes were last read, for the visual terms.
        """
        self.heading = _wrap(self.heading + turned + np.sqrt(2.0 * HEADING_DIFFUSION * dt) * self.rng.normal())
        if view is not None and view_dt > 0.0:
            self._see(view, view_dt)
        c, s = np.cos(self.heading), np.sin(self.heading)
        self.position += (c * moved[0] - s * moved[1], s * moved[0] + c * moved[1])
        if self.goal_strength > 0.0:
            self.goal_strength *= float(np.exp(-dt / GOAL_TAU))

    def _see(self, view: complex, view_dt: float) -> None:
        lopsided = _ramp(abs(view), VIEW_ASYMMETRY)
        if lopsided <= 0.0:
            return
        mismatch = 0.0
        if self.viewed >= LANDMARK_MATURE_S:
            # Where the compass should point, given where the landmark is seen.
            seen_heading = _wrap(float(np.angle(self.landmark)) - float(np.angle(view)))
            mismatch = _wrap(seen_heading - self.heading)
            trust = lopsided * _ramp(abs(self.landmark), LEARNED_CONSISTENCY)
            if trust > 0.0:
                self.heading = _wrap(self.heading + min(1.0, VISUAL_GAIN * trust * view_dt) * mismatch)
                mismatch = _wrap(seen_heading - self.heading)
        if self.viewed < LANDMARK_MATURE_S or abs(mismatch) <= np.deg2rad(LEARN_WITHIN_DEG):
            bearing = np.exp(1j * (float(np.angle(view)) + self.heading))
            rate = min(1.0, LANDMARK_RATE * view_dt)
            self.landmark += rate * (lopsided * bearing - self.landmark)
            self.viewed += view_dt

    def idle(self, seconds: float) -> None:
        """Time off-stage: the goal fades as it would have."""
        self.goal_strength *= float(np.exp(-max(0.0, seconds) / GOAL_TAU))

    def remember_goal(self) -> None:
        self.goal = self.position.copy()
        self.goal_strength = 1.0

    def mark_food(self) -> None:
        self.food = self.position.copy()

    def steer(self, target: np.ndarray) -> tuple[float, float]:
        """(turn toward `target` as the sine of the bearing error, distance), both
        from the fly's own estimates."""
        vector = np.asarray(target) - self.position
        bearing = float(np.arctan2(vector[1], vector[0]))
        return float(np.sin(bearing - self.heading)), float(np.linalg.norm(vector))

    # --- carried across processes (flyplay.session) ------------------------------

    def state(self) -> tuple[dict, dict[str, np.ndarray]]:
        meta = {"heading": self.heading, "position": self.position.tolist(),
                "goal": None if self.goal is None else self.goal.tolist(), "goal_strength": self.goal_strength,
                "food": None if self.food is None else self.food.tolist(),
                "landmark": [self.landmark.real, self.landmark.imag], "viewed": self.viewed}
        return meta, {}

    def load(self, meta: dict, arrays: dict[str, np.ndarray]) -> None:
        if meta.get("landmark"):
            self.landmark = complex(*meta["landmark"])
            self.viewed = float(meta.get("viewed", 0.0))
        self.goal = None if meta.get("goal") is None else np.array(meta["goal"], dtype=float)
        self.goal_strength = float(meta.get("goal_strength", 0.0))

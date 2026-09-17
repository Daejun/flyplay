"""Olfactory conditioning: train on one odour, then let the fly choose.

Deliberately not a `gym.Env`. A gym environment resets its world and its agent
together, and that is exactly wrong here -- the mushroom body has to carry its
synaptic weights across trials or there is no learning to measure. What resets
between trials is the physics; what persists is the memory. So this is an
experiment runner, closer in shape to a behavioural protocol than to an RL loop.

The protocol follows the T-maze assay the mushroom-body literature is built on:

    training trial   one odour, alone. If it is the CS+, dopamine fires while
                     the fly is inside the odour.
    test trial       both odours, equidistant, sides randomised. Score the
                     preference index from time spent near each.
    reversal         swap which odour is the CS+ and train again.

**Steering.** The mushroom body reports learned valence only; a novel odour
reads exactly zero. Naive attraction to an unfamiliar smell is not a mushroom
body computation in the fly either -- it is the lateral horn, a parallel
pathway from the same projection neurons. `INNATE_VALENCE` stands in for it:

    drive = INNATE_VALENCE + valence        lateral horn + mushroom body
    turn  = TURN_GAIN * drive * asymmetry   toward the smell if drive > 0

so a naive fly approaches, a punished odour drives `valence` negative, and once
it passes `-INNATE_VALENCE` the same steering law walks the fly away. Avoidance
is not a second controller; it is the sign flip of the one that was there.

**Colour** (``ConditioningConfig.modality = "colour"``, see `colour_config`) runs
the visual assay of Vogt et al. (2014) through the same mushroom body. The
floor is the stimulus: a training trial paints every tile the CS colour for 60
s with 1 s dopamine pulses every 5 s, dark intervals of 12 s separate trials,
and the test is a checkerboard of both colours scored by time on each.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any

import mujoco as mj
import numpy as np

from flyplay.arena import FLOOR_COLOURS, move_geom
from flyplay.build import build
from flyplay.control import DEFAULT_DECIMATION, Walker
from flyplay.env import SIGNAL_HIGH, SIGNAL_LOW
from flyplay.mushroom_body import Compartment, MushroomBody
from flyplay.odor import OdorField, OdorSource
from flyplay.olfactory import OlfactoryFrontEnd
from flyplay.visual_pathway import VisualFrontEnd

#: Innate attraction to any odour, standing in for the lateral horn. Sized
#: against the mushroom body's range: a fully depressed approach MBON drives
#: valence to about -0.7, so 0.35 puts the behavioural flip at roughly half of
#: full depression -- reachable in a few trials, but not on the first one.
INNATE_VALENCE = 0.35
#: Turn command per unit of (drive x asymmetry). Measured, not guessed: with a
#: source 10-40 mm away at the 35 degree bearing a test trial uses, asymmetry
#: runs 0.009 to 0.040 and averages 0.020, so this puts a confident approach at
#: a turn command of ~0.6. The +/-0.04 quoted in `flyplay.env_multimodal` is the
#: near-lateral case; head-on the signal is 0.0006 and no gain rescues it, which
#: is why sources are never placed straight ahead here.
TURN_GAIN = 80.0
#: Clamp on the inverse-square falloff, in mm. This is the number that keeps
#: the Kenyon-cell code stable: unclamped, a fly with its head on the source
#: sees intensity 2.1, twenty times past the range where the code holds its
#: shape, and the mushroom body would file "A up close" as a different odour
#: from "A at a distance". At 3.0 mm the intensity ceiling is 0.111, which is
#: exactly the top of the measured working range.
ODOR_MIN_DISTANCE = 3.0
#: Dopamine fires only while the fly is actually in the odour. Threshold is on
#: mean sensor intensity and corresponds to roughly 12 mm from the source.
US_THRESHOLD = 0.008
#: Odour A and odour B as unit vectors in the two-dimensional odour space.
ODOUR_A = np.array([1.0, 0.0])
ODOUR_B = np.array([0.0, 1.0])
ODOUR_NAMES = ("A", "B")
#: Marker colours, so the two odours are told apart in the viewer and videos.
ODOUR_RGBA = ((0.95, 0.45, 0.10, 1.0), (0.25, 0.55, 0.95, 1.0))
#: Tiles per side of the colour task's floor grid. Measured at 20 mm tiles: not
#: one of the 736 lower-field ommatidia changes when the grid is doubled to 24 a
#: side, so its edge is out of sight; the 144 tiles add ~8 ms to a 25 ms render.
COLOUR_GRID_SIDE = 12
#: Ablations `ConditioningConfig.blocked` accepts.
BLOCKABLE = ("visual_kc", "colour_vpn", "brightness_vpn", "punish_dan", "reward_dan")
#: Steering stays inside this band when visual kinesis changes speed, so a
#: turn always keeps at least 0.3 of authority on each side.
KINESIS_RANGE = (0.7, 1.3)


@dataclass
class ConditioningConfig:
    """Protocol and task settings.

    Attributes:
        source_distance: How far a source is placed from the fly, in mm.
        bearing_deg: Half-angle of the cone sources are placed in, measured
            from the spawn heading. A test trial puts one odour at each edge;
            a training trial picks a random bearing inside it. **Never zero.**
            Head-on, the left-right asymmetry is 0.0006 and the fly has no
            steering signal to act on, so a trained fly could not demonstrate
            avoidance even if it had learned it.
        near_radius: Radius counted as "at" a source. Reported as a diagnostic;
            the preference index does not use it.
        score_from: Fraction of the trial to skip before scoring. The fly
            starts equidistant from both sources, so the opening is
            uninformative by construction.
        trial_seconds: Length of one trial. The fly walks at ~12 mm/s, so this
            sets how far past a source it can travel.
        action_hz: Rate at which the mushroom body reads and the steering law
            commands. Matches `ForageEnv` so the numbers are comparable.
        decimation: Physics steps per hybrid-controller update.
        us_delay: Seconds the fly must already have been inside the odour
            before dopamine fires. Measured from first detection, not from the
            start of the trial -- classical conditioning needs the CS to lead
            the US, and a trial-clock delay just misses the encounter entirely
            (measured: the fly crosses the odour between 0.7 s and 2.2 s).
        innate_valence / turn_gain: Steering law, see module docstring.
        eta / recovery: Plasticity rates, see `flyplay.mushroom_body`.
        shared_dan: Ablation -- route both compartments from one scalar.
        frozen: Ablation -- no plasticity at all. The floor for every measure.
        differential: Train with CS+ and CS- presentations alternating, as the
            T-maze assay does, instead of the CS+ alone. Without it the old
            CS+ is never smelled during reversal training, so nothing but
            passive forgetting can act on its memory.
        extinction: Compartment the omission-of-punishment signal writes to,
            or None for no such signal. "avoid" is the biological wiring;
            "approach" is the control that routes the same signal to the wrong
            place. See `_omission`.
        extinction_gain: Dopamine per unit of predicted punishment. 1.0 makes
            an omission as strong as a real punishment would have been when
            the prediction is maximally aversive (valence -1). Chosen on that
            ground, not fitted to any behavioural result.
    """

    source_distance: float = 20.0
    bearing_deg: float = 35.0
    near_radius: float = 8.0
    score_from: float = 0.5
    trial_seconds: float = 5.0
    action_hz: float = 100.0
    decimation: int = DEFAULT_DECIMATION
    us_delay: float = 0.5
    innate_valence: float = INNATE_VALENCE
    turn_gain: float = TURN_GAIN
    us_threshold: float = US_THRESHOLD
    odor_min_distance: float = ODOR_MIN_DISTANCE
    #: Plasticity rates. These override `flyplay.mushroom_body`'s defaults
    #: because the timings differ: the module was calibrated on synthetic
    #: trials carrying 4 s of dopamine, and a real trial here delivers 1.1 s
    #: (measured) because the fly crosses the odour and keeps walking. `eta` is
    #: raised to match the dopamine budget per trial. `recovery` is slowed to a
    #: 100 s time constant because the memory now has to survive a whole test
    #: block: at the module's 30 s, six unreinforced test trials took the CS+
    #: valence from -0.31 to -0.12, losing 63% of the depression (measured --
    #: before the per-odour steering fix, so that run's behaviour is not
    #: comparable, only its synaptic decay). The price is reversal: the full
    #: sweep reverses to PI -0.30 against +0.49 for acquisition, because the old
    #: memory is still 15% deep when the reversal test starts.
    eta: float | None = 8.0
    recovery: float | None = 0.01
    shared_dan: bool = False
    frozen: bool = False
    differential: bool = False
    extinction: str | None = None
    extinction_gain: float = 1.0
    warmup_seconds: float = 0.2
    #: "odour" -- the task above -- or "colour", the floor-colour assay of Vogt
    #: et al. (2014). Everything below is read only by the colour task; build
    #: its config with `colour_config`.
    modality: str = "odour"
    #: The colour task's two stimuli, keys of `flyplay.arena.FLOOR_COLOURS`.
    #: Test preference is toward the second: with the default it is Vogt's
    #: PI_G, positive for green.
    stimuli: tuple[str, str] = ("blue", "green")
    #: Length of a test trial; None means `trial_seconds`.
    test_seconds: float | None = None
    #: Darkness before every trial but the first, and extra darkness before a
    #: test that follows training. Recovery only, applied exactly
    #: (`MushroomBody.idle`) -- nothing is seen, so nothing else can happen.
    iti_seconds: float = 0.0
    retention_seconds: float = 0.0
    #: "punish" depresses the approach compartment (PPL1 in the fly), "reward"
    #: the avoid compartment (PAM).
    reinforcer: str = "punish"
    #: Dopamine timing on a colour training trial: a `us_pulse` second pulse at
    #: the end of every `us_period` seconds, so the colour leads each pulse by 4
    #: s. Vogt et al. (2014): twelve 1 s shocks within the 60 s of the CS+.
    us_period: float = 5.0
    us_pulse: float = 1.0
    #: How often the eyes are rendered, Hz. Between renders the last readouts
    #: are held; the floor under a walking fly changes over ~1 s (20 mm tiles).
    vision_hz: float = 10.0
    tile_size: float = 20.0
    #: Colour steering, in units of normalised eye valence (valence divided by
    #: the visual code's scale, so -1 is a fully depressed colour). See
    #: `ConditioningExperiment._visual_command`.
    visual_gain: float = 0.0
    visual_kinesis: float = 0.0
    border_turn_seconds: float = 0.0
    border_threshold: float = 0.25
    #: Ablations, any of `BLOCKABLE`.
    blocked: tuple[str, ...] = ()


#: Trial kinds. "train" presents one odour with punishment, "expose" presents
#: one odour without (the CS- of differential conditioning), "test" presents
#: both, unreinforced, and is the only kind that is scored.
TRIAL_KINDS = ("train", "expose", "test")


@dataclass
class TrialResult:
    """What one trial produced. `preference` is None outside test trials."""

    kind: str  # one of TRIAL_KINDS
    index: int
    #: The odour punished on this trial, or None -- always None for "expose"
    #: and "test".
    cs_plus: str | None
    odours: tuple[str, ...]
    #: Seconds spent within `near_radius` of each presented source.
    time_near: dict[str, float]
    #: (time_near[B] - time_near[A]) / (sum), or None for a training trial.
    preference: float | None
    #: Mushroom-body valence for each odour, read at the end of the trial.
    valence: dict[str, float]
    #: MBON output per compartment per odour, read at the end of the trial.
    mbon: dict[str, dict[str, float]]
    #: Seconds of punishment dopamine delivered.
    dopamine_seconds: float
    path: np.ndarray = field(repr=False)
    sources: dict[str, np.ndarray] = field(repr=False)
    #: Omission dopamine delivered, summed over the trial, in dopamine-seconds
    #: (the signal is graded, so this is an integral rather than a duration).
    omission_dopamine: float = 0.0
    #: Protocol block this trial belongs to -- a key of `PROTOCOL`, set by
    #: whoever runs the schedule. Empty when a trial is run on its own.
    block: str = ""
    #: Colour task only: how the floor was painted, ``("uniform", colour)`` or
    #: ``("checker", even, odd)`` -- with the path, enough to recover which
    #: colour the fly stood on at every step. None for odour trials.
    floor: tuple | None = None


class ConditioningExperiment:
    """One simulated fly, run through a protocol of trials.

    The simulation and the mushroom body are both built once. `run_trial`
    resets the physics and lays out the sources; it never touches the synapses.
    Only `reset_memory` does that, and it means "a different fly".
    """

    def __init__(
        self,
        config: ConditioningConfig | None = None,
        *,
        seed: int = 0,
        wiring_seed: int | None = None,
    ):
        self.config = config or ConditioningConfig()
        self.rng = np.random.default_rng(seed)

        # Two sources, one per odour. Both exist for the whole run; a training
        # trial parks the one it is not using far away.
        self.odor_field = OdorField(
            sources=[
                OdorSource(pos=(20.0, 0.0, 1.5), peak=(1.0, 0.0), rgba=ODOUR_RGBA[0]),
                OdorSource(pos=(20.0, 8.0, 1.5), peak=(0.0, 1.0), rgba=ODOUR_RGBA[1]),
            ],
            min_distance=self.config.odor_min_distance,
        )
        colour = self.config.modality == "colour"
        if self.config.modality not in ("odour", "colour"):
            raise ValueError(f"modality must be 'odour' or 'colour'; got {self.config.modality!r}")
        self.fs = build(
            "flat",
            odor_field=self.odor_field,
            vision=colour,
            floor_tiles=(self.config.tile_size, COLOUR_GRID_SIDE) if colour else None,
            seed=seed,
        )
        self.walker = Walker(self.fs, decimation=self.config.decimation)

        wiring = seed if wiring_seed is None else wiring_seed
        front_end = OlfactoryFrontEnd(2, seed=wiring)
        visual = VisualFrontEnd(seed=wiring) if colour else None
        n_kc = front_end.n_kc + (visual.n_kc if visual is not None else 0)
        plasticity: dict[str, Any] = {}
        if self.config.eta is not None:
            plasticity["eta"] = self.config.eta
        if self.config.recovery is not None:
            plasticity["recovery"] = self.config.recovery
        if self.config.frozen:
            plasticity["eta"] = 0.0
            plasticity["recovery"] = 0.0
        self.mb = MushroomBody(
            front_end,
            [
                Compartment("approach", sign=+1.0, n_kc=n_kc, **plasticity),
                Compartment("avoid", sign=-1.0, n_kc=n_kc, **plasticity),
            ],
            shared_dan=self.config.shared_dan,
            visual=visual,
        )
        blocked = set(self.config.blocked)
        if blocked - set(BLOCKABLE):
            raise ValueError(f"unknown ablations {sorted(blocked - set(BLOCKABLE))}; "
                             f"choose from {BLOCKABLE}")
        self.mb.visual_blocked = "visual_kc" in blocked
        if visual is not None:
            visual.silenced = {family for family, key in (("colour", "colour_vpn"),
                                                          ("brightness", "brightness_vpn"))
                               if key in blocked}

        # Kenyon-cell code for each odour presented alone. Fixed for the life
        # of the fly: the wiring never changes and the code is concentration
        # invariant, so the steering law can read a valence per odour without
        # re-running the front end every step.
        self._unit_kc = [
            self.mb.embed(front_end.encode(vector * 0.05)) for vector in (ODOUR_A, ODOUR_B)
        ]

        #: What the eyes see standing on each colour, rendered once from the
        #: spawn pose -- the visual counterpart of `_unit_kc`, for reading a
        #: colour's valence without painting the floor. Measured: a rendered
        #: floor and a synthetic one give the same code (cosine 0.99-0.999).
        self._unit_readouts: dict[str, np.ndarray] = {}
        #: Kind of the previous trial, for the retention interval before a test.
        self._last_kind: str | None = None
        if colour:
            for index in range(len(self.odor_field.sources)):
                self._park(index)
            self._sync_markers()
            self.walker.reset(seed=0, warmup_s=self.config.warmup_seconds)
            for name in self.config.stimuli:
                if name not in FLOOR_COLOURS:
                    raise ValueError(f"unknown colour {name!r}; choose from {sorted(FLOOR_COLOURS)}")
                self.fs.floor.paint_uniform(name)
                self._unit_readouts[name] = self.fs.sim.get_ommatidia_readouts(self.fs.name).copy()
        # Border-turn state for the colour steering; reset every trial.
        self._valence_trace: deque[float] = deque(maxlen=1)
        self._turn_steps_left = 0
        self._turn_sign = 0.0

        physics_dt = self.fs.sim.timestep
        self.steps_per_action = int(round(1.0 / self.config.action_hz / physics_dt))
        self.dt = 1.0 / self.config.action_hz
        self.steps_per_trial = int(round(self.config.trial_seconds * self.config.action_hz))
        self._trial_count = 0
        #: Most recent mushroom-body state, for the web viewer.
        self.last_state = None
        #: Name of the memory restored into this fly, if any. Set by load_memory.
        self.memory_name = ""
        #: Whether a restored memory carried visual synapses. Set by load_memory.
        self.memory_visual = ""

    # --- scene ----------------------------------------------------------

    def _park(self, index: int) -> None:
        """Move a source out of range without removing it from the model."""
        self.odor_field.sources[index].pos = (500.0, 500.0, 1.5)

    def _place(self, index: int, bearing: float) -> np.ndarray:
        start = self.fs.thorax_pos()
        angle = self.fs.yaw() + bearing
        xy = start[:2] + self.config.source_distance * np.array(
            [np.cos(angle), np.sin(angle)]
        )
        self.odor_field.sources[index].pos = (float(xy[0]), float(xy[1]), 1.5)
        return xy

    def _sync_markers(self) -> None:
        for gid, source in zip(self.fs.marker_geom_ids, self.odor_field.sources):
            move_geom(self.fs.sim, int(gid), source.pos)
        mj.mj_forward(self.fs.sim.mj_model, self.fs.sim.mj_data)

    # --- sensing and steering --------------------------------------------

    def _concentration(self) -> np.ndarray:
        """Odour vector at the antennae: the mushroom body's input."""
        return self.odor_field.read(self.fs.sim).mean(axis=0)

    def _odour_valences(self) -> np.ndarray:
        """Learned valence of each odour on its own, one per odour dimension.

        Cheap because the Kenyon-cell code for a given odour is fixed -- that
        is what the front end's concentration invariance buys. The codes are
        cached at construction, so this is two dot products per odour rather
        than a full pass through ORN, antennal lobe and top-k.
        """
        return np.array(
            [
                sum(c.sign * float(c.weights @ kc) for c in self.mb.compartments)
                for kc in self._unit_kc
            ]
        )

    def _turn_command(self) -> float:
        """Valence-weighted chemotaxis: steer up the gradient of what is good.

        One turn command has to come out of two odour gradients, and the naive
        way -- sum the asymmetries, scale by the valence of the mixture -- is
        exactly wrong for this task. On a test trial the two sources sit on
        opposite sides, so their asymmetries cancel to about zero and the fly
        walks straight between them no matter what it has learned. It would
        have no way to act on the difference, which is the only thing the
        experiment measures.

        So each odour channel carries its own valence and its own gradient, and
        they are combined weighted by how much of that odour is actually
        present:

            turn ~ sum_d (innate + valence_d) * asymmetry_d * share_d

        which is chemotaxis up the gradient of a valence-weighted intensity.
        A punished odour enters with a negative coefficient and pushes the same
        steering law the other way.
        """
        intensities = self.odor_field.read(self.fs.sim)
        level = intensities.mean(axis=0)
        total = float(level.sum())
        if total <= 0.0:
            return 0.0
        drive = self.config.innate_valence + self._odour_valences()
        asymmetry = OdorField.asymmetry(intensities)
        weighted = float(np.sum(drive * asymmetry * (level / total)))
        return float(np.clip(self.config.turn_gain * weighted, -1.0, 1.0))

    def _steer(self) -> np.ndarray:
        turn = self._turn_command()
        mid = 0.5 * (SIGNAL_HIGH + SIGNAL_LOW)
        half = 0.5 * (SIGNAL_HIGH - SIGNAL_LOW)
        # descending[0] < descending[1] turns left, and positive asymmetry
        # means the odour is stronger on the fly's left.
        return np.array([mid - half * turn, mid + half * turn])

    def _visual_command(self, state) -> np.ndarray:
        """Descending signal from the learned valence of what each eye sees.

        Three terms, each off when its config value is zero, so the ceiling
        test can measure what each one buys:

        - **bilateral**: turn toward the eye whose view is valued more,
          ``visual_gain * (v_left - v_right)``. Blind to a boundary met head-on,
          where both eyes see the same.
        - **kinesis**: walk faster over a floor valued less, which alone lowers
          the time spent there.
        - **border turn**: a full-authority turn for `border_turn_seconds` when
          the valence under the fly drops by `border_threshold` within 0.3 s --
          the sharp turns at a zone border that Tao et al. (2020) found drive
          walking flies' zone choice in still air.

        Valences are normalised by the visual code's scale, so -1 means every
        synapse of that view fully depressed.
        """
        cfg = self.config
        if state.visual is None:
            return np.array([1.0, 1.0])
        v = self.mb.eye_valences(state.visual) / self.mb.visual_scale
        mean = 0.5 * float(v[0] + v[1])
        turn = cfg.visual_gain * float(v[0] - v[1])
        mid = float(np.clip(1.0 - cfg.visual_kinesis * mean, *KINESIS_RANGE))
        if cfg.border_turn_seconds > 0.0:
            if self._turn_steps_left > 0:
                self._turn_steps_left -= 1
                turn, mid = self._turn_sign, 1.0
            elif self._valence_trace and max(self._valence_trace) - mean > cfg.border_threshold:
                self._turn_steps_left = int(round(cfg.border_turn_seconds * cfg.action_hz)) - 1
                # Toward the eye that still sees something better; a coin toss
                # when the drop was met head-on and both eyes agree.
                side = float(np.sign(v[0] - v[1]))
                self._turn_sign = side if side else float(self.rng.choice((-1.0, 1.0)))
                self._valence_trace.clear()
                turn, mid = self._turn_sign, 1.0
            self._valence_trace.append(mean)
        turn = float(np.clip(turn, -1.0, 1.0))
        half = min(mid - SIGNAL_LOW, SIGNAL_HIGH - mid)
        return np.array([mid - half * turn, mid + half * turn])

    @property
    def stimulus_names(self) -> tuple[str, ...]:
        """The two stimuli trials are named by: odours A/B, or the colours."""
        return tuple(self.config.stimuli) if self.config.modality == "colour" else ODOUR_NAMES

    # --- trials -----------------------------------------------------------

    def reset_memory(self) -> None:
        """Forget everything. Use between subjects, never between trials."""
        self.mb.reset()

    def _reset_body(self) -> None:
        self.walker.reset(
            seed=int(self.rng.integers(0, 2**31 - 1)),
            warmup_s=self.config.warmup_seconds,
        )

    def run_trial(self, kind: str, odour: str | None, *, block: str = "") -> TrialResult:
        """Run one trial to completion. See `trial_steps`."""
        steps = self.trial_steps(kind, odour, block=block)
        while True:
            try:
                next(steps)
            except StopIteration as finished:
                return finished.value

    def _omission(self, reinforced: str | None):
        """The MBON -> DAN feedback for this trial, or None.

        If the mushroom body predicts punishment for what the fly is smelling
        -- negative valence -- and none comes, the size of that prediction
        drives dopamine into `config.extinction`. Wired to "avoid", this
        depresses the avoid compartment for the odour that was feared, which
        cancels the valence without touching the depressed approach synapses:
        the original memory stays and a second, opposing one forms beside it.
        That is the picture Felsenberg et al. (2018) give for extinction, where
        the omission of punishment is remembered as a positive experience and
        the two memories are combined downstream.

        Off for punished trials. The model has no timing -- no eligibility
        trace, no expected time of arrival for the shock -- so "no punishment
        yet" a second into a trial where it will come could not be told apart
        from "no punishment at all". The prediction is resolved per trial
        instead: a trial that punishes confirms it, one that does not refutes it.
        """
        if self.config.extinction is None or reinforced is not None:
            return None
        target, gain = self.config.extinction, self.config.extinction_gain
        return lambda mbon, valence: {target: gain * max(0.0, -valence)}

    def trial_steps(self, kind: str, odour: str | None, *, block: str = ""):
        """One trial as a generator: a yield per 100 Hz action step.

        This is the only copy of the trial loop. `run_trial` drains it for
        batch runs; the web viewer advances it a step at a time so it can
        render and take commands in between. The viewer used to carry its own
        copy of this loop, and the first change made here -- steering reading
        per-odour valences instead of the mixture's -- broke it with a
        TypeError nobody saw until the page was opened. One loop, two consumers.

        Physics resets at the start; the mushroom body's memory does not.

        Args:
            kind: One of `TRIAL_KINDS`. "train" presents `odour` and punishes
                it, "expose" presents `odour` and does not, "test" presents
                both without punishment.
            odour: The odour presented on a "train" or "expose" trial. Ignored
                on test trials -- a test is always unreinforced, or it would be
                another training trial with extra steps.
            block: Protocol block to record on the result.

        Yields:
            ``(MushroomBodyState, concentration)`` after each action step.

        Returns:
            The `TrialResult`, as ``StopIteration.value``.

        A colour experiment runs `_colour_trial_steps` from here instead: its
        stimulus, reinforcement schedule and score differ throughout, and
        keeping the odour loop as it was is what keeps odour results
        bit-identical. Both consumers still come through this one entry.
        """
        if self.config.modality == "colour":
            return (yield from self._colour_trial_steps(kind, odour, block))
        if kind not in TRIAL_KINDS:
            raise ValueError(f"kind must be one of {TRIAL_KINDS}; got {kind!r}")
        if kind != "test" and odour not in ODOUR_NAMES:
            raise ValueError(f"a {kind!r} trial needs an odour; got {odour!r}")
        self._reset_body()

        sources: dict[str, np.ndarray] = {}
        half = np.deg2rad(self.config.bearing_deg)
        if kind == "test":
            # Randomise which side each odour is on, every trial. Fixing it
            # lets the spawn heading and any turning bias masquerade as a
            # learned preference.
            flip = bool(self.rng.integers(2))
            sources["A"] = self._place(0, -half if flip else +half)
            sources["B"] = self._place(1, +half if flip else -half)
            present = ("A", "B")
            reinforced = None
        else:
            index = ODOUR_NAMES.index(odour)
            # Off-axis, at a random bearing inside the same cone the test uses.
            # A head-on source would train the fly with no steering signal
            # available, and a fly that had learned avoidance would have no way
            # to show it.
            sources[odour] = self._place(index, float(self.rng.uniform(-half, half)))
            self._park(1 - index)
            present = (odour,)
            reinforced = odour if kind == "train" else None
        self._sync_markers()

        time_near = {name: 0.0 for name in present}
        dopamine_seconds = 0.0
        omission_dopamine = 0.0
        feedback = self._omission(reinforced)
        target = self.config.extinction
        path: list[np.ndarray] = []
        inside_seconds = 0.0

        for _ in range(self.steps_per_trial):
            concentration = self._concentration()
            intensity = float(concentration.sum())
            inside = intensity >= self.config.us_threshold
            inside_seconds = inside_seconds + self.dt if inside else 0.0

            dan: dict[str, float] = {}
            if reinforced is not None and inside_seconds > self.config.us_delay:
                # Punishment depresses the approach compartment, so the paired
                # odour drives approach less. Avoidance is the absence of
                # approach, not a separate excitatory drive.
                dan["approach"] = 1.0
                dopamine_seconds += self.dt

            state = self.mb.step(concentration, dan, self.dt, feedback=feedback)
            if feedback is not None:
                omission_dopamine += state.dan.get(target, 0.0) * self.dt
            self.last_state = state
            self.walker.descending_signal = self._steer()
            for _ in range(self.steps_per_action):
                self.walker.physics_step()

            pos = self.fs.thorax_pos()[:2]
            path.append(pos.copy())
            for name, source_xy in sources.items():
                if np.linalg.norm(pos - source_xy) <= self.config.near_radius:
                    time_near[name] += self.dt

            yield state, concentration

        preference = None
        if kind == "test":
            preference = self._preference(np.asarray(path), sources)

        self._trial_count += 1
        self._last_kind = kind
        return TrialResult(
            kind=kind,
            index=self._trial_count,
            cs_plus=reinforced,
            odours=present,
            time_near=time_near,
            preference=preference,
            valence={n: self.valence_of(n) for n in ODOUR_NAMES},
            mbon={n: dict(self.read(n).mbon) for n in ODOUR_NAMES},
            dopamine_seconds=dopamine_seconds,
            path=np.asarray(path),
            sources=sources,
            omission_dopamine=omission_dopamine,
            block=block,
        )

    def _colour_trial_steps(self, kind: str, stimulus: str | None, block: str):
        """`trial_steps` for the colour task (Vogt et al. 2014).

        "train" paints the whole floor `stimulus` and delivers dopamine pulses,
        "expose" paints it without them, "test" lays a checkerboard of both
        colours. `TrialResult.time_near` holds seconds on each colour and
        `preference` is ``(t[second] - t[first]) / (t[first] + t[second])`` over
        the scored part of a test -- PI_G for blue and green.
        """
        cfg = self.config
        names = tuple(cfg.stimuli)
        if kind not in TRIAL_KINDS:
            raise ValueError(f"kind must be one of {TRIAL_KINDS}; got {kind!r}")
        if kind != "test" and stimulus not in names:
            raise ValueError(f"a {kind!r} trial needs one of {names}; got {stimulus!r}")

        # The dark interval before this trial: nothing is seen, so recovery is
        # the only thing that happens to the synapses.
        dark = cfg.iti_seconds if self._trial_count > 0 else 0.0
        if kind == "test" and self._last_kind in ("train", "expose"):
            dark += cfg.retention_seconds
        if dark > 0.0:
            self.mb.idle(dark)

        self._reset_body()
        floor = self.fs.floor
        floor.follow_to((0.0, 0.0))
        if kind == "test":
            # The fly spawns on a corner facing along an edge, so this decides
            # which colour starts on its left. Randomised every trial.
            flip = bool(self.rng.integers(2))
            floor.paint_checker(*(names[::-1] if flip else names))
            present, reinforced = names, None
            seconds = cfg.test_seconds if cfg.test_seconds is not None else cfg.trial_seconds
        else:
            floor.paint_uniform(stimulus)
            present = (stimulus,)
            reinforced = stimulus if kind == "train" else None
            seconds = cfg.trial_seconds
        mj.mj_forward(self.fs.sim.mj_model, self.fs.sim.mj_data)

        target = "approach" if cfg.reinforcer == "punish" else "avoid"
        dan_blocked = f"{cfg.reinforcer}_dan" in cfg.blocked
        steps = int(round(seconds * cfg.action_hz))
        look_every = max(1, int(round(cfg.action_hz / cfg.vision_hz)))
        period = int(round(cfg.us_period * cfg.action_hz))
        pulse = int(round(cfg.us_pulse * cfg.action_hz))
        scored_from = int(steps * cfg.score_from)
        time_on = {name: 0.0 for name in present}
        dopamine_seconds = omission_dopamine = 0.0
        feedback = self._omission(reinforced)
        self._valence_trace = deque(maxlen=max(1, int(round(0.3 * cfg.action_hz))))
        self._turn_steps_left = 0
        path: list[np.ndarray] = []
        nothing = np.zeros(2)
        readouts = None

        for i in range(steps):
            if i % look_every == 0:
                readouts = self.fs.sim.get_ommatidia_readouts(self.fs.name)
            dan: dict[str, float] = {}
            if reinforced is not None and i % period >= period - pulse and not dan_blocked:
                dan[target] = 1.0
                dopamine_seconds += self.dt

            state = self.mb.step(nothing, dan, self.dt, feedback=feedback, readouts=readouts)
            if feedback is not None:
                omission_dopamine += state.dan.get(cfg.extinction, 0.0) * self.dt
            self.last_state = state
            self.walker.descending_signal = self._visual_command(state)
            for _ in range(self.steps_per_action):
                self.walker.physics_step()

            pos = self.fs.thorax_pos()[:2]
            path.append(pos.copy())
            floor.follow(pos)
            if i >= scored_from:
                colour = floor.colour_at(pos)
                if colour in time_on:
                    time_on[colour] += self.dt

            yield state, nothing

        preference = None
        if kind == "test":
            total = time_on[names[0]] + time_on[names[1]]
            preference = (time_on[names[1]] - time_on[names[0]]) / total if total > 0 else 0.0

        self._trial_count += 1
        self._last_kind = kind
        return TrialResult(
            kind=kind,
            index=self._trial_count,
            cs_plus=reinforced,
            odours=present,
            time_near=time_on,
            preference=preference,
            valence={n: self.valence_of(n) for n in names},
            mbon={n: dict(self.read(n).mbon) for n in names},
            dopamine_seconds=dopamine_seconds,
            path=np.asarray(path),
            sources={},
            omission_dopamine=omission_dopamine,
            block=block,
            floor=tuple(floor.pattern),
        )

    # --- memory on disk ----------------------------------------------------

    def save_memory(self, path, *, name: str = "", note: str = "", source: str = "") -> None:
        """Write what this fly has learned: every compartment's KC->MBON weights.

        The Kenyon-cell wiring is not saved -- it is a pure function of the
        wiring seed, which is. So a saved fly is its synapses plus enough to
        rebuild the rest identically.

        The fly's configuration goes in too. Weights alone are not a fly: the
        same synapses under a different `extinction` setting behave differently
        on the very next unpunished trial, so a memory restored without its
        config would quietly be someone else's. A summary of the valence each
        odour carries is stored alongside, so a library of memories can be
        listed without rebuilding a simulation for each one.
        """
        meta = {
            "format": 2,
            "name": name,
            "note": note,
            "source": source,
            "created": datetime.now().isoformat(timespec="seconds"),
            "config": asdict(self.config),
            "summary": {
                odour: {"valence": state.valence, "mbon": dict(state.mbon)}
                for odour, state in ((o, self.read(o)) for o in self.stimulus_names)
            },
        }
        visual = self.mb.visual
        np.savez_compressed(
            path,
            wiring_seed=self.mb.front_end.seed,
            n_kc=self.mb.n_kc,
            n_olfactory=self.mb.n_olfactory,
            n_visual=self.mb.n_visual,
            visual_params=np.array(json.dumps(visual.params if visual is not None else None)),
            compartments=np.array([c.name for c in self.mb.compartments]),
            weights=np.stack([c.weights for c in self.mb.compartments]),
            trials=self._trial_count,
            meta=np.array(json.dumps(meta)),
        )

    def load_memory(self, path) -> None:
        """Restore weights written by `save_memory` into this fly.

        Refuses a file from different wiring: the weights index Kenyon cells,
        and loaded onto another wiring they would be attached to the wrong
        cells -- a fly that "remembers" a random pattern and raises nothing.

        A memory saved before vision existed (or by a fly without it) loads
        into a seeing fly with every visual synapse at rest: a fly that has
        smelled things but never learned anything about what it saw. The other
        way round is refused, because it would silently drop what was learned.
        """
        data = np.load(path)
        if int(data["wiring_seed"]) != self.mb.front_end.seed:
            raise ValueError(
                f"{path} was learned on wiring seed {int(data['wiring_seed'])}, "
                f"this fly has {self.mb.front_end.seed}. Build the experiment "
                f"with wiring_seed={int(data['wiring_seed'])}."
            )
        names = [str(n) for n in data["compartments"]]
        if names != [c.name for c in self.mb.compartments]:
            raise ValueError(f"{path} has compartments {names}, this fly has "
                             f"{[c.name for c in self.mb.compartments]}.")
        n_visual = int(data["n_visual"]) if "n_visual" in data.files else 0
        if n_visual:
            if self.mb.visual is None:
                raise ValueError(f"{path} holds visual synapses and this fly has no "
                                 f"vision; loading it would discard them.")
            stored = json.loads(str(data["visual_params"]))
            if stored != self.mb.visual.params:
                raise ValueError(f"{path} was learned on visual wiring {stored}, "
                                 f"this fly has {self.mb.visual.params}.")
        n_olfactory = self.mb.n_olfactory
        for compartment, weights in zip(self.mb.compartments, data["weights"]):
            compartment.weights[:n_olfactory] = weights[:n_olfactory]
            if n_visual:
                compartment.weights[n_olfactory:] = weights[n_olfactory:]
            else:
                compartment.weights[n_olfactory:] = compartment.w0
        #: "learned" if the file carried visual synapses, "naive" if they were
        #: set to rest on loading, "" for a fly without vision.
        self.memory_visual = (
            "" if self.mb.visual is None else "learned" if n_visual else "naive"
        )
        self._trial_count = int(data["trials"])
        #: Name of the memory this fly was restored from, for display.
        self.memory_name = read_memory(path).get("name", "")

    def _preference(self, path: np.ndarray, sources: dict[str, np.ndarray]) -> float:
        """Which source the fly committed to, from -1 (all A) to +1 (all B).

        Scored from position rather than from dwell time. In an open arena a
        fly that chooses a source still walks straight past it -- measured, it
        covers 12 mm/s and crosses a source in about 1.5 s -- so time spent
        within a radius mostly measures walking speed. The distance ratio is
        the T-maze readout, which arm did it end up in, and speed cannot
        confound it.
        """
        start = int(len(path) * self.config.score_from)
        tail = path[start:]
        if tail.size == 0:
            return 0.0
        d_a = np.linalg.norm(tail - sources["A"], axis=1)
        d_b = np.linalg.norm(tail - sources["B"], axis=1)
        return float(np.mean((d_a - d_b) / np.maximum(d_a + d_b, 1e-9)))

    # --- probing ----------------------------------------------------------

    def read(self, odour: str, concentration: float = 0.05):
        """Read the mushroom body for one stimulus without changing it.

        `odour` is an odour name, or a colour name in a colour experiment, which
        is read from the floor as seen from the spawn pose.
        """
        if self.config.modality == "colour":
            return self.mb.read(np.zeros(2), self._unit_readouts[odour])
        vector = ODOUR_A if odour == "A" else ODOUR_B
        return self.mb.read(vector * concentration)

    def valence_of(self, odour: str, concentration: float = 0.05) -> float:
        return self.read(odour, concentration).valence

    def close(self) -> None:
        self.fs.close()


#: Protocol blocks in order: (block key, trial kind, which CS+ it reinforces).
#: "cs+" and "other" are resolved against the session's CS+ in `session_plan`.
PROTOCOL = (
    ("naive", "test", None),
    ("train", "train", "cs+"),
    ("acquisition", "test", None),
    ("reversal_train", "train", "other"),
    ("reversal", "test", None),
)


def session_plan(
    n_train: int = 20,
    n_test: int = 6,
    cs_plus: str = "A",
    reversal: bool = True,
    differential: bool = False,
    names: tuple[str, ...] = ODOUR_NAMES,
    naive: bool = True,
) -> list[tuple[str, str, str | None]]:
    """The whole session as a flat list of ``(block, kind, odour)``.

    Flat because the web viewer advances one trial at a time and cannot drive
    a blocking loop; `run_session` walks the same list, so the batch run and
    the viewer cannot disagree about what the protocol is. `odour` is the odour
    a "train" or "expose" trial presents, and None on a test trial.

    The leading test block is what makes the rest interpretable: it measures
    whether this fly had a preference before anything was taught to it. A
    session whose naive block is not near zero has a layout or steering bug,
    and its learning curve cannot be read.

    With `differential`, a training block of `n_train` trials alternates the
    punished odour with the other one presented unpunished, CS+ first. The
    block keeps its length, so the time passive forgetting has to act is the
    same as without -- which is what lets a differential run be compared with
    a plain one and the difference put down to the CS- presentations alone.

    `names` are the two stimuli -- odours A and B, or a colour experiment's
    `stimulus_names`. `naive=False` drops the leading test block, as the colour
    assay does: Vogt et al. (2014) cancel any innate colour bias by training two
    groups with opposite contingencies instead.
    """
    if cs_plus not in names:
        raise ValueError(f"cs_plus must be one of {names}; got {cs_plus!r}")
    other = names[1] if cs_plus == names[0] else names[0]
    plan: list[tuple[str, str, str | None]] = []
    for block, kind, reinforced in PROTOCOL:
        if block.startswith("reversal") and not reversal:
            continue
        if block == "naive" and not naive:
            continue
        if kind != "train":
            plan.extend([(block, kind, None)] * n_test)
            continue
        punished = {"cs+": cs_plus, "other": other}[reinforced]
        unpunished = other if punished == cs_plus else cs_plus
        for i in range(n_train):
            if differential and i % 2 == 1:
                plan.append((block, "expose", unpunished))
            else:
                plan.append((block, "train", punished))
    return plan


def run_session(
    experiment: ConditioningExperiment,
    *,
    n_train: int = 20,
    n_test: int = 6,
    cs_plus: str = "A",
    reversal: bool = True,
    naive: bool = True,
    on_trial=None,
    memory_dir=None,
) -> list[TrialResult]:
    """Naive test, train, test, then reverse the contingency and repeat.

    See `session_plan` for the schedule; whether training is differential comes
    from ``experiment.config.differential``.

    Args:
        memory_dir: If given, the fly's synapses are saved there at the end of
            every block, as ``after_<block>.npz``. The trial stream records what
            the fly *did*; these record what it *knows*, and are what
            `ConditioningExperiment.load_memory` restores.
    """
    plan = session_plan(
        n_train, n_test, cs_plus, reversal, experiment.config.differential,
        experiment.stimulus_names, naive,
    )
    results: list[TrialResult] = []
    for i, (block, kind, odour) in enumerate(plan):
        result = experiment.run_trial(kind, odour, block=block)
        results.append(result)
        if on_trial is not None:
            on_trial(result)
        block_ends = i + 1 == len(plan) or plan[i + 1][0] != block
        if memory_dir is not None and block_ends:
            experiment.save_memory(Path(memory_dir) / f"after_{block}.npz")
    return results


def colour_config(**overrides) -> ConditioningConfig:
    """The colour assay of Vogt et al. (2014), as a config.

    Whole-floor CS periods of 60 s, twelve 1 s dopamine pulses in each CS+
    period, 12 s of darkness between trials, differential training, and a 90 s
    test 60 s after training, scored from 1 s as their 1-90 s average was.
    The depression rate is the odour experiment's, unchanged: the visual code
    is scaled so an active visual cell learns at the rate an olfactory one does
    (`flyplay.mushroom_body`).

    **Forgetting is not the odour experiment's.** Its 100 s recovery time
    constant was a compromise for reversal inside a session of 5 s trials. This
    protocol runs for ten minutes, and with that constant the memory was gone
    before it was tested -- measured on the first sweep (seed 0, CS+ blue):
    blue valence -0.92 at the end of every CS+ period (one period saturates it,
    as Vogt found one trial enough), -0.45 after the next dark interval and CS-
    period, about -0.22 when the test began and -0.09 when it ended, and the
    flies' preference toward their CS+ came out at -0.14, +0.01 and -0.06. Fly
    aversive memory lasts hours, not minutes (memory-phase mutants lose it
    within 30-60 min, wild type is tested at 3-24 h; review:
    https://pmc.ncbi.nlm.nih.gov/articles/PMC2692900/), so the colour task
    forgets on that scale: a one-hour time constant. Chosen from that, not from
    the learning index. Those three flies are kept in out/mb_colour_tau100/.

    **Steering was chosen by a ceiling test before any training**: flies given
    a hand-set memory -- the CS+ colour's approach synapses at half, normalised
    valence -0.5 -- ran one 30 s checker test each, 4 per contingency, and
    LI = [PI_G(G+) - PI_G(B+)]/2 was scored (negative = avoids the CS+):

        no colour steering              +0.00 +/- 0.01
        bilateral, gain 3               -0.56 +/- 0.05   <- used
        bilateral, gain 10              -0.74 +/- 0.06
        kinesis 0.6 alone               +0.02 +/- 0.06
        bilateral 3 + kinesis 0.6       -0.32 +/- 0.05
        border turn 0.8 s alone         -0.18 +/- 0.03
        bilateral 3 + border turn       -0.61 +/- 0.04

    Comparing the eyes does nearly all of it. Kinesis costs more than it gives,
    because walking faster takes turning authority away (`KINESIS_RANGE`), and
    border turns add nothing beyond the error bars. Gain 3 over 10 because 3
    saturates the turn only at a 0.33 difference between the eyes, so how
    strongly the fly avoids still scales with how deep the memory is; 10 is
    close to bang-bang.
    """
    values: dict[str, Any] = dict(
        modality="colour",
        trial_seconds=60.0,
        test_seconds=90.0,
        iti_seconds=12.0,
        retention_seconds=60.0,
        differential=True,
        score_from=1.0 / 90.0,
        visual_gain=3.0,
        recovery=1.0 / 3600.0,
    )
    values.update(overrides)
    return ConditioningConfig(**values)


def preference_index(results: list[TrialResult]) -> float:
    """Mean preference over the test trials in a list. NaN if there are none."""
    scores = [r.preference for r in results if r.preference is not None]
    return float(np.mean(scores)) if scores else float("nan")


# --- memories ---------------------------------------------------------------


def read_memory(path) -> dict:
    """What a saved memory is, without building a fly to hold it.

    Returns the stored metadata -- name, note, source, config, per-odour
    summary -- plus the wiring seed and trial count. Snapshots written before
    metadata existed come back with ``format`` 0, an empty name and ``config``
    None; they still load, but whoever restores one has to supply the config.
    """
    with np.load(path) as data:
        info = json.loads(str(data["meta"])) if "meta" in data.files else {
            "format": 0, "name": "", "note": "", "source": "", "created": "",
            "config": None, "summary": None,
        }
        info["wiring_seed"] = int(data["wiring_seed"])
        info["trials"] = int(data["trials"])
        info["compartments"] = [str(n) for n in data["compartments"]]
        info["n_visual"] = int(data["n_visual"]) if "n_visual" in data.files else 0
    return info


def config_from_dict(values: dict | None) -> ConditioningConfig:
    """A `ConditioningConfig` from a stored dict, tolerant of version drift.

    Keys the dataclass no longer has are dropped and keys it has gained take
    their defaults, so a memory saved by an older version of this module still
    restores -- with the new defaults for anything it predates.
    """
    known = {f.name for f in fields(ConditioningConfig)}
    return ConditioningConfig(**{k: v for k, v in (values or {}).items() if k in known})


def experiment_from_memory(
    path, *, seed: int = 0, config: ConditioningConfig | None = None
) -> ConditioningExperiment:
    """Build a fly and restore a saved memory into it.

    The wiring comes from the file, never from `seed` -- weights attached to
    another wiring would sit on the wrong Kenyon cells. `seed` only sets the
    body's trial-to-trial randomness (spawn pose, source bearings). The config
    is the one saved with the memory unless `config` overrides it.
    """
    info = read_memory(path)
    if config is None:
        if info["config"] is None:
            raise ValueError(
                f"{path} predates stored configs; pass config= explicitly."
            )
        config = config_from_dict(info["config"])
    experiment = ConditioningExperiment(config, seed=seed, wiring_seed=info["wiring_seed"])
    experiment.load_memory(path)
    return experiment

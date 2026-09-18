"""Kenyon cell -> MBON synapses, and the dopamine that depresses them.

The mushroom body is not a controller. It is an associative memory: it learns
which odour predicts which outcome, and reports that as a valence. The steering
is somebody else's job -- here, the hand-written chemotaxis rule in
`flyplay.conditioning`, whose sign this module supplies.

Three properties of the real circuit shape the implementation, and leaving any
of them out produces something that still runs but is no longer a mushroom
body:

**Dopamine depresses.** ``dw = -eta * KC * DAN``. The synapses that weaken are
the ones active *at the moment* dopamine arrives. So an odour paired with
punishment drives its approach-MBON less than it used to -- avoidance is the
absence of approach, not a separate excitatory pathway.

**Compartments are independent.** A dopaminergic neuron innervates one
compartment and changes only the KC->MBON synapses inside it. This is the
sharpest difference from the scalar TD error the rest of this project uses: one
global number cannot teach "avoid A" and "approach B" at the same time, and
`scripts/15_analyze_mb.py` runs exactly that ablation.

**Forgetting is not optional.** A depression-only rule is a ratchet: weights
walk to zero and the fly stops responding to anything. Real mushroom bodies
have dopamine-independent recovery, and without the ``lam * (w0 - w)`` term
here the circuit dies within a few trials. That term is what makes reversal
learning possible at all.

Measured at the defaults, on synthetic 8 s trials with dopamine covering the
second half (`scripts/13_mb_check.py` reruns all of it):

    20 CS+ trials      MBON_approach for the paired odour     -69.8%
                       ... for the unpaired odour              -1.3%
                       ... the other compartment                0.0%
    reversal           valence crosses over by trial 5
    recovery           back to 99% of rest after 120 s of no odour
    shared dopamine    valence separation exactly 0.000

That last number is not a near-miss, it is an identity: with one scalar signal
both compartments see the same input and the same Kenyon cells, so their
weights can never diverge, whatever the contingency. Compartments are what make
opposite valences representable at all.

Innate valence lives outside this module. A novel odour reads exactly zero here
-- the mushroom body reports what was *learned*, and the naive attraction a fly
shows to an unfamiliar smell comes from the lateral horn. `flyplay.conditioning`
adds that term where it belongs, in the steering law.

**Vision shares the compartments.** Visual Kenyon cells (`flyplay.visual_pathway`)
are appended after the olfactory ones and feed the same MBONs through the same
dopamine: in the fly the same PPL1 neurons are needed for colour and odour
memories (Vogt et al. 2014), and visual and olfactory Kenyon cells share DANs
and MBONs (Ganguly et al. 2024). Each sense's code sums to 1 on its own, which
would hand 5 visual cells the weight of 100 olfactory ones -- per cell twenty
times the drive and twenty times the plasticity. The visual code is therefore
scaled by ``k_visual / k_olfactory``, so every active cell carries about 1/100
whichever sense drove it, and vision's share of the MBON input follows from
the cell counts. Without vision the arrays are the olfactory ones, untouched.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from flyplay.olfactory import N_KC, OlfactoryFrontEnd, OlfactoryState

if TYPE_CHECKING:
    from flyplay.visual_pathway import VisualFrontEnd, VisualState

#: Depression rate, per unit of dopamine per second. With the Kenyon-cell code
#: normalised to sum to 1, one second of full dopamine removes `eta` from the
#: total weight carried by the active cells. 3.0 was picked by measurement:
#: 1.0 only reaches 29% depression after 20 trials, 10.0 saturates at 60% in a
#: single trial and leaves no learning curve to look at.
DEFAULT_ETA = 3.0
#: Recovery rate back toward the resting weight, per second. Roughly a 30 s
#: time constant: slow enough that a memory survives the trials that follow it,
#: fast enough that reversal is not fighting a saturated synapse. Dropping it
#: to 0.01 deepens the memory to 93% but makes reversal take three times as
#: long, which is the trade this parameter *is*.
DEFAULT_RECOVERY = 0.033


@dataclass
class Compartment:
    """One mushroom-body compartment: a KC->MBON weight vector and its DAN.

    Attributes:
        name: Compartment label, used in logs and by `MushroomBody.step`.
        sign: How this compartment's MBON contributes to valence. +1 for an
            approach-driving MBON, -1 for an avoidance-driving one.
        n_kc: Size of the Kenyon-cell population feeding it.
        w0: Resting weight, identical across Kenyon cells. Uniform on purpose:
            any structure here would give one odour a head start and the naive
            preference index would not start at zero.
        eta: Depression rate.
        recovery: Rate of return to `w0`.
        w_min: Floor on a synapse. Zero -- a synapse cannot go negative, and
            without the floor the depression term can drive weights past it.
        cells: The Kenyon cells whose axons run through this compartment, as a
            boolean mask, or None for all of them. A gamma1pedc MBON reads
            gamma cells only; the others' synapses stay at `w0` and count for
            nothing.
        coverage: The share of a code these cells carry (the lobes' share of
            the population, when the front end fixes it). Input is divided by
            it, so a novel odour still reads `w0` and full depression still
            reads 0: the MBON's gain matches the part of the mushroom body it
            sees.
    """

    name: str
    sign: float
    n_kc: int = N_KC
    w0: float = 1.0
    eta: float = DEFAULT_ETA
    recovery: float = DEFAULT_RECOVERY
    w_min: float = 0.0
    cells: np.ndarray | None = None
    coverage: float = 1.0
    weights: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.weights = np.full(self.n_kc, float(self.w0))
        if self.cells is not None:
            self.cells = np.asarray(self.cells, dtype=bool)
            if self.cells.shape != (self.n_kc,):
                raise ValueError(f"cells mask has shape {self.cells.shape}, expected ({self.n_kc},)")

    def reset(self) -> None:
        self.weights[:] = self.w0

    def _input(self, kc: np.ndarray) -> np.ndarray:
        if self.cells is None:
            return kc
        return np.where(self.cells, kc, 0.0) / self.coverage

    def response(self, kc: np.ndarray) -> float:
        """MBON output for a Kenyon-cell pattern.

        The code sums to 1, so this is the mean resting weight of the cells
        that odour happens to activate -- `w0` for a novel odour, less for one
        that has been paired with dopamine.
        """
        return float(self.weights @ self._input(kc))

    def below_rest(self, kc: np.ndarray) -> float:
        """How far a pattern's synapses sit below rest here: 0 for anything
        never paired with this compartment's dopamine."""
        return float((self.w0 - self.weights) @ self._input(kc))

    def learn(self, kc: np.ndarray, dan: float, dt: float) -> None:
        """Apply one step of depression and recovery.

        Both terms run every step. Recovery is not a separate "forgetting
        phase": a synapse is always being pulled back toward rest, and a
        memory is what survives that pull.
        """
        if dan > 0.0:
            self.weights -= self.eta * dan * self._input(kc) * dt
        self.weights += self.recovery * (self.w0 - self.weights) * dt
        np.clip(self.weights, self.w_min, None, out=self.weights)

    def depression(self, kc: np.ndarray) -> float:
        """How far this odour's synapses have been driven below rest, 0 to 1."""
        return float(1.0 - self.response(kc) / self.w0) if self.w0 else 0.0


@dataclass(frozen=True)
class MushroomBodyState:
    """One readout, kept for plotting and for the web viewer."""

    olfactory: OlfactoryState
    #: MBON output per compartment, keyed by name.
    mbon: dict[str, float]
    #: Dopamine input per compartment at this step.
    dan: dict[str, float]
    #: Signed sum of the MBON outputs. >0 approach, <0 avoid.
    valence: float
    #: The visual front end's pass, both eyes, or None without vision.
    visual: VisualState | None = None
    #: The whole Kenyon-cell vector the synapses see -- olfactory cells, then
    #: scaled visual ones. None without vision, where it is `kc`.
    kc_all: np.ndarray | None = None

    @property
    def kc(self) -> np.ndarray:
        """The olfactory Kenyon cells only."""
        return self.olfactory.kc

    @property
    def smelling(self) -> bool:
        return bool(self.olfactory.kc.any())

    @property
    def seeing(self) -> bool:
        return self.visual is not None and bool(self.visual.kc.any())


class MushroomBody:
    """An olfactory front end plus a set of independently plastic compartments.

    Args:
        front_end: The `OlfactoryFrontEnd` supplying Kenyon-cell codes.
        compartments: Compartments sharing that population. The default pair --
            one approach-driving, one avoidance-driving -- is the smallest set
            that can represent opposite valences.
        shared_dan: Route every compartment's dopamine from one global scalar
            instead of per-compartment inputs. This is the ablation: it is what
            a single TD error would give you, and it should fail to learn two
            opposite contingencies at once.
        visual: A `flyplay.visual_pathway.VisualFrontEnd` whose Kenyon cells
            join the population, or None for a smell-only mushroom body.
    """

    def __init__(
        self,
        front_end: OlfactoryFrontEnd,
        compartments: list[Compartment] | None = None,
        *,
        shared_dan: bool = False,
        visual: VisualFrontEnd | None = None,
    ):
        self.front_end = front_end
        self.visual = visual
        self.n_olfactory = front_end.n_kc
        self.n_visual = visual.n_kc if visual is not None else 0
        #: Kenyon cells the compartments see: olfactory first, then visual.
        self.n_kc = self.n_olfactory + self.n_visual
        #: Per-cell scale of the visual code. See the module docstring.
        self.visual_scale = visual.k / front_end.k if visual is not None else 0.0
        #: Ablation: visual Kenyon-cell output blocked, as with shibire in the
        #: gamma-d cells (Vogt et al. 2016). The cells still compute a code --
        #: the viewer can show it -- but it reaches no MBON and changes no
        #: synapse.
        self.visual_blocked = False
        if compartments is None:
            compartments = [
                Compartment("approach", sign=+1.0, n_kc=self.n_kc),
                Compartment("avoid", sign=-1.0, n_kc=self.n_kc),
            ]
        for compartment in compartments:
            if compartment.n_kc != self.n_kc:
                raise ValueError(
                    f"Compartment '{compartment.name}' expects {compartment.n_kc} "
                    f"Kenyon cells but the population has {self.n_kc} "
                    f"({self.n_olfactory} olfactory + {self.n_visual} visual)."
                )
        self.compartments = list(compartments)
        self.shared_dan = bool(shared_dan)

    def __getitem__(self, name: str) -> Compartment:
        for compartment in self.compartments:
            if compartment.name == name:
                return compartment
        raise KeyError(f"No compartment named {name!r}.")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.compartments)

    def reset(self) -> None:
        """Forget everything. Between subjects, never between trials."""
        for compartment in self.compartments:
            compartment.reset()

    def wiring_signature(self) -> str:
        """A fingerprint of everything fixed between the senses and the synapses.

        Two mushroom bodies with the same signature give every Kenyon cell the
        same meaning, so synapses learned on one mean the same on the other.
        Saved weights carry it (`flyplay.session`): loaded onto a front end that
        codes smells or colours differently, they would sit on the wrong cells
        and nothing would complain.
        """
        front = self.front_end
        digest = hashlib.sha256(json.dumps({
            "olfactory": [front.n_dimensions, front.n_receptors, front.n_kc, front.k,
                          front.hill_k, front.sigma, front.min_drive],
            "visual": self.visual.params if self.visual is not None else None,
        }, sort_keys=True).encode())
        arrays = [front.affinity, front.projection]
        if self.visual is not None:
            arrays += [self.visual.patches, self.visual.projection]
        arrays += [c.cells for c in self.compartments if c.cells is not None]
        for array in arrays:
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()[:16]

    # --- readout and learning -------------------------------------------

    def embed(
        self, olfactory_kc: np.ndarray | None = None, visual_kc: np.ndarray | None = None
    ) -> np.ndarray:
        """One vector over the whole population from per-sense codes.

        Without vision this is the olfactory code itself, not a copy, so a
        smell-only fly computes exactly what it did before vision existed.
        """
        if self.visual is None:
            return olfactory_kc if olfactory_kc is not None else np.zeros(self.n_kc)
        kc = np.zeros(self.n_kc)
        if olfactory_kc is not None:
            kc[: self.n_olfactory] = olfactory_kc
        if visual_kc is not None and not self.visual_blocked:
            kc[self.n_olfactory :] = self.visual_scale * visual_kc
        return kc

    def _encode(self, concentration, readouts):
        olfactory = self.front_end(np.asarray(concentration, dtype=float))
        if self.visual is None:
            return olfactory, None, olfactory.kc
        visual = self.visual(readouts) if readouts is not None else None
        kc = self.embed(olfactory.kc, visual.mean_kc if visual is not None else None)
        return olfactory, visual, kc

    def valence_from(self, kc: np.ndarray) -> float:
        """Valence of a population vector, as `embed` builds them. Read-only."""
        return float(sum(c.sign * c.response(kc) for c in self.compartments))

    def eye_valences(self, visual: VisualState) -> np.ndarray:
        """Learned valence of what each eye sees, ``(left, right)``.

        Read-only, and from visual cells alone: the steering compares the two
        eyes, and anything smelled is the same for both.
        """
        if self.visual is None or self.visual_blocked:
            return np.zeros(2)
        signed = sum(c.sign * np.where(c.cells, c.weights, 0.0)[self.n_olfactory :] / c.coverage
                     if c.cells is not None else c.sign * c.weights[self.n_olfactory :]
                     for c in self.compartments)
        return self.visual_scale * (visual.kc @ signed)

    def read(self, concentration: np.ndarray, readouts: np.ndarray | None = None) -> MushroomBodyState:
        """Encode what is sensed and read every MBON, without changing anything."""
        olfactory, visual, kc = self._encode(concentration, readouts)
        mbon = {c.name: c.response(kc) for c in self.compartments}
        valence = sum(c.sign * mbon[c.name] for c in self.compartments)
        return MushroomBodyState(olfactory, mbon, {c.name: 0.0 for c in self.compartments},
                                 float(valence), visual,
                                 kc if self.visual is not None else None)

    def step(
        self,
        concentration: np.ndarray,
        dan: dict[str, float],
        dt: float,
        feedback=None,
        readouts: np.ndarray | None = None,
    ) -> MushroomBodyState:
        """Read the MBONs, then apply plasticity for this timestep.

        The read happens first so the returned state is the circuit's output
        *for* this step, not the consequence of learning during it.

        Args:
            concentration: Odour concentration vector at the antennae.
            dan: Dopamine per compartment name. Missing names mean zero. With
                `shared_dan` set, the values are summed and broadcast instead.
            dt: Timestep in seconds.
            feedback: Optional MBON -> DAN loop, ``f(mbon, valence) -> dict``.
                Called with this step's readout before any plasticity; what it
                returns is added to `dan`. In the fly, MBONs drive dopaminergic
                neurons directly (Felsenberg et al. 2017), which is how the
                circuit's own prediction can become a teaching signal -- the
                omission-of-punishment signal in `flyplay.conditioning` is built
                on this. Not called while no Kenyon cell is active: with nothing
                sensed there is no prediction to be wrong about.
            readouts: Compound-eye readouts ``(2, n_ommatidia, 2)`` for a
                mushroom body with vision. None means the eyes were not read
                this step, and the visual cells are silent for it.
        """
        olfactory, visual, kc = self._encode(concentration, readouts)
        mbon = {c.name: c.response(kc) for c in self.compartments}
        valence = float(sum(c.sign * mbon[c.name] for c in self.compartments))

        dan = dict(dan)
        if feedback is not None and kc.any():
            for name, value in feedback(mbon, valence).items():
                dan[name] = dan.get(name, 0.0) + float(value)

        if self.shared_dan:
            total = float(sum(dan.values()))
            applied = {c.name: total for c in self.compartments}
        else:
            applied = {c.name: float(dan.get(c.name, 0.0)) for c in self.compartments}

        # Nothing sensed, no Kenyon cells, nothing to depress -- dopamine
        # arriving while the fly perceives nothing must leave the memory
        # untouched, or the experiment measures a global decay instead of an
        # association.
        if kc.any():
            for compartment in self.compartments:
                compartment.learn(kc, applied[compartment.name], dt)
        else:
            for compartment in self.compartments:
                compartment.learn(kc, 0.0, dt)

        return MushroomBodyState(olfactory, mbon, applied, valence, visual,
                                 kc if self.visual is not None else None)

    # --- inspection -----------------------------------------------------

    def valence_of(self, concentration: np.ndarray, readouts: np.ndarray | None = None) -> float:
        """Current valence for what is sensed. Read-only; does not learn."""
        return self.read(concentration, readouts).valence

    def idle(self, seconds: float) -> None:
        """Let time pass with nothing sensed: recovery only, solved exactly.

        For the dark intervals of a protocol, which there is no reason to
        simulate step by step. Stepping instead gives the Euler version of the
        same decay; over a 12 s interval at 100 Hz the two differ by 6e-6 of
        the remaining depression at recovery 0.01, and 6.5e-5 at 0.033
        (computed).
        """
        for compartment in self.compartments:
            decay = np.exp(-compartment.recovery * seconds)
            compartment.weights[:] = compartment.w0 + (compartment.weights - compartment.w0) * decay

    def weight_summary(self) -> dict[str, float]:
        """Mean weight per compartment, for logging a session's progress."""
        return {c.name: float(c.weights.mean()) for c in self.compartments}

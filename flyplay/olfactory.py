"""ORN -> antennal lobe -> Kenyon cells: the olfactory front end FlyGym lacks.

`flyplay.odor` gives one scalar per sensor. That is enough to steer up a
gradient, but it cannot represent odour *identity* -- and without identity
there is nothing for a mushroom body to associate a punishment with. This
module inserts the three stages between the sensor and the mushroom body, each
of which has a clear computational job:

    ORN (50 types)   an odorant excites a *subset* of receptor types, and
                     each receptor saturates
    AL  (50 PNs)     divisive normalisation, bounding the population output
    KC  (2000, 5%)   sparse random expansion -> different odours land on
                     nearly disjoint cell populations

No connectome data is involved. Only the structure is copied, which is enough
to reproduce the two properties that matter downstream: **concentration
invariance** and **odour separation**. `scripts/13_mb_check.py` measures both.

Measured here, over the concentration range the fly actually walks through
(0.0016 to 0.11, a 69x span -- inverse-square falloff from a peak-1.0 source
between 25 mm and 3 mm away), 8 wiring seeds:

    KC pattern correlation, same odour, 69x concentration change   0.989
    KC pattern correlation, odour A against odour B                0.051
    Kenyon cells shared between odour A and odour B                8.2%

Push past that range and invariance decays -- 0.87 across 0.1 to 1.0, which is
a fly with its head on the source. Keep sources far enough away that the
working range holds, or the mushroom body will learn about "A at a distance"
and fail to recognise "A up close".

Separation has less headroom than the mean suggests: the worst of the 8 seeds
reaches 0.180 against a 0.2 ceiling. If a wiring seed ever fails that check,
raise `n_kc` or lower `receptor_fraction` rather than moving the threshold --
overlapping codes are exactly what stops two odours acquiring separate
memories.

The mushroom body codes identity, not direction, so this takes the mean over
the four sensors. Left-right asymmetry stays where it was -- in
`OdorField.asymmetry`, feeding the steering law directly.

**Two front ends.** `OlfactoryFrontEnd` is the structure-only version above:
random receptors, random claws, an exact top-k. The conditioning tasks
(`flyplay.conditioning`) run on it and their results were measured with it.
`ConnectomeFrontEnd` (bottom of this module) puts measured data in each stage:
receptor responses from DoOR, one right mushroom body's claws from the
hemibrain, Kenyon-cell types and the lobes they project to, and feedback
inhibition by an APL neuron instead of an exact count of winners. The sandbox
runs on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

#: Receptor types. The fly has ~50 functional ORN classes.
N_RECEPTORS = 50
#: Kenyon cells. The real number is ~2000 per hemisphere.
N_KC = 2000
#: Fraction of Kenyon cells active for any one odour. Measured in the fly at
#: roughly 5-10%; the APL feedback neuron is what enforces it.
KC_SPARSITY = 0.05
#: Projection neurons each Kenyon cell samples. ~6 in the connectome.
PN_PER_KC = 6
#: Fraction of receptor types one odour dimension drives. Must stay well below
#: 1.0: with a dense affinity matrix every entry is positive, so two odours
#: produce PN patterns correlated at ~0.7 and the KC layer cannot separate
#: them. 0.3 keeps the expected overlap between two odours at ~0.09.
RECEPTOR_FRACTION = 0.3


@dataclass(frozen=True)
class OlfactoryState:
    """One pass through the front end, kept for plotting and for the viewer."""

    #: Raw receptor drive before saturation, shape ``(n_receptors,)``.
    drive: np.ndarray
    #: Saturating ORN response, same shape.
    orn: np.ndarray
    #: Projection-neuron output after divisive normalisation, same shape.
    pn: np.ndarray
    #: Kenyon-cell activity, shape ``(n_kc,)``. Sparse, sums to 1 when any
    #: odour is present and is all zeros when none is.
    kc: np.ndarray
    #: Total receptor drive. Below `OlfactoryFrontEnd.min_drive` the fly is
    #: treated as smelling nothing.
    total_drive: float

    @property
    def active(self) -> np.ndarray:
        """Indices of the Kenyon cells that won, for raster plots."""
        return np.flatnonzero(self.kc)


class OlfactoryFrontEnd:
    """Turn an odour concentration vector into a sparse Kenyon-cell code.

    The random matrices are drawn once from `seed` and then fixed -- they are
    developmental wiring, not something that learns. Only the KC->MBON weights
    in `flyplay.mushroom_body` change with experience.

    Args:
        n_dimensions: Size of the odour space, matching `OdorField.n_dimensions`.
        n_receptors: ORN types.
        n_kc: Kenyon cells.
        sparsity: Fraction of Kenyon cells allowed to be active.
        pn_per_kc: Projection neurons sampled by each Kenyon cell.
        receptor_fraction: Fraction of receptor types each odour dimension
            drives. See `RECEPTOR_FRACTION`.
        hill_k: Half-saturating drive for the ORN response. Receptors saturate,
            which is *why* the antennal lobe needs to normalise; without it the
            concentration-invariance test would pass trivially because a linear
            stage followed by divisive normalisation is exactly scale-free.
        sigma: Additive term in the divisive normalisation. Sets how much of
            the low-concentration range stays un-normalised. Tuning it is
            pointless if you are looking at the Kenyon-cell code -- see
            `antennal_lobe` for why.
        min_drive: Total receptor drive below which the output is all zeros.
            Needed so a fly far from any source does not hallucinate a full
            Kenyon-cell pattern out of numerical dust -- and, more importantly,
            so dopamine arriving while no odour is present changes nothing.
        seed: Seed for the fixed random wiring.
    """

    def __init__(
        self,
        n_dimensions: int,
        *,
        n_receptors: int = N_RECEPTORS,
        n_kc: int = N_KC,
        sparsity: float = KC_SPARSITY,
        pn_per_kc: int = PN_PER_KC,
        receptor_fraction: float = RECEPTOR_FRACTION,
        hill_k: float = 1.0,
        sigma: float = 0.1,
        min_drive: float = 1e-4,
        seed: int = 0,
    ):
        if not 0 < sparsity < 1:
            raise ValueError(f"sparsity must be in (0, 1); got {sparsity}")
        #: Kept so saved synapses can be checked against the wiring they index.
        self.seed = int(seed)
        self.n_dimensions = int(n_dimensions)
        self.n_receptors = int(n_receptors)
        self.n_kc = int(n_kc)
        self.sparsity = float(sparsity)
        self.hill_k = float(hill_k)
        self.sigma = float(sigma)
        self.min_drive = float(min_drive)
        self.k = max(1, int(round(self.sparsity * self.n_kc)))

        rng = np.random.default_rng(seed)

        # ORN affinity. Each odour dimension excites a random subset of
        # receptor types with lognormal strengths -- receptor sensitivities
        # span orders of magnitude, so a uniform draw would make every odour
        # look like every other after normalisation.
        n_hit = max(1, int(round(receptor_fraction * self.n_receptors)))
        self.affinity = np.zeros((self.n_receptors, self.n_dimensions))
        for dim in range(self.n_dimensions):
            rows = rng.choice(self.n_receptors, size=n_hit, replace=False)
            self.affinity[rows, dim] = rng.lognormal(0.0, 0.7, size=n_hit)

        # KC projection: sparse connectivity with graded claw strengths. The
        # sparsity is the biology everyone quotes; the grading matters for a
        # duller reason. A pure odour leaves ~70% of glomeruli silent, so two
        # Kenyon cells that happen to claw the same *active* glomeruli have
        # identical drive however different the rest of their wiring is --
        # measured, 3 cells tied at the threshold on 2 of 5 seeds, which
        # silently costs the code its top-k sparsity. Real claws vary in
        # synapse count, and that variation breaks the ties.
        self.projection = np.zeros((self.n_kc, self.n_receptors))
        claws = np.minimum(pn_per_kc, self.n_receptors)
        for cell in range(self.n_kc):
            chosen = rng.choice(self.n_receptors, claws, replace=False)
            self.projection[cell, chosen] = rng.lognormal(0.0, 0.3, size=claws)

    # --- stages ---------------------------------------------------------

    def orn(self, concentration: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Receptor drive and saturating response for one concentration vector."""
        drive = self.affinity @ np.asarray(concentration, dtype=float)
        return drive, drive / (drive + self.hill_k)

    def antennal_lobe(self, orn: np.ndarray) -> np.ndarray:
        """Divisive normalisation across the glomerulus population.

        ``r_i / (sigma + sum_j r_j)``: gain control that bounds the projection
        neurons however strong the odour is.

        Be clear about what this does *not* do here. The denominator is a
        scalar, so this rescales the whole PN vector -- and `kenyon` ranks
        cells by a linear function of that vector, which is scale-invariant.
        Sweeping `sigma` from 0 to 1.0 changes the Kenyon-cell code by exactly
        nothing (measured). The concentration invariance in this pipeline comes
        from the ORN Hill stage staying near-linear over the working range, not
        from this step.

        The full Olsen-Wilson form, ``r^1.5 / (r^1.5 + sigma^1.5 + (m sum
        r)^1.5)``, *is* per-glomerulus and does reshape the code -- it was
        tried and measured worse here, dropping invariance from 0.989 to 0.822,
        because its compression flattens the pattern and destabilises the
        top-k ranking. It earns its keep across a far wider dynamic range than
        this task spans, and alongside the ORN adaptation this model omits.
        """
        return orn / (self.sigma + orn.sum())

    def kenyon(self, pn: np.ndarray) -> np.ndarray:
        """Sparse expansion: random projection, then keep the top k.

        The threshold is a hard top-k rather than a fixed value, which is what
        APL feedback does -- it scales inhibition until only a small fraction
        of Kenyon cells clear threshold, whatever the input strength. Winners
        are renormalised to sum to 1 so that the mushroom body reads out odour
        *identity* and leaves intensity to the steering law.
        """
        return top_k_code(self.projection @ pn, self.k)

    # --- whole pathway --------------------------------------------------

    def __call__(self, concentration: np.ndarray) -> OlfactoryState:
        """Run one concentration vector through all three stages."""
        drive, orn = self.orn(concentration)
        total = float(drive.sum())
        if total < self.min_drive:
            zeros = np.zeros(self.n_receptors)
            return OlfactoryState(drive, zeros, zeros, np.zeros(self.n_kc), total)
        pn = self.antennal_lobe(orn)
        return OlfactoryState(drive, orn, pn, self.kenyon(pn), total)

    def encode(self, concentration: np.ndarray) -> np.ndarray:
        """Just the Kenyon-cell code. Shorthand for ``self(c).kc``."""
        return self(concentration).kc


def top_k_code(drive: np.ndarray, k: int) -> np.ndarray:
    """Keep the `k` most driven cells, relative to the strongest loser; sum to 1.

    Shared by the olfactory and visual Kenyon cells: APL inhibition does not
    care which sense drove a cell.
    """
    # Threshold at the strongest *loser*, not the weakest winner: that is
    # what an inhibition level is, and it leaves all k winners strictly
    # above it. Subtracting the weakest winner instead silences that cell,
    # so the code carries k-1 active cells and the sparsity is a lie.
    order = np.argpartition(drive, -(k + 1))
    winners = order[-k:]
    threshold = drive[order[-(k + 1)]]
    kc = np.zeros(drive.shape[0])
    kc[winners] = np.maximum(drive[winners] - threshold, 0.0)
    total = kc.sum()
    if total <= 0:
        # Degenerate case: every winner tied with the threshold.
        kc[winners] = 1.0 / len(winners)
        return kc
    return kc / total


def pattern_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two population patterns.

    Cosine rather than Pearson: these are non-negative rate vectors where the
    zeros are meaningful (that cell is silent), and mean-subtracting a sparse
    code manufactures negative values that no neuron has.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else 0.0



# --- measured data: DoOR receptors and hemibrain Kenyon cells ---------------------------

#: Tables built by `scripts/18_build_brain_data.py`; sources and licences in
#: `flyplay/data/SOURCES.md`.
DATA = Path(__file__).with_name("data")

#: The room's odours as blends of DoOR odorants (`door_odorants.csv` keys).
#: DoOR has no vinegar, only pure compounds. Acetic acid alone drives DP1m, DC4
#: and DP1l but leaves DM1 and VA2 slightly inhibited (-0.02, -0.03), and those
#: two glomeruli are the ones flies need to be drawn to vinegar (Semmelhack &
#: Wang 2009): ethyl acetate drives DM1 (0.61) and acetoin VA2. The proportions
#: are a model choice, not a measured headspace. 3-octanol and
#: 4-methylcyclohexanol are single compounds, as in the classic assay.
ODORANT_BLENDS: dict[str, dict[str, float]] = {
    "vinegar": {"acetic_acid": 1.0, "ethyl_acetate": 0.5, "acetoin": 0.5},
    "octanol": {"octanol": 1.0},
    "mch": {"mch": 1.0},
}
#: Kenyon-cell types whose input is visual, not olfactory (Li et al. 2020):
#: gamma-d and alpha/beta-p. They join through `flyplay.visual_pathway`.
VISUAL_KC_TYPES = ("KCg-d", "KCab-p")
#: The three Kenyon-cell classes, named by the lobes their axons run through.
LOBES = ("gamma", "apbp", "ab")
#: A PN-to-KC connection counts as a claw from this many synapses. The hemibrain
#: gives 6.05 glomeruli per KC counting every synapse, 5.32 from 3 synapses and
#: 5.15 from 5; the light microscopy count is 5.2 claws (Zheng et al. 2022).
MIN_CLAW_SYNAPSES = 3
#: APL feedback strength: the gain of `apl_code` times the number of cells in
#: the lobe it inhibits, so lobes of any size get the same sparseness. Measured
#: over the 12 DoOR odorants of `door_odorants.csv` at concentration 0.03, two
#: sampled flies: 45 activates a median 7.2-8.8% of each lobe's cells, 60
#: activates 6.0-6.9%; 52 puts the median odour in the middle of the measured
#: 5-10% (Lin et al. 2014), with single odorants ranging over about 3-11%.
APL_FEEDBACK = 52.0
#: Share of APL output left by the block of Lin et al. (2014), whose
#: temperature-sensitive shibire silences it incompletely.
APL_BLOCKED = 0.2
#: The share of cells a typical odour activates at `APL_FEEDBACK`: the
#: population's nominal code size, for scaling the visual code and learning
#: rates.
NOMINAL_ACTIVE = 0.075


def lobe_of(kc_type: str) -> str:
    """"gamma", "apbp" or "ab" for a hemibrain type name such as KCab-s."""
    if kc_type.startswith("KCg"):
        return "gamma"
    if kc_type.startswith("KCa'b'"):
        return "apbp"
    if kc_type.startswith("KCab"):
        return "ab"
    raise ValueError(f"not a Kenyon-cell type: {kc_type!r}")


@lru_cache(maxsize=1)
def door_odorants() -> tuple[tuple[str, ...], dict[str, np.ndarray]]:
    """(glomeruli, odorant key -> response per glomerulus, NaN where unmeasured)."""
    import csv

    with open(DATA / "door_odorants.csv", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    glomeruli = tuple(header[3:])
    table = {row[0]: np.array([float(v) if v else np.nan for v in row[3:]]) for row in body}
    return glomeruli, table


@lru_cache(maxsize=1)
def hemibrain_kcs() -> dict[str, np.ndarray]:
    with np.load(DATA / "hemibrain_kc.npz") as data:
        return {k: data[k] for k in data.files}


def apl_code(drive: np.ndarray, gain: float) -> np.ndarray:
    """Kenyon-cell output under feedback inhibition from the APL neuron.

    Every cell is inhibited by the same amount, `gain` times the summed output
    of all of them: ``a_i = max(0, d_i - gain * sum(a))``. The fixed point is
    unique and found exactly: with the drives sorted, the top m cells are
    active until the next cell's drive falls under the inhibition they raise.
    Unlike a fixed count of winners, how many cells clear the inhibition
    depends on the odour -- one that drives a few cells strongly activates
    fewer than one that drives many moderately. Scaling every drive scales
    the solution, so the pattern is independent of overall intensity, as with
    top-k. Returns the output summed to 1, or zeros for no drive.
    """
    d = np.maximum(np.asarray(drive, dtype=float), 0.0)
    if not d.any():
        return np.zeros_like(d)
    order = np.argsort(d)[::-1]
    ranked = d[order]
    m = np.arange(1, ranked.size + 1)
    inhibition = gain * np.cumsum(ranked) / (1.0 + gain * m)
    below = np.flatnonzero(np.append(ranked[1:], 0.0) <= inhibition)
    active = int(below[0]) + 1
    code = np.zeros_like(d)
    code[order[:active]] = ranked[:active] - inhibition[active - 1]
    total = code.sum()
    if total <= 0.0:
        code[order[:active]] = 1.0 / active
        return code
    return code / total


class ConnectomeFrontEnd:
    """Odour -> glomeruli -> Kenyon cells, on measured receptors and wiring.

    **Receptors.** Each glomerulus responds to an odour as DoOR's consensus for
    the receptor it holds (spontaneous rate subtracted; inhibition clipped to
    zero, and an unmeasured pair counts as no response). A room odour is a
    blend of odorants (`ODORANT_BLENDS`). Receptors saturate and the antennal
    lobe normalises, exactly as in `OlfactoryFrontEnd`.

    **Claws.** Every olfactory Kenyon cell of the hemibrain's right mushroom
    body (1761 of its 1927 cells have PN input; the gamma-d and alpha/beta-p
    cells are visual) keeps its type, and so its lobe. Its drive is the
    synapse-weighted mean of the PN activity at its claws, so a gamma cell with
    8 claws is not driven harder than an alpha/beta cell with 5.

    ``wiring="hemibrain"`` uses that one fly's claws. ``wiring="sampled"``
    (default) gives each fly its own: every cell keeps its number of claws and
    their synapse counts, and draws which glomeruli they come from in
    proportion to how much its cell type samples each glomerulus in the
    hemibrain. Claw assignment differs from fly to fly (Caron et al. 2013) while
    the biases of each type -- the over-sampled food-odour glomeruli, for one
    (Zheng et al. 2022) -- stay. `seed` picks the fly.

    **APL.** `apl_code`, run on each lobe's cells separately: APL activity and
    inhibition are local to where it is excited (Amin et al. 2020). Each lobe
    then carries a fixed share of the code, its share of the cells -- which
    also keeps a compartment's input from swinging with how an odour happens to
    split between lobes. With one global inhibition, gamma cells (8 claws,
    averaging more glomeruli, so rarely the most driven) took 13-23% of the
    code for their 35% of the cells, and the gamma1pedc compartment learned a
    third as fast.

    Args:
        odours: Names of the odour dimensions, keys of `ODORANT_BLENDS`.
        wiring: "sampled" or "hemibrain".
        seed: The fly, for sampled wiring.
        apl: APL output, 1 intact; `APL_BLOCKED` for the ablation.
        hill_k, sigma, min_drive: As in `OlfactoryFrontEnd`.
    """

    def __init__(
        self,
        odours,
        *,
        wiring: str = "sampled",
        seed: int = 0,
        apl: float = 1.0,
        hill_k: float = 1.0,
        sigma: float = 0.1,
        min_drive: float = 1e-4,
    ):
        if wiring not in ("sampled", "hemibrain"):
            raise ValueError(f"wiring is 'sampled' or 'hemibrain'; got {wiring!r}")
        self.odours = tuple(odours)
        unknown = [o for o in self.odours if o not in ODORANT_BLENDS]
        if unknown:
            raise ValueError(f"no DoOR blend for {unknown}; known: {sorted(ODORANT_BLENDS)}")
        self.wiring = wiring
        self.seed = int(seed)
        self.apl = float(apl)
        self.hill_k = float(hill_k)
        self.sigma = float(sigma)
        self.min_drive = float(min_drive)
        self.n_dimensions = len(self.odours)

        glomeruli, table = door_odorants()
        data = hemibrain_kcs()
        if tuple(str(g) for g in data["glomeruli"]) != glomeruli:
            raise ValueError("door_odorants.csv and hemibrain_kc.npz disagree on the glomeruli; "
                             "rebuild both with scripts/18_build_brain_data.py")
        self.glomeruli = glomeruli
        self.n_receptors = len(glomeruli)

        # Receptors: glomerulus x odour dimension.
        self.affinity = np.zeros((self.n_receptors, self.n_dimensions))
        for j, odour in enumerate(self.odours):
            for odorant, share in ODORANT_BLENDS[odour].items():
                self.affinity[:, j] += share * np.nan_to_num(np.maximum(table[odorant], 0.0))

        # Claws: olfactory Kenyon cells only.
        types = np.array([str(t) for t in data["kc_type"]])
        synapses = data["pn_synapses"].astype(float)
        synapses[synapses < MIN_CLAW_SYNAPSES] = 0.0
        olfactory = ~np.isin(types, VISUAL_KC_TYPES) & (synapses.sum(axis=1) > 0)
        self.kc_types = types[olfactory]
        self.kc_body_id = data["kc_body_id"][olfactory]
        self.lobes = np.array([lobe_of(t) for t in self.kc_types])
        claws = synapses[olfactory]
        if wiring == "sampled":
            claws = self._sample(claws, self.kc_types, np.random.default_rng(seed))
        self.claws = claws
        self.projection = claws / claws.sum(axis=1, keepdims=True)
        self.n_kc = int(self.projection.shape[0])
        self._lobe_cells = {lobe: np.flatnonzero(self.lobes == lobe) for lobe in LOBES}
        #: The nominal number of active cells, for scaling the visual code
        #: (`flyplay.mushroom_body`).
        self.k = int(round(NOMINAL_ACTIVE * self.n_kc))

    @staticmethod
    def _sample(claws: np.ndarray, types: np.ndarray, rng) -> np.ndarray:
        """Each cell's claws, redrawn from its type's glomerulus preferences."""
        out = np.zeros_like(claws)
        for kc_type in np.unique(types):
            cells = np.flatnonzero(types == kc_type)
            # Types with a handful of cells borrow their lobe's preferences.
            pool = cells if cells.size >= 20 else np.flatnonzero(
                np.char.startswith(types.astype(str), kc_type.split("-")[0]))
            # A floor keeps every glomerulus drawable, so a cell with more
            # claws than its type has favoured glomeruli can still be wired.
            preference = claws[pool].sum(axis=0) + 1e-6
            preference = preference / preference.sum()
            for cell in cells:
                counts = claws[cell][claws[cell] > 0]
                chosen = rng.choice(claws.shape[1], size=counts.size, replace=False, p=preference)
                out[cell, chosen] = rng.permutation(counts)
        return out

    # --- stages: as OlfactoryFrontEnd, with measured receptors and claws ------------

    def orn(self, concentration: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        drive = self.affinity @ np.asarray(concentration, dtype=float)
        return drive, drive / (drive + self.hill_k)

    def antennal_lobe(self, orn: np.ndarray) -> np.ndarray:
        return orn / (self.sigma + orn.sum())

    def kenyon(self, pn: np.ndarray) -> np.ndarray:
        """Per-lobe APL codes, each lobe weighted by its share of the cells."""
        drive = self.projection @ pn
        kc = np.zeros(self.n_kc)
        for cells in self._lobe_cells.values():
            if cells.size:
                gain = self.apl * APL_FEEDBACK / cells.size
                kc[cells] = apl_code(drive[cells], gain) * (cells.size / self.n_kc)
        return kc

    def __call__(self, concentration: np.ndarray) -> OlfactoryState:
        drive, orn = self.orn(concentration)
        total = float(drive.sum())
        if total < self.min_drive:
            zeros = np.zeros(self.n_receptors)
            return OlfactoryState(drive, zeros, zeros, np.zeros(self.n_kc), total)
        pn = self.antennal_lobe(orn)
        return OlfactoryState(drive, orn, pn, self.kenyon(pn), total)

    def encode(self, concentration: np.ndarray) -> np.ndarray:
        return self(concentration).kc

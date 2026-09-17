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
"""

from __future__ import annotations

from dataclasses import dataclass

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

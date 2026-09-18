"""Retina -> visual projection neurons -> visual Kenyon cells.

About 8% of the fly's Kenyon cells are visual (Ganguly et al. 2024), and the
gamma-d cells among them are what a walking fly needs to learn that a colour
predicts punishment or sugar (Vogt et al. 2014, 2016). This module is their
front end, built the way `flyplay.olfactory` builds the olfactory one: the
structure is copied, no connectome data is used.

    photoreceptors   each ommatidium reads one render channel -- yellow-type
                     the green one, pale-type the blue one, nothing the red
    colour VPNs      small patches of the floor band (below), each reporting
                     how far the pale:yellow balance leans to its side of
                     neutral -- hue, not brightness (VPN-MB1: many cells,
                     ~20 cartridges each; Vogt et al. 2016)
    brightness VPNs  the band's brightness, range-coded by four units tuned
                     to log-spaced levels (VPN-MB2: wide field, 1-3 cells;
                     Vogt et al. 2016)
    visual KCs       100 cells, three VPN inputs each, top-k

**Where it looks.** The colour sits on the floor in the Vogt assay, and medulla
VPNs onto gamma-d cells cover the ventral field (Ganguly et al. 2024). But the
lower half of the eye sees more than floor -- measured over a black floor,
yellow-type / pale-type, by row quantile:

    0.5-0.6   0.121 / 0.067   the horizon rows blend in sky
    0.6-0.8   0.000 / 0.000   floor only
    0.8-1.0   up to 0.076     the fly's own legs

Pooled over the whole lower half, those offsets made a dim green look
desaturated (pale share 0.30 against 0.15 at full brightness). The VPNs
therefore read only the 0.6-0.8 band, where green and green at a tenth read
0.486/0.027 and 0.047/0.004 -- nearly pure scaling -- and a rendered floor
gives the same code as a synthetic one (cosine 0.99-0.999).

**Hue lives in the ratio, brightness in the mean**, which is what lets the two
VPN families be silenced independently: the double dissociation of Vogt et al.
(2016), colour memory needing VPN-MB1 and brightness memory VPN-MB2. Brightness
is the average of the two types' means, not of all ommatidia -- the fly's
luminance channel is R1-6, which FlyGym's retina does not model -- and with it
the blue and green floors are exactly equally bright: their band readings are
mirror images, 0.027/0.486 and 0.486/0.027.

**Hue and brightness compete for the same few winners**, and the design is a
trade-off between them. Cosine between codes, mean over 16 wiring seeds; the
first two rows are alternatives that were measured and dropped:

                                  blue-green  green-tenth green     blue-dim blue
    ratio hue, ON/OFF pair           0.000      0.88 (0.59-1.00)          0.98
    opponent hue, 4 bands            0.000      0.83 (0.57-1.00)          0.92
    this module (brightness input    0.000      0.71 (0.15-0.98)          0.85
    on 1 cell in 2)

With hue units outnumbering brightness units, a plain random draw leaves the
code nearly blind to brightness, and the brightness task of Vogt et al. (2016)
could not be learned. Yet VPN-MB2 is only 1-3 cells and that task still needs
the gamma-d cells, so those few cells must reach many of them; hence the last
row, a brightness input on half the cells. The prices: hue generalises less
across brightness, and wiring seeds differ a lot in how well they code
brightness -- the worst is nearly blind to it. Both silencing identities hold
exactly on every seed. See `BRIGHTNESS_INPUT_FRACTION`.

Both eyes share one wiring -- two mirror-image mushroom bodies -- and each eye
gets its own code, so steering can compare them. One pass costs 0.06 ms for
both eyes, against 25 ms for the render feeding it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from flygym.vision.retina import Retina

from flyplay.olfactory import top_k_code
from flyplay.vision import ommatidia_centroids

#: Visual Kenyon cells per hemisphere. The hemibrain has 99 gamma-d cells (Li
#: et al. 2020); FlyWire 147-148.
N_VISUAL_KC = 100
#: Fraction active, as for the olfactory cells.
VISUAL_SPARSITY = 0.05
#: Colour VPNs per eye. Vogt et al. (2016) report "many cell bodies" for
#: VPN-MB1 without a count.
N_COLOUR_VPN = 24
#: Ommatidia pooled by one colour VPN: ~20 cartridges for VPN-MB1.
PATCH_SIZE = 20
#: VPN inputs per visual Kenyon cell: median 3, range 1-7 (Ganguly et al. 2024).
VPN_PER_KC = 3
#: Preferred brightness of each brightness unit. Log-spaced over the measured
#: floor range: 0.025 (green at a tenth of full intensity) to 0.486 (white).
BRIGHTNESS_BANDS = (0.02, 0.06, 0.18, 0.54)
#: Tuning width of a brightness unit, in natural-log units -- half the band
#: spacing (ln 3), so neighbouring units overlap at 0.61 of their peak.
BAND_WIDTH = 0.55
#: Fraction of visual Kenyon cells with one brightness input. The trade-off in
#: the module docstring: at 0.5 green and a tenth-bright green share a code
#: with cosine 0.71 on average instead of 0.83-0.88, while blue and green stay
#: fully separate. Not fitted to any behavioural score; inferred from VPN-MB2
#: being few cells that the brightness task nonetheless needs.
BRIGHTNESS_INPUT_FRACTION = 0.5
#: Ommatidia below this row quantile form the lower field.
VENTRAL_QUANTILE = 0.5
#: Row-quantile band the VPNs pool: floor only, no sky and no legs (module
#: docstring). 150 ommatidia per eye, 40 of them pale-type.
VPN_BAND = (0.6, 0.8)
#: Kinds of silencing `VisualFrontEnd.silenced` accepts.
VPN_FAMILIES = ("colour", "brightness")

#: Measured means over the `VPN_BAND` ommatidia, (yellow-type, pale-type), for
#: `flyplay.arena.FLOOR_COLOURS` tiles under the default headlight. Blue and
#: green are exact mirror images, so to this model they are exactly equally
#: bright (0.257); the dim ones keep their hue (pale share 0.945 vs 0.947).
#: "checker" is FlyGym's default grey floor.
FLOOR_READINGS: dict[str, tuple[float, float]] = {
    "blue": (0.027, 0.486),
    "green": (0.486, 0.027),
    "dim_blue": (0.012, 0.204),
    "dim_green": (0.204, 0.012),
    "green_tenth": (0.047, 0.004),
    "grey": (0.255, 0.255),
    "white": (0.486, 0.486),
    "checker": (0.179, 0.178),
}
#: Upper-field reading of the white sky, measured.
SKY_READING = 0.99


@lru_cache(maxsize=1)
def eye_layout() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(pale, lower, row, col)`` per ommatidium, shared by both eyes."""
    retina = Retina()
    row, col = ommatidia_centroids(retina)
    pale = retina.pale_type_mask.astype(bool)
    lower = row >= np.quantile(row, VENTRAL_QUANTILE)
    return pale, lower, row, col


@lru_cache(maxsize=1)
def vpn_field() -> np.ndarray:
    """Mask of the ommatidia the VPNs pool: the `VPN_BAND` rows."""
    _, _, row, _ = eye_layout()
    lo, hi = np.quantile(row, VPN_BAND)
    return (row >= lo) & (row <= hi)


def photoreceptors(readouts: np.ndarray) -> np.ndarray:
    """Each ommatidium's reading on its own channel, shape ``(2, n_ommatidia)``.

    ``get_ommatidia_readouts`` puts yellow-type ommatidia in channel 0 and
    pale-type in channel 1, with zero in the other channel.
    """
    pale = eye_layout()[0]
    return np.where(pale[None, :], readouts[:, :, 1], readouts[:, :, 0])


def floor_readouts(
    left: tuple[float, float],
    right: tuple[float, float] | None = None,
    *,
    sky: float = SKY_READING,
) -> np.ndarray:
    """Synthetic ``(2, n_ommatidia, 2)`` readouts: a uniform floor under the sky.

    For checks that should not need a renderer. The values are per-type means,
    e.g. ``FLOOR_READINGS["blue"]``, written over the whole lower half (the VPNs
    read only their band of it); each eye gets its own floor so a
    colour boundary can be put between them.
    """
    pale, lower, _, _ = eye_layout()
    readouts = np.zeros((2, pale.size, 2))
    for eye, (yellow, pale_value) in enumerate((left, right or left)):
        own = np.where(pale, pale_value, yellow)
        own = np.where(lower, own, sky)
        readouts[eye, ~pale, 0] = own[~pale]
        readouts[eye, pale, 1] = own[pale]
    return readouts


@dataclass(frozen=True)
class VisualState:
    """One pass through the visual front end, both eyes."""

    #: Own-channel reading per ommatidium, ``(2, n_ommatidia)``.
    own: np.ndarray
    #: VPN outputs, ``(2, n_vpn)``: colour VPNs first, then the brightness bands.
    vpn: np.ndarray
    #: Kenyon-cell code per eye, ``(2, n_kc)``. Each row sums to 1, or is all
    #: zeros when nothing drives the cells.
    kc: np.ndarray

    @property
    def mean_kc(self) -> np.ndarray:
        """Both eyes' codes averaged: what the shared synapses learn from."""
        return self.kc.mean(axis=0)

    def active(self, eye: int) -> np.ndarray:
        """Indices of the winning cells in one eye, for raster plots."""
        return np.flatnonzero(self.kc[eye])


class VisualFrontEnd:
    """Turn compound-eye readouts into a sparse visual Kenyon-cell code.

    The wiring -- which ommatidia each colour VPN pools, which VPNs each Kenyon
    cell samples, and with what weight -- is drawn once from `seed` and fixed.
    Ganguly et al. (2024) could not tell the VPN -> gamma-d wiring apart from
    shuffled matrices, so random is the faithful choice, not a shortcut.

    Args:
        n_kc: Visual Kenyon cells.
        sparsity: Fraction of them allowed to be active.
        n_colour_vpn: Colour VPNs; half prefer the pale (blue) side of the
            balance and half the yellow (green) side.
        patch_size: Ommatidia pooled by each colour VPN.
        vpn_per_kc: VPN inputs per Kenyon cell.
        brightness_bands: Preferred brightness of each brightness unit.
        band_width: Tuning width of those units, natural-log units.
        brightness_input_fraction: Fraction of cells given one brightness
            input; the rest draw every input from the colour VPNs.
        seed: Seed for the fixed wiring.
    """

    def __init__(
        self,
        *,
        n_kc: int = N_VISUAL_KC,
        sparsity: float = VISUAL_SPARSITY,
        n_colour_vpn: int = N_COLOUR_VPN,
        patch_size: int = PATCH_SIZE,
        vpn_per_kc: int = VPN_PER_KC,
        brightness_bands: tuple[float, ...] = BRIGHTNESS_BANDS,
        band_width: float = BAND_WIDTH,
        brightness_input_fraction: float = BRIGHTNESS_INPUT_FRACTION,
        seed: int = 0,
    ):
        if not 0 < sparsity < 1:
            raise ValueError(f"sparsity must be in (0, 1); got {sparsity}")
        self.seed = int(seed)
        self.n_kc = int(n_kc)
        self.sparsity = float(sparsity)
        self.n_colour_vpn = int(n_colour_vpn)
        self.patch_size = int(patch_size)
        self.vpn_per_kc = int(vpn_per_kc)
        self.brightness_bands = tuple(float(b) for b in brightness_bands)
        self.band_width = float(band_width)
        self.brightness_input_fraction = float(brightness_input_fraction)
        self.k = max(1, int(round(self.sparsity * self.n_kc)))
        self.n_brightness_vpn = len(self.brightness_bands)
        self.n_vpn = self.n_colour_vpn + self.n_brightness_vpn
        self._log_bands = np.log(np.asarray(self.brightness_bands))
        #: VPN families switched off, a subset of `VPN_FAMILIES`. The
        #: silencing experiments of Vogt et al. (2016), done in software.
        self.silenced: set[str] = set()
        #: The last call's readouts, silencing and answer; see `__call__`.
        self._last: tuple[np.ndarray, frozenset[str], VisualState] | None = None

        pale, _, row, col = eye_layout()
        self.pale = pale
        #: The ommatidia the VPNs pool. Named `lower` because it is the lower
        #: eye, trimmed of the rows that see sky or legs.
        self.lower = vpn_field()
        rng = np.random.default_rng(seed)

        # Colour VPN patches: the ommatidia nearest a random centre in the
        # field. Label-image pixels are the only geometry FlyGym gives, and
        # neighbouring ommatidia are ~15 px apart in it. The field holds 150
        # ommatidia, so 24 patches of 20 overlap about three times over.
        field = np.flatnonzero(self.lower)
        centres = rng.choice(field, self.n_colour_vpn, replace=False)
        xy = np.stack([row, col], axis=1)
        distance = np.linalg.norm(xy[field][None, :, :] - xy[centres][:, None, :], axis=2)
        self.patches = field[np.argsort(distance, axis=1)[:, : self.patch_size]]
        self._patch_pale = pale[self.patches]
        self._n_patch_pale = np.maximum(self._patch_pale.sum(axis=1), 1)
        self._n_patch_yellow = np.maximum((~self._patch_pale).sum(axis=1), 1)
        #: True where a colour VPN reports the pale (blue) share.
        self.prefers_blue = np.arange(self.n_colour_vpn) % 2 == 0

        # Claw weights are lognormal for the reason given in
        # `flyplay.olfactory`: without them, cells sampling the same active
        # inputs tie at the threshold and the code loses its sparsity.
        self.projection = np.zeros((self.n_kc, self.n_vpn))
        for cell in range(self.n_kc):
            n_brightness = int(rng.random() < self.brightness_input_fraction)
            chosen = np.concatenate([
                rng.choice(self.n_colour_vpn, self.vpn_per_kc - n_brightness, replace=False),
                self.n_colour_vpn
                + rng.choice(self.n_brightness_vpn, n_brightness, replace=False),
            ])
            self.projection[cell, chosen] = rng.lognormal(0.0, 0.3, size=chosen.size)

    @property
    def params(self) -> dict:
        """Everything that determines the wiring, for saved memories to check."""
        return {
            "n_kc": self.n_kc,
            "sparsity": self.sparsity,
            "n_colour_vpn": self.n_colour_vpn,
            "patch_size": self.patch_size,
            "vpn_per_kc": self.vpn_per_kc,
            "brightness_bands": list(self.brightness_bands),
            "band_width": self.band_width,
            "brightness_input_fraction": self.brightness_input_fraction,
            "seed": self.seed,
        }

    @property
    def vpn_labels(self) -> list[str]:
        colour = ["blue" if b else "green" for b in self.prefers_blue]
        return colour + [f"brightness {b:g}" for b in self.brightness_bands]

    # --- stages ---------------------------------------------------------

    def vpns(self, own: np.ndarray) -> np.ndarray:
        """VPN outputs for both eyes from own-channel readings, ``(2, n_vpn)``."""
        values = own[:, self.patches]  # (2, n_colour_vpn, patch_size)
        pale_mean = (values * self._patch_pale).sum(axis=2) / self._n_patch_pale
        yellow_mean = (values * ~self._patch_pale).sum(axis=2) / self._n_patch_yellow
        total = pale_mean + yellow_mean
        # Opponent and rectified: zero for a grey patch, up to 1 for a pure
        # one. A plain ratio sits at 0.5 on grey, which made every colour unit
        # look half-driven and let them crowd brightness out of the code.
        balance = np.divide(
            pale_mean - yellow_mean, total, out=np.zeros_like(total), where=total > 0
        )
        colour = np.where(
            self.prefers_blue[None, :], np.maximum(balance, 0.0), np.maximum(-balance, 0.0)
        )

        brightness = 0.5 * (
            own[:, self.lower & self.pale].mean(axis=1)
            + own[:, self.lower & ~self.pale].mean(axis=1)
        )
        log_brightness = np.log(np.maximum(brightness, 1e-6))
        bands = np.exp(
            -((log_brightness[:, None] - self._log_bands[None, :]) ** 2)
            / (2.0 * self.band_width**2)
        )

        vpn = np.concatenate([colour, bands], axis=1)
        if "colour" in self.silenced:
            vpn[:, : self.n_colour_vpn] = 0.0
        if "brightness" in self.silenced:
            vpn[:, self.n_colour_vpn :] = 0.0
        return vpn

    def kenyon(self, vpn: np.ndarray) -> np.ndarray:
        """Sparse code per eye, ``(2, n_kc)``. All zeros for an eye with no drive."""
        drive = vpn @ self.projection.T
        kc = np.zeros_like(drive)
        for eye in range(drive.shape[0]):
            # With every input silenced there is nothing to rank, and top-k
            # would pick k arbitrary cells out of a row of zeros.
            if drive[eye].max() > 0.0:
                kc[eye] = top_k_code(drive[eye], self.k)
        return kc

    def __call__(self, readouts: np.ndarray) -> VisualState:
        """The code for one pass of the eyes, recomputed only when they change.

        The sandbox reads the eyes at 10 Hz in most rooms and hands the same
        readouts to `MushroomBody.step` for the next nine physics steps, so
        this ran a hundred times a second on one picture. Keeping the last
        answer costs a copy of the readouts and a comparison, about 2 us,
        against 135 us to redo the pooling and the top-k: 13.5 ms of each
        simulated second down to 1.9, the sandbox fingerprint unchanged. The
        comparison is of the values, not the array's identity, so a caller
        that writes into its readouts in place still gets a fresh answer.
        Nothing is saved in a room where something moves: there the eyes are
        read at 100 Hz and every readout is new.
        """
        silenced = frozenset(self.silenced)
        last = self._last
        if last is not None and last[1] == silenced and np.array_equal(last[0], readouts):
            return last[2]
        own = photoreceptors(np.asarray(readouts, dtype=float))
        vpn = self.vpns(own)
        state = VisualState(own, vpn, self.kenyon(vpn))
        self._last = (np.array(readouts, copy=True), silenced, state)
        return state

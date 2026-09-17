"""Compress 1442 ommatidia into a handful of numbers a policy can learn from.

`Simulation.get_ommatidia_readouts` returns ``(2 eyes, 721 ommatidia, 2
channels)``. Feeding that to an MLP is hopeless and feeding it to a CNN needs a
separate supervised training stage, so this module extracts a small analytic
feature vector instead -- the kind of bilateral contrast summary an elementary
visual projection neuron could plausibly compute.

Two measurements shaped the design:

- **The ground dominates.** On flat terrain with nothing in view, 337 of 721
  ommatidia per eye already read dark, because the lower visual field is
  filled with the checkerboard floor. A naive "count dark ommatidia" feature is
  therefore mostly a floor detector. Features here are restricted to the
  **upper visual field**, above a horizon row calibrated from an empty scene.
- **Lateralisation works.** A dark pillar 6 mm ahead and 1.5 mm to the left
  darkened 34 extra ommatidia in the left eye and 0 in the right. Left-right
  difference is the usable signal.

Ommatidium positions come from `Retina.ommatidia_id_map`, a ``(512, 450)``
label image: the centroid of each label gives that ommatidium's row (roughly
elevation) and column (roughly azimuth).
"""

from __future__ import annotations

import numpy as np
from flygym.vision.retina import Retina

#: Feature names, in the order `RetinaFeatures.extract` returns them.
FEATURE_NAMES = (
    "left_mass",  # fraction of the left eye's upper field that is dark
    "right_mass",  # same, right eye
    "left_azimuth",  # where in the left eye's field the dark stuff sits, -1..1
    "right_azimuth",  # same, right eye
    "mass_asymmetry",  # left_mass - right_mass; >0 means obstacle on the left
    "total_mass",  # left_mass + right_mass; grows as something looms
)
N_FEATURES = len(FEATURE_NAMES)


def ommatidia_centroids(retina: Retina | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Centroid row and column of every ommatidium, from the retina's label image.

    FlyGym ships no viewing-direction table, only ``ommatidia_id_map``: a
    ``(512, 450)`` raster labelling each pixel with its ommatidium (0 is
    background). Row is roughly elevation -- smaller is higher -- and column
    roughly azimuth. Measured: the half of the ommatidia below the median row
    reads the floor in both eyes, the half above reads the sky.

    Returns:
        ``(row, col)``, each of shape ``(num_ommatidia_per_eye,)``.
    """
    retina = retina or Retina()
    n = retina.num_ommatidia_per_eye
    label_map = retina.ommatidia_id_map
    rows, cols = np.nonzero(label_map)
    ids = label_map[rows, cols] - 1
    counts = np.bincount(ids, minlength=n).astype(float)
    return (
        np.bincount(ids, weights=rows, minlength=n) / counts,
        np.bincount(ids, weights=cols, minlength=n) / counts,
    )


class RetinaFeatures:
    """Analytic bilateral features from raw ommatidia readouts.

    Args:
        dark_threshold: Intensity below this counts as "dark". Readouts are in
            [0, 1].
        horizon_quantile: Fraction of the visual field treated as sky. The
            horizon row is set so this fraction of ommatidia lies above it.
            Only the upper field is used, to keep the floor out of the signal.
    """

    def __init__(self, dark_threshold: float = 0.35, horizon_quantile: float = 0.5):
        self.dark_threshold = dark_threshold
        retina = Retina()
        self.n_ommatidia = retina.num_ommatidia_per_eye
        self.row, self.col = ommatidia_centroids(retina)

        self.horizon_row = float(np.quantile(self.row, horizon_quantile))
        self.upper = self.row < self.horizon_row  # smaller row index = higher up
        self.n_upper = int(self.upper.sum())

        # Azimuth normalised to [-1, 1] across the upper field, so the feature
        # is independent of the retina's pixel dimensions.
        col_upper = self.col[self.upper]
        lo, hi = col_upper.min(), col_upper.max()
        self._azimuth = np.zeros(self.n_ommatidia)
        self._azimuth[self.upper] = 2 * (col_upper - lo) / max(hi - lo, 1e-9) - 1

    def calibrate_horizon(self, readouts: np.ndarray, margin: int = 12) -> None:
        """Set the horizon from an empty reference scene.

        Walks the horizon row upward until the reference scene leaves almost
        nothing dark in the upper field, so anything dark there later is an
        actual object rather than the floor.

        Args:
            readouts: ``(2, n_ommatidia, 2)`` from an obstacle-free scene.
            margin: How many dark ommatidia per eye to tolerate in the
                reference.
        """
        intensity = self.intensity(readouts)
        dark = intensity < self.dark_threshold
        for quantile in np.arange(0.5, 0.04, -0.02):
            row = float(np.quantile(self.row, quantile))
            upper = self.row < row
            if dark[:, upper].sum(axis=1).max() <= margin:
                self.horizon_row = row
                self.upper = upper
                self.n_upper = int(upper.sum())
                col_upper = self.col[upper]
                lo, hi = col_upper.min(), col_upper.max()
                self._azimuth = np.zeros(self.n_ommatidia)
                self._azimuth[upper] = 2 * (col_upper - lo) / max(hi - lo, 1e-9) - 1
                return
        # Nothing was clean enough; keep the default and let the caller see it
        # in the sense-check plots rather than failing silently here.

    @staticmethod
    def intensity(readouts: np.ndarray) -> np.ndarray:
        """Collapse the pale/yellow channels to one intensity per ommatidium.

        Each ommatidium reports on exactly one of the two channels and zero on
        the other, so the maximum picks out the real reading.
        """
        return readouts.max(axis=2)

    def extract(self, readouts: np.ndarray) -> np.ndarray:
        """Feature vector of length `N_FEATURES` from ``(2, n_ommatidia, 2)``."""
        intensity = self.intensity(readouts)
        dark = (intensity < self.dark_threshold) & self.upper[None, :]

        mass = dark.sum(axis=1) / max(self.n_upper, 1)
        azimuth = np.zeros(2)
        for eye in (0, 1):
            n = dark[eye].sum()
            if n:
                azimuth[eye] = float(self._azimuth[dark[eye]].mean())

        return np.array(
            [
                mass[0],
                mass[1],
                azimuth[0],
                azimuth[1],
                mass[0] - mass[1],
                mass[0] + mass[1],
            ],
            dtype=np.float32,
        )

    def describe(self, features: np.ndarray) -> str:
        return "  ".join(f"{k}={v:+.3f}" for k, v in zip(FEATURE_NAMES, features))

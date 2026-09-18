"""Motion vision: T4/T5-like motion detectors on the compound eyes.

FlyGym's eyes give a brightness per ommatidium; nothing in `flyplay` saw motion
until this module. It sits beside the colour pathway (`flyplay.visual_pathway`)
and reads the same readouts, but it needs them often: a detector correlates
each ommatidium's signal with a delayed copy of its neighbour's, and the eyes'
10 Hz, fine for floor colours, would alias any real motion. The sandbox
samples the eyes every 10 ms while something in the room moves by itself
(`flyplay.room.MOVING_KINDS`), and not otherwise (see `flyplay.sandbox`).

    photoreceptors   each ommatidium's own channel (`photoreceptors`)
    high-pass        tau 250 ms, then split ON (brightening) and OFF (dimming)
    T4 / T5          per neighbouring pair a -> b:  LP(x_a) * x_b - x_a * LP(x_b)
                     with LP a 50 ms low-pass: the Hassenstein-Reichardt
                     correlator on the ON and on the OFF signals
    wide field       summed per eye over pairs along the lattice's two
                     near-horizontal axes: horizontal motion, as the
                     lobula plate's tangential cells pool it
    overhead         net dimming over the upper field, OFF minus ON: something
                     dark passing over the fly. Moving stripes brighten as many
                     ommatidia as they darken and read about zero

The filter constants are Borst (2018)'s three-arm model (centre high-pass
250 ms, flank low-pass 50 ms, 10 ms steps), whose correlator peaks near
1/(2 pi 50 ms) = 3.2 Hz, inside the 1-3 Hz of measured T4/T5 tuning (Creamer et
al. 2018). The hexagonal lattice has one vertical axis and two at +/-30 degrees
from horizontal (measured on FlyGym's ommatidia centroids: 690 neighbour pairs
at 90 degrees, 506-524 at +/-30 and +/-150), so horizontal motion is read from
the two diagonal axes together, whose vertical parts cancel.

Which way along the image columns is front or back differs between the eyes;
`ROTATION_SIGNS` fixes it from a measurement (see there). No connectome here:
flyvis (Lappalainen et al. 2024) would be, but it takes about 2 s of CPU per
network update, and 14 experiment workers share this machine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flyplay.visual_pathway import eye_layout, photoreceptors

#: High-pass time constant before the ON/OFF split, s (Borst 2018).
TAU_HIGH = 0.25
#: Delay low-pass of the correlator, s (Borst 2018).
TAU_LOW = 0.05
#: Neighbour pairs used for horizontal motion: lattice axes this close to the
#: image rows, degrees.
HORIZONTAL_AXES_DEG = 40.0
#: The upper field the overhead signal pools: ommatidia above this row quantile.
#: Measured on a still fly, net dimming (OFF minus ON): a shadow pass peaks at
#: 0.34 at the centre and 0.28-0.34 beside a wall; the drum turning at 60-240
#: deg/s stays under 0.0054 at the centre and 0.026 beside the wall facing it.
#: With OFF alone the drum's stripes, which fill the upper field near a wall,
#: read as shadows: flies by the turning drum froze and averaged 2-9 mm/s
#: against 13-16 by the still drum.
DORSAL_QUANTILE = 0.2
#: Sign per eye (left, right) that turns "motion toward higher image columns"
#: into "the world turning counterclockwise, seen from above". Measured on
#: rendered eyes, fly held still in the sandbox room, mean over 1-3 s: the drum
#: turning counterclockwise at 60 deg/s gave -1.30e-2 in both eyes, clockwise
#: +1.30e-2, and the fly itself turning left at 60 deg/s +1.3e-2 (the world
#: turns clockwise relative to it). Walking forward gave -1.75e-3 left and
#: +1.69e-3 right, which the sum cancels. Tuning over drum speed, 20-degree
#: period: 30 deg/s 0.90e-2, 60 1.30e-2, 120 1.10e-2, 240 0.64e-2.
ROTATION_SIGNS = (-1.0, -1.0)


def _pairs() -> tuple[np.ndarray, np.ndarray]:
    """Neighbouring ommatidia (a, b) along the near-horizontal lattice axes,
    oriented so b sits at a higher image column than a."""
    _, _, row, col = eye_layout()
    xy = np.stack([col, row], axis=1)
    distance = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    np.fill_diagonal(distance, np.inf)
    spacing = np.median(np.sort(distance, axis=1)[:, 0])
    a, b = np.nonzero(distance < 1.3 * spacing)
    angle = np.degrees(np.arctan2(row[b] - row[a], col[b] - col[a]))
    keep = np.abs(angle) < HORIZONTAL_AXES_DEG
    return a[keep], b[keep]


@dataclass(frozen=True)
class MotionState:
    """One sample of the motion pathway."""

    #: Horizontal motion per eye (left, right), toward higher image columns.
    horizontal: np.ndarray
    #: Wide-field rotation of the world, counterclockwise positive (arbitrary
    #: units; `flyplay.sandbox` scales it).
    rotation: float
    #: Net dimming over the upper field of both eyes (OFF minus ON, not below 0).
    overhead: float


class MotionVision:
    """Both eyes' motion detectors and their pooled outputs.

    Args:
        dt: Time between eye samples, s. The filters are exact first-order
            discretisations for it.
    """

    def __init__(self, dt: float, *, tau_high: float = TAU_HIGH, tau_low: float = TAU_LOW):
        self.dt = float(dt)
        self.alpha_high = 1.0 - np.exp(-self.dt / tau_high)
        self.alpha_low = 1.0 - np.exp(-self.dt / tau_low)
        self.a, self.b = _pairs()
        _, _, row, _ = eye_layout()
        self.dorsal = row <= np.quantile(row, DORSAL_QUANTILE)
        self.reset()

    def reset(self) -> None:
        """Forget the filters' history: the next sample only primes them."""
        self._slow = None
        self._on = self._off = None
        self._delayed_on = self._delayed_off = None

    def __call__(self, readouts: np.ndarray) -> MotionState:
        light = photoreceptors(np.asarray(readouts, dtype=float))  # (2, n)
        if self._slow is None:
            self._slow = light.copy()
            zeros = np.zeros_like(light)
            self._on, self._off = zeros, zeros.copy()
            self._delayed_on, self._delayed_off = zeros.copy(), zeros.copy()
            return MotionState(np.zeros(2), 0.0, 0.0)
        self._slow += self.alpha_high * (light - self._slow)
        change = light - self._slow
        on, off = np.maximum(change, 0.0), np.maximum(-change, 0.0)
        # Delayed copies of the previous sample's signals, then the correlator.
        self._delayed_on += self.alpha_low * (self._on - self._delayed_on)
        self._delayed_off += self.alpha_low * (self._off - self._delayed_off)
        a, b = self.a, self.b
        t4 = self._delayed_on[:, a] * on[:, b] - on[:, a] * self._delayed_on[:, b]
        t5 = self._delayed_off[:, a] * off[:, b] - off[:, a] * self._delayed_off[:, b]
        self._on, self._off = on, off
        horizontal = (t4 + t5).mean(axis=1)
        rotation = float(0.5 * (ROTATION_SIGNS[0] * horizontal[0] + ROTATION_SIGNS[1] * horizontal[1]))
        overhead = max(0.0, float((off - on)[:, self.dorsal].mean()))
        return MotionState(horizontal, rotation, overhead)

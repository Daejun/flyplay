"""Olfaction for FlyGym 2.x, which does not ship with any.

FlyGym 1.x had an `OdorArena`; the 2.x rewrite dropped it and has not brought
it back (the newest thread on the project's discussion board, from April 2026,
is someone asking when the 1.x tutorials will be ported). This module puts it
back on top of the 2.x API, using the same sensor placement and the same
inverse-square intensity model as the reference implementation, so numbers are
comparable with the official 1.x tutorials.

Four sensors, matching `flygym_gymnasium/config.yaml`:

    l/r maxillary palp   on `c_rostrum`     offset (-0.15, +/-0.15, -0.15) mm
    l/r antenna          on `l/r_funiculus` offset ( 0.02,    0.00 , -0.10) mm

Intensity at a sensor is summed over sources, per odour dimension:

    I[sensor, dim] = sum_source  peak[source, dim] * f(distance)

with ``f(d) = d**-2`` by default. The odour space is deliberately separate from
the source list: three physical sources can span a two-dimensional
(attractive, aversive) space, which is how the NeuroMechFly papers set up
approach-avoidance tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco as mj
import numpy as np
from flygym.anatomy import BodySegment

#: Sensor name -> (parent body segment, offset in the parent frame, marker colour).
#: Order matters: it is the row order of every array this module returns, and it
#: matches the 1.x config so the official tutorials' index arithmetic still works.
SENSOR_LAYOUT: tuple[tuple[str, str, tuple[float, float, float]], ...] = (
    ("l_maxillary_palp", "c_rostrum", (-0.15, 0.15, -0.15)),
    ("r_maxillary_palp", "c_rostrum", (-0.15, -0.15, -0.15)),
    ("l_antenna", "l_funiculus", (0.02, 0.0, -0.10)),
    ("r_antenna", "r_funiculus", (0.02, 0.0, -0.10)),
)
SENSOR_NAMES = tuple(name for name, _, _ in SENSOR_LAYOUT)
N_SENSORS = len(SENSOR_LAYOUT)

#: Palps are weighted above antennae when collapsing the four sensors to a
#: left/right pair, following the 1.x olfaction tutorial.
PALP_ANTENNA_WEIGHTS = (9.0, 1.0)


def add_odor_sensors(fly, *, prefix: str = "odorsensor", marker: bool = False) -> list:
    """Attach the four olfactory sensor sites to a fly, before it is compiled.

    Args:
        fly: A `flygym.compose.NeuroMechFly` that has not been added to a world
            yet (sites have to exist before `Simulation` compiles the model).
        prefix: Site name prefix. The compiled site name is
            ``f"{prefix}_{sensor}"``, which is what `OdorField` looks up.
        marker: Draw a small sphere at each sensor. Useful for checking
            placement; the markers go in geom group 1, which the eye cameras
            ignore, so the fly cannot see its own sensors.

    Returns:
        The created MJCF site elements, in `SENSOR_LAYOUT` order. They are also
        stored on the fly as ``odorsensorname_to_mjcfsite``, mirroring how
        FlyGym keeps ``cameraname_to_mjcfcamera``.

    Note:
        Attaching the fly to a world renames its elements in place, prefixing
        them with the fly's name (``odorsensor_l_antenna`` becomes
        ``nmf/odorsensor_l_antenna``). That is why the elements are kept and
        re-read after compilation rather than the names being rebuilt.
    """
    sites = []
    lookup = {}
    for name, parent, offset in SENSOR_LAYOUT:
        body = fly.bodyseg_to_mjcfbody[BodySegment(parent)]
        site = body.add_site(name=f"{prefix}_{name}", pos=offset, group=1)
        sites.append(site)
        lookup[name] = site
        if marker:
            body.add_geom(
                name=f"{prefix}_{name}_marker",
                type=mj.mjtGeom.mjGEOM_SPHERE,
                size=(0.05, 0, 0),
                pos=offset,
                rgba=(0.08, 0.4, 0.9, 1.0) if "antenna" in name else (0.9, 0.73, 0.08, 1.0),
                contype=0,
                conaffinity=0,
                group=1,
            )
    fly.odorsensorname_to_mjcfsite = lookup
    return sites


@dataclass
class OdorSource:
    """One point source of odour.

    Attributes:
        pos: Position in world coordinates, in mm.
        peak: Peak intensity per odour dimension at zero distance.
        rgba: Colour of the visualisation marker, if one is drawn.
    """

    pos: tuple[float, float, float]
    peak: tuple[float, ...]
    rgba: tuple[float, float, float, float] = (1.0, 0.5, 0.05, 1.0)


@dataclass
class OdorField:
    """The odour sources in a scene, and how they are sensed.

    Args:
        sources: The point sources.
        min_distance: Distance floor, in mm, applied before the falloff. The
            1.x implementation uses a bare ``d**-2``, which diverges as the fly
            puts its head on the source; clamping keeps the observation bounded
            without changing anything at the distances that matter (>1 mm).
        exponent: Falloff exponent. -2 is the reference inverse-square law.
    """

    sources: list[OdorSource] = field(default_factory=list)
    min_distance: float = 0.5
    exponent: float = -2.0

    def __post_init__(self) -> None:
        self._site_ids: np.ndarray | None = None

    # --- scene setup ----------------------------------------------------

    @property
    def n_dimensions(self) -> int:
        return len(self.sources[0].peak) if self.sources else 0

    @property
    def positions(self) -> np.ndarray:
        """Source positions, shape ``(n_sources, 3)``."""
        return np.array([s.pos for s in self.sources], dtype=float).reshape(-1, 3)

    @property
    def peaks(self) -> np.ndarray:
        """Peak intensities, shape ``(n_sources, n_dimensions)``."""
        return np.array([s.peak for s in self.sources], dtype=float).reshape(
            len(self.sources), -1
        )

    def source_position(self, dim: int = 0) -> np.ndarray:
        """Position of the strongest source in odour dimension `dim`."""
        return self.positions[int(np.argmax(self.peaks[:, dim]))]

    # --- reading --------------------------------------------------------

    def bind(self, sim, fly) -> None:
        """Resolve sensor site ids against a compiled model.

        `Simulation.get_site_positions` only covers the anatomical-joint sites
        that `add_joint_sites` creates, so the odour sites are looked up here.
        Names are taken from the stored MJCF elements because attaching the fly
        rewrites them with the fly's prefix.
        """
        lookup = getattr(fly, "odorsensorname_to_mjcfsite", None)
        if not lookup:
            raise ValueError(
                "This fly has no odour sensors. Call add_odor_sensors(fly) "
                "before world.add_fly()."
            )
        ids = []
        for name in SENSOR_NAMES:
            site_name = lookup[name].name
            site_id = mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_SITE, site_name)
            if site_id < 0:
                raise ValueError(
                    f"Odour sensor site '{site_name}' is not in the compiled model."
                )
            ids.append(site_id)
        self._site_ids = np.array(ids, dtype=np.int32)

    def sensor_positions(self, sim) -> np.ndarray:
        """World positions of the four sensors, shape ``(4, 3)``."""
        if self._site_ids is None:
            raise ValueError("Call OdorField.bind(sim, fly) first.")
        return sim.mj_data.site_xpos[self._site_ids]

    def read(self, sim) -> np.ndarray:
        """Odour intensity at each sensor.

        Returns:
            Shape ``(4, n_dimensions)``, rows in `SENSOR_NAMES` order.
        """
        if not self.sources:
            return np.zeros((N_SENSORS, max(self.n_dimensions, 1)))
        sensors = self.sensor_positions(sim)  # (4, 3)
        delta = sensors[:, None, :] - self.positions[None, :, :]  # (4, n_src, 3)
        distance = np.linalg.norm(delta, axis=-1)  # (4, n_src)
        falloff = np.maximum(distance, self.min_distance) ** self.exponent
        return falloff @ self.peaks  # (4, n_dim)

    @staticmethod
    def left_right(intensities: np.ndarray) -> np.ndarray:
        """Collapse the four sensors to a left/right pair per odour dimension.

        Args:
            intensities: ``(4, n_dimensions)`` from `read`.

        Returns:
            Shape ``(2, n_dimensions)`` -- row 0 left, row 1 right.
        """
        palps, antennae = intensities[:2], intensities[2:]
        w_palp, w_ant = PALP_ANTENNA_WEIGHTS
        total = w_palp + w_ant
        return (w_palp * palps + w_ant * antennae) / total

    @classmethod
    def asymmetry(cls, intensities: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        """Normalised left-right difference per odour dimension.

        ``(I_left - I_right) / mean(I_left, I_right)`` -- the quantity the
        hand-written chemotaxis controller in the NeuroMechFly papers steers on.
        Positive means the odour is stronger on the fly's left.
        """
        lr = cls.left_right(intensities)
        mean = lr.mean(axis=0)
        return (lr[0] - lr[1]) / np.maximum(mean, eps)

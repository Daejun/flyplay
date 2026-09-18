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

**Walls.** `OdorField` measures `d` in a straight line, so odour passes through
anything in between. `WalledOdorField` measures it along the shortest path that
goes round the room's solids instead (see there): the same field wherever a
source is in plain view of the sensor, weaker behind a wall, and pointing round
the wall's end rather than into it.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable
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


# --- odour round walls ---------------------------------------------------------------

#: Added to every solid's half-extents when testing whether a straight line is
#: blocked, mm. Two walls drawn edge to edge leave a gap of zero width that a
#: line could slip through; grown by this much they overlap.
SOLID_MARGIN = 0.05
#: How far outside a solid's corner, diagonally, a path turns round it, mm.
#: Beyond `SOLID_MARGIN`, so a path along a wall's face does not touch it.
CORNER_CLEARANCE = 0.15


def _blocked(starts: np.ndarray, ends: np.ndarray, rects: np.ndarray) -> np.ndarray:
    """Whether each segment ``starts[i] -> ends[i]`` crosses the inside of any rect.

    Args:
        starts, ends: ``(..., 2)`` segment end points, broadcast against each other.
        rects: ``(n, 4)`` rows of centre x, centre y, half x, half y.

    Returns:
        Boolean array of the broadcast leading shape. A rect holding either end
        of a segment does not block it: a source placed on top of a block smells
        from the block, and a sensor pressed against a wall is still in the room.
    """
    p = np.asarray(starts, dtype=float)[..., None, :]
    q = np.asarray(ends, dtype=float)[..., None, :]
    centre, half = rects[:, :2], rects[:, 2:]
    d = q - p
    with np.errstate(divide="ignore", invalid="ignore"):
        ta = (centre - half - p) / d
        tb = (centre + half - p) / d
    # A segment parallel to an axis is inside that slab everywhere or nowhere.
    parallel = np.abs(d) < 1e-12
    inside_slab = np.abs(p - centre) < half
    t_near = np.where(parallel, np.where(inside_slab, -np.inf, np.inf), np.minimum(ta, tb))
    t_far = np.where(parallel, np.where(inside_slab, np.inf, -np.inf), np.maximum(ta, tb))
    enter = np.maximum(t_near.max(axis=-1), 0.0)
    leave = np.minimum(t_far.min(axis=-1), 1.0)
    holds = (np.all(np.abs(p - centre) < half, axis=-1) | np.all(np.abs(q - centre) < half, axis=-1))
    return np.any((enter < leave) & ~holds, axis=-1)


class PathGraph:
    """Shortest paths round axis-aligned rectangles in the floor plane.

    A shortest path between two points past convex obstacles bends only at
    obstacle corners, so the corners (moved `CORNER_CLEARANCE` outward) are the
    graph's nodes and a straight line that crosses no rectangle is an edge.
    Built once per layout; `lengths` then answers for any points.

    Args:
        rects: ``(n, 4)`` centre x, centre y, half x, half y of every solid.
        half: The room's interior half-width. Corners outside it, where an
            obstacle meets the room's own wall, are not ways round.
    """

    def __init__(self, rects: np.ndarray, half: float):
        base = np.asarray(rects, dtype=float).reshape(-1, 4)
        self.rects = base + np.array([0.0, 0.0, SOLID_MARGIN, SOLID_MARGIN])
        signs = np.array([(-1, -1), (-1, 1), (1, -1), (1, 1)], dtype=float)
        corners = (base[:, None, :2] + signs[None] * (base[:, None, 2:] + CORNER_CLEARANCE)).reshape(-1, 2)
        inside_any = np.any(np.all(np.abs(corners[:, None, :] - self.rects[None, :, :2])
                                   < self.rects[None, :, 2:], axis=-1), axis=1) if len(base) else np.zeros(0, bool)
        in_room = np.all(np.abs(corners) < half, axis=1)
        self.nodes = corners[in_room & ~inside_any]
        n = len(self.nodes)
        # Node-to-node edges.
        self.edges = np.full((n, n), np.inf)
        if n:
            free = ~_blocked(self.nodes[:, None, :], self.nodes[None, :, :], self.rects)
            length = np.linalg.norm(self.nodes[:, None, :] - self.nodes[None, :, :], axis=-1)
            self.edges = np.where(free, length, np.inf)
        self._from_source: dict[tuple[float, float], np.ndarray] = {}

    def distances_from(self, source: np.ndarray) -> np.ndarray:
        """Shortest path length from `source` to every node (inf if walled off)."""
        key = (float(source[0]), float(source[1]))
        cached = self._from_source.get(key)
        if cached is not None:
            return cached
        n = len(self.nodes)
        dist = np.full(n, np.inf)
        if n:
            free = ~_blocked(np.broadcast_to(source, (n, 2)), self.nodes, self.rects)
            dist[free] = np.linalg.norm(self.nodes[free] - source, axis=1)
            heap = [(d, i) for i, d in enumerate(dist) if np.isfinite(d)]
            heapq.heapify(heap)
            done = np.zeros(n, bool)
            while heap:
                d, i = heapq.heappop(heap)
                if done[i]:
                    continue
                done[i] = True
                better = d + self.edges[i] < dist
                for j in np.flatnonzero(better & ~done):
                    dist[j] = d + self.edges[i, j]
                    heapq.heappush(heap, (dist[j], j))
        if len(self._from_source) >= 64:
            # A source dragged across the map leaves a trail of positions.
            self._from_source.clear()
        self._from_source[key] = dist
        return dist

    def lengths(self, points: np.ndarray, sources: np.ndarray, blocked: np.ndarray) -> np.ndarray:
        """Path lengths ``(n_points, n_sources)`` for the pairs `blocked` marks
        (the others are not computed and read inf)."""
        out = np.full(blocked.shape, np.inf)
        if not len(self.nodes):
            return out
        seen = ~_blocked(points[:, None, :], self.nodes[None, :, :], self.rects)
        leg = np.where(seen, np.linalg.norm(points[:, None, :] - self.nodes[None, :, :], axis=-1), np.inf)
        for j in np.flatnonzero(blocked.any(axis=0)):
            total = leg + self.distances_from(sources[j])[None, :]
            rows = np.flatnonzero(blocked[:, j])
            out[rows, j] = total[rows].min(axis=1)
        return out


@dataclass
class WalledOdorField(OdorField):
    """`OdorField` in a room with solids: odour goes round them, not through.

    Intensity keeps the inverse-square law, but `d` is the length of the
    shortest path from source to sensor that stays out of every solid
    (`PathGraph`), with the height difference added in quadrature. Wherever a
    source is in plain view of a sensor this is the straight line and the
    reading is bit-identical to `OdorField`'s, so an open room, and every
    measurement calibrated in one (the Kenyon-cell working range, the steering
    asymmetries), is unchanged. Behind a wall the reading is weaker and its
    left-right difference points round the wall's end.

    This is geometry standing in for physics. Still air carries odour round a
    wall by diffusion, and the steady state of that is not exactly inverse
    square in the path length; PLAN.md B-1's grid solution of the diffusion
    equation would be the physical version. It would also change the field in
    the open room, which is what every calibration here was measured in -- the
    reason this came first. A room with no way round (a closed box) reads zero.

    Args:
        solids: Called on every read; returns the solids as objects with
            ``cx, cy, hx, hy`` (`flyplay.room.Rect`). The graph is rebuilt only
            when they change.
        room_half: Interior half-width of the room. Sources outside it -- parked
            pool slots -- are read in a straight line, as before.
    """

    solids: Callable[[], list] | None = None
    room_half: float = 50.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self._graph: PathGraph | None = None
        self._graph_key: tuple | None = None

    def graph(self) -> PathGraph | None:
        rects = self.solids() if self.solids is not None else []
        if not rects:
            return None
        key = tuple((r.cx, r.cy, r.hx, r.hy) for r in rects)
        if key != self._graph_key:
            self._graph = PathGraph(np.array(key), self.room_half)
            self._graph_key = key
        return self._graph

    def read(self, sim) -> np.ndarray:
        if not self.sources:
            return np.zeros((N_SENSORS, max(self.n_dimensions, 1)))
        sensors = self.sensor_positions(sim)  # (4, 3)
        positions = self.positions
        delta = sensors[:, None, :] - positions[None, :, :]  # (4, n_src, 3)
        distance = np.linalg.norm(delta, axis=-1)  # (4, n_src)
        graph = self.graph()
        if graph is not None:
            distance = self._round_walls(graph, sensors, positions, distance)
        falloff = np.maximum(distance, self.min_distance) ** self.exponent
        return falloff @ self.peaks  # (4, n_dim)

    def path_distance(self, points: np.ndarray, positions: np.ndarray, distance: np.ndarray) -> np.ndarray:
        """`distance` ``(n_points, n_src)`` with every pair a solid hides replaced
        by the path round it. Points ``(n_points, 3)``, source positions
        ``(n_src, 3)``. For maps and checks as well as `read`."""
        graph = self.graph()
        return distance if graph is None else self._round_walls(graph, points, positions, distance)

    def _round_walls(self, graph: PathGraph, sensors, positions, distance) -> np.ndarray:
        in_room = np.all(np.abs(positions[:, :2]) <= self.room_half, axis=1)
        if not in_room.any():
            return distance
        which = np.flatnonzero(in_room)
        points, sources = sensors[:, :2], positions[which, :2]
        blocked = _blocked(points[:, None, :], sources[None, :, :], graph.rects)
        if not blocked.any():
            return distance
        around = graph.lengths(points, sources, blocked)
        rise = sensors[:, None, 2] - positions[None, which, 2]
        out = distance.copy()
        sub = out[:, which]
        sub[blocked] = np.sqrt(around[blocked] ** 2 + rise[blocked] ** 2)
        out[:, which] = sub
        return out

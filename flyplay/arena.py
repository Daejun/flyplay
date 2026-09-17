"""Props: obstacles the fly can see and bump into, markers it cannot.

Two facts about FlyGym 2.x drive this module, both verified by experiment:

- **Nothing collides implicitly.** Every geom on the fly carries
  ``contype=0, conaffinity=0``; contact comes from explicit ``<pair>`` elements
  that `_GroundContactMixin` generates between the fly's contact geoms and
  everything in ``world.ground_geoms``. So an obstacle becomes solid by being
  appended to that list -- and it has to happen *before* ``world.add_fly()``,
  because that is when the pairs are made.
- **Geom groups 1 and 2 are invisible to the fly.** The eye renderer switches
  those groups off to stop the fly seeing its own eye markers. A visualisation
  marker in group 2 therefore shows up in recorded video and in the web viewer
  but not in the ommatidia -- which is what the NeuroMechFly paper means when
  it says the odour marker "is not visible to the simulated fly".
"""

from __future__ import annotations

import mujoco as mj
import numpy as np
from flygym.utils.mjcf import GEOM_TYPES

#: Dark enough to stand out against the sky in the fly's upper visual field.
PILLAR_RGBA = (0.08, 0.08, 0.10, 1.0)


def add_pillar(
    world,
    pos: tuple[float, float],
    *,
    radius: float = 0.8,
    height: float = 4.0,
    name: str | None = None,
    rgba: tuple[float, float, float, float] = PILLAR_RGBA,
):
    """Add a visible, solid pillar. Call before `world.add_fly()`.

    Args:
        world: The world under construction.
        pos: ``(x, y)`` position in mm.
        radius: Pillar radius in mm.
        height: Full height in mm; the pillar stands on z = 0.
        name: Geom name. Defaults to ``pillar_<n>``.
        rgba: Colour. Keep it dark -- the visual features key on contrast
            against the sky.

    Returns:
        The created MJCF geom element.
    """
    existing = getattr(world, "_flyplay_pillars", [])
    name = name or f"pillar_{len(existing)}"
    geom = world.mjcf_root.worldbody.add_geom(
        type=GEOM_TYPES["cylinder"],
        name=name,
        size=(radius, height / 2, 0),
        pos=(pos[0], pos[1], height / 2),
        rgba=rgba,
        contype=0,  # collision comes from the generated contact pairs
        conaffinity=0,
    )
    # Group 0 by default, so the eye cameras see it.
    world.ground_geoms.append(geom)
    existing.append(geom)
    world._flyplay_pillars = existing
    return geom


def add_odor_marker(
    world,
    pos: tuple[float, float, float],
    *,
    radius: float = 0.5,
    name: str | None = None,
    rgba: tuple[float, float, float, float] = (1.0, 0.5, 0.05, 1.0),
):
    """Add a marker showing where an odour source is. The fly cannot see it.

    Group 2 keeps it out of the ommatidia while leaving it in ordinary
    renders, and it is not added to ``ground_geoms`` so it stays intangible.
    """
    existing = getattr(world, "_flyplay_markers", [])
    name = name or f"odor_marker_{len(existing)}"
    geom = world.mjcf_root.worldbody.add_geom(
        type=GEOM_TYPES["sphere"],
        name=name,
        size=(radius, 0, 0),
        pos=tuple(pos),
        rgba=rgba,
        contype=0,
        conaffinity=0,
        group=2,
    )
    existing.append(geom)
    world._flyplay_markers = existing
    return geom


#: Paint for visual-only floor tiles, (r, g, b, a). Lower-field eye readings
#: for the first four are measured (see `flyplay.visual_pathway.FLOOR_READINGS`):
#: blue and green come out nearly equally bright to the model (0.271 / 0.273).
FLOOR_COLOURS: dict[str, tuple[float, float, float, float]] = {
    "blue": (0.05, 0.05, 0.95, 1.0),
    "green": (0.05, 0.95, 0.05, 1.0),
    "dim_blue": (0.02, 0.02, 0.40, 1.0),
    "dim_green": (0.02, 0.40, 0.02, 1.0),
    # A tenth of "green": the 10:1 brightness task of Vogt et al. (2016).
    "green_tenth": (0.005, 0.095, 0.005, 1.0),
    "grey": (0.5, 0.5, 0.5, 1.0),
}
#: Height of the tiles' top above the ground plane, in mm. Measured: plates this
#: high render without z-fighting and are read by the eyes; the feet pass
#: through them, since tiles are not in `ground_geoms`.
FLOOR_TILE_Z = 0.02


def add_floor_tiles(world, *, tile_size: float, n_side: int) -> list:
    """Lay an ``n_side x n_side`` grid of visual-only tiles, centred on the origin.

    Group 0, so the eyes see them; not in ``ground_geoms``, so nothing collides
    with them. Call before `world.add_fly()` like any prop. `ColourFloor`
    paints and moves them at runtime.
    """
    tiles = []
    half = tile_size / 2
    for i in range(n_side):
        for j in range(n_side):
            x = (i - (n_side - 1) / 2) * tile_size
            y = (j - (n_side - 1) / 2) * tile_size
            tiles.append(world.mjcf_root.worldbody.add_geom(
                type=GEOM_TYPES["box"],
                name=f"floor_tile_{i}_{j}",
                size=(half, half, 0.005),
                pos=(x, y, FLOOR_TILE_Z - 0.005),
                rgba=FLOOR_COLOURS["grey"],
                contype=0,
                conaffinity=0,
            ))
    world._flyplay_tiles = tiles
    world._flyplay_tile_grid = (float(tile_size), int(n_side))
    return tiles


class ColourFloor:
    """Paint the floor tiles and keep them under a walking fly.

    A test lasts 90 s and the fly covers over a metre in that time, far more
    than any grid of tiles. The grid is therefore recentred under the fly in
    steps of one checker period (two tiles), which leaves the pattern at every
    world position exactly as it was: the fly cannot tell, and `colour_at`
    keeps answering from the pattern rather than from the geoms.

    Tile edges sit on multiples of `tile_size` for an even `n_side`, so the
    origin -- where the fly spawns -- is a corner shared by four tiles.
    """

    def __init__(self, sim, world):
        tile_size, n_side = world._flyplay_tile_grid
        if n_side % 2:
            raise ValueError("ColourFloor needs an even n_side so the origin is a tile corner.")
        self.sim = sim
        self.tile_size = tile_size
        self.n_side = n_side
        self.ids = _geom_ids(sim, world._flyplay_tiles).reshape(n_side, n_side)
        # Offsets of each tile centre from the grid centre.
        steps = (np.arange(n_side) - (n_side - 1) / 2) * tile_size
        self._offset_x, self._offset_y = np.meshgrid(steps, steps, indexing="ij")
        self.centre = np.zeros(2)
        self.pattern: tuple = ("uniform", "grey")
        self.paint_uniform("grey")

    # --- painting ---------------------------------------------------------

    def paint_uniform(self, colour: str) -> None:
        """Every tile one colour: a training trial in Vogt et al. (2014)."""
        self.pattern = ("uniform", colour)
        self.sim.mj_model.geom_rgba[self.ids.ravel()] = FLOOR_COLOURS[colour]

    def paint_checker(self, even: str, odd: str) -> None:
        """Alternate two colours, so diagonal neighbours match: the quadrant
        test of Vogt et al. (2014), repeated over the whole floor."""
        self.pattern = ("checker", even, odd)
        self._repaint()

    def _repaint(self) -> None:
        if self.pattern[0] != "checker":
            return
        _, even, odd = self.pattern
        xy = self._tile_centres()
        parity = self._parity(xy[..., 0], xy[..., 1])
        rgba = self.sim.mj_model.geom_rgba
        rgba[self.ids[parity == 0]] = FLOOR_COLOURS[even]
        rgba[self.ids[parity == 1]] = FLOOR_COLOURS[odd]

    def _tile_centres(self) -> np.ndarray:
        return np.stack([self.centre[0] + self._offset_x, self.centre[1] + self._offset_y], axis=-1)

    def _parity(self, x, y):
        return (np.floor(np.asarray(x) / self.tile_size).astype(int)
                + np.floor(np.asarray(y) / self.tile_size).astype(int)) % 2

    # --- following the fly ------------------------------------------------

    def follow(self, xy) -> None:
        """Recentre the grid under `xy` if the fly has walked a period away."""
        period = 2 * self.tile_size
        shift = np.round((np.asarray(xy, dtype=float) - self.centre) / period) * period
        if not shift.any():
            return
        self.centre = self.centre + shift
        centres = self._tile_centres()
        pos = self.sim.mj_model.geom_pos
        pos[self.ids, 0] = centres[..., 0]
        pos[self.ids, 1] = centres[..., 1]
        self._repaint()

    def reset(self) -> None:
        """Back to the origin, for a new trial."""
        self.follow_to(np.zeros(2))

    def follow_to(self, xy) -> None:
        self.centre = np.asarray(xy, dtype=float)
        centres = self._tile_centres()
        pos = self.sim.mj_model.geom_pos
        pos[self.ids, 0] = centres[..., 0]
        pos[self.ids, 1] = centres[..., 1]
        self._repaint()

    # --- reading ----------------------------------------------------------

    def colour_at(self, xy) -> str:
        """The colour painted at a world position, from the pattern."""
        kind = self.pattern[0]
        if kind == "uniform":
            return self.pattern[1]
        parity = int(self._parity(xy[0], xy[1]))
        return self.pattern[1 + parity]


def pillar_geom_ids(sim, world) -> np.ndarray:
    """Compiled geom ids of the pillars, for collision checks and moving them."""
    return _geom_ids(sim, getattr(world, "_flyplay_pillars", []))


def marker_geom_ids(sim, world) -> np.ndarray:
    """Compiled geom ids of the odour markers, so they can follow their source."""
    return _geom_ids(sim, getattr(world, "_flyplay_markers", []))


def _geom_ids(sim, elements) -> np.ndarray:
    return np.array(
        [
            mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_GEOM, g.name)
            for g in elements
        ],
        dtype=np.int32,
    )


def move_geom(sim, geom_id: int, pos) -> None:
    """Move a world-attached geom at runtime.

    Static geoms live in the model, not the data, so this writes ``geom_pos``
    and lets the next ``mj_forward`` refresh ``geom_xpos``. Verified to move
    both what the fly sees and what it collides with, which is what lets one
    compiled scene serve a whole training run with a new layout every episode.
    """
    sim.mj_model.geom_pos[geom_id] = pos


def ring_positions(
    n: int, radius: float, *, start_deg: float = 0.0, center=(0.0, 0.0)
) -> list[tuple[float, float]]:
    """`n` positions evenly spaced on a circle. Handy for scripted scenes."""
    angles = np.deg2rad(start_deg) + np.arange(n) * 2 * np.pi / max(n, 1)
    return [
        (center[0] + radius * float(np.cos(a)), center[1] + radius * float(np.sin(a)))
        for a in angles
    ]

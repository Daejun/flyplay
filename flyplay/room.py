"""A sealed square room whose contents can be rearranged while the fly walks.

Everything the user can put in the room -- obstacle blocks, sugar drops, shock
zones, coloured floor patches, odour sources -- is built once, before the model
is compiled, as a *pool* of geoms parked far outside the room. Placing an item
moves a free geom from its pool into the room; removing it parks the geom again.
Nothing is ever rebuilt, which is what makes live editing possible: compiling
this model takes seconds, moving a geom takes nothing. The same trick already
carries `ForageEnv`'s pillars and `ColourFloor`'s tiles, and was verified there
to move what the fly sees and what it collides with.

Two measured facts about this body shape the module:

**A wall contact flips the fly.** Walking into a 4 or 8 mm wall head-on or at 45
degrees, it climbed and flipped within 0.35-2.3 s; wall friction 0.1 or 0.01 did
not prevent it, only a 75-degree graze slid along. Walls and obstacles are
therefore solid (a safety net) but the fly is kept off them by
`ProximityReflex`, which turns before contact. Real flies touch walls with their
antennae and forelegs and follow them; this body cannot survive the touch, so
the reflex reads distance instead. It is a stand-in for the body's limits, not a
model of any fly behaviour.

**The tightest turn has a 4.7 mm radius** (descending signal [0.4, 1.6], 228
deg/s at 18.6 mm/s). The reflex starts turning when anything is within reach of
that arc, and corridors narrower than about twice it cannot be turned around in.

Geometry is 2-D and axis-aligned: every solid is a rectangle in the floor plane,
which keeps ray casting exact and cheap.

**Two things move by themselves**, for motion vision (`flyplay.motion_vision`):
a striped drum turning round the outside of the room -- the optomotor stimulus
-- and a dark bar sweeping over it, the passing shadow of Gibson et al. (2015).
Each is one mocap body built with the room and parked under the floor until
placed (`MOVING_KINDS`); they are seen, never touched or smelled.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco as mj
import numpy as np
from flygym.utils.mjcf import GEOM_TYPES

from flyplay.arena import FLOOR_COLOURS

#: Interior half-width of the room, mm: a 100 mm square. Open-field arenas for
#: walking flies are 84-118 mm across (Soibam et al. 2012; Corfas et al. 2019).
ROOM_HALF = 50.0
WALL_HEIGHT = 6.0
WALL_THICKNESS = 2.0
#: Where unused pool geoms wait, far outside the room and out of every sense.
PARK_ORIGIN = (900.0, 900.0)

#: Items that move by themselves, one of each at most (module docstring).
MOVING_KINDS = ("drum", "shadow")
#: The drum: stripes on a cylinder round the room, outside its walls (inner
#: faces at 50 mm) and taller, so they fill the fly's view above them: from the
#: centre the drum spans 0-30 degrees of elevation, the walls 0-7.
DRUM_RADIUS = 70.0
DRUM_HEIGHT = 40.0
#: 10-degree stripes: dark and clear alternate into a 20-degree period, near
#: the 20-22 degree wavelengths optomotor studies use (Seelig 2010, Mano 2023).
DRUM_STRIPES = 36
DRUM_RGBA = (0.04, 0.04, 0.05, 1.0)
#: The drum's stripes darkened in its "landmarks" pattern (10 degrees each).
LANDMARK_STRIPES = (*range(0, 9), 18, 25, 26, 27)
#: The shadow: a dark bar crossing the room overhead, above the walls. Gibson
#: et al. (2015) swept a paddle over a 100 mm arena at about 4.2 rad/s; this bar
#: crosses at `SHADOW_SPEED`, one pass taking about half a second.
SHADOW_HEIGHT = 30.0
SHADOW_LENGTH = 140.0
SHADOW_WIDTH = 20.0
SHADOW_SPEED = 320.0
SHADOW_RGBA = (0.02, 0.02, 0.02, 1.0)
#: Where the moving stimuli wait: under the floor plane, out of every render.
MOVING_PARK_Z = -300.0

#: Pool sizes: how many of each item can be in the room at once. Doubling the
#: obstacles from 8 to 16 took the contact pairs from 468 to 756 and left a
#: sandbox step at 23.2 ms against 23.0 (T-maze, measured): parked geoms are
#: culled by their bounds. The mazes among the presets need more than 8 walls.
POOL_SIZES = {"obstacle": 16, "sugar": 12, "shock": 8, "patch": 12, "odour": 8, "heat": 8, "light": 4}
#: Floor zones sensed through the legs or read by dopamine neurons directly,
#: never seen: shock plates, heated floor, the light of an optogenetic zone.
ZONE_KINDS = ("shock", "patch", "heat", "light")
#: Default half-extent of an obstacle block, mm. Obstacles are resizable boxes
#: -- a block or a long thin wall -- which means their bounding sphere
#: (`geom_rbound`) and box (`geom_aabb`) must follow every resize: contacts come
#: from explicit pairs culled by those bounds, and a stale bound silently drops
#: them (see `Room._apply_params` and the measurement there).
OBSTACLE_HALF = 4.0
#: How near the thorax a solid has to be for its bounding sphere to be its real
#: one; beyond that the sphere shrinks to `FAR_RBOUND` and MuJoCo's broad phase
#: drops every pair with it (`Room.gate_bounds`). The room's walls are 104 mm
#: long, so their spheres reach across the whole floor and every leg is tested
#: against every wall on every step: collision detection was 313.5 ms of each
#: simulated second, about two thirds of it walls. Measured over 20 s, with the
#: gate against without, `qpos` fingerprints identical in each: `t_maze` 1469 ->
#: 1039 ms of wall time per simulated second, `scented_sugar` 1206 -> 982,
#: `obstacle_field` 1257 -> 1059, `corridor` 1328 -> 1186.
#:
#: 8 mm is the margin: the fly reaches 3.0 mm from its thorax to a tarsus tip
#: and moves under 0.3 mm in the 10 ms between updates, so a solid that is
#: gated off cannot be touched before the next update. In `corridor` and
#: `t_maze` the thorax came within 1.2 and 0.8 mm of a solid and the gate was
#: open there.
GATE_MM = 8.0
#: The bounding radius of a gated-off solid. **Not 0**: MuJoCo reads rbound 0 as
#: "this geom is a plane" and stops filtering it at all.
FAR_RBOUND = 1e-6
#: Thinnest and longest an obstacle may be, as half-extents in mm. A wall is 2 mm
#: thick like the room's own.
OBSTACLE_MIN_HALF = 1.0
OBSTACLE_MAX_HALF = ROOM_HALF
#: Largest sugar drop radius, mm (a 500 nl drop; see `flyplay.sandbox`).
SUGAR_MAX_RADIUS = 6.0
#: Default half-extent of a shock zone or colour patch, mm. These are visual
#: only, so they can be resized live.
ZONE_HALF = 8.0
SUGAR_RADIUS = 2.5

OBSTACLE_RGBA = (0.42, 0.42, 0.45, 1.0)
WALL_RGBA = (0.55, 0.55, 0.58, 1.0)
#: Shock zones are invisible in every render, the fly's and the viewer's: a
#: coloured plate in the 3-D view read as a floor colour, which is exactly the
#: kind of cue the fly *can* see (user request). The page's map outlines them.
#: A cue for the fly has to be a colour patch placed with the zone.
SHOCK_RGBA = (1.0, 0.25, 0.1, 0.0)
#: Odour markers, likewise invisible to the fly.
ODOUR_MARKER_RGBA = (0.95, 0.75, 0.2, 1.0)
SUGAR_MARKER_RGBA = (0.97, 0.97, 0.9, 1.0)

#: The room's own floor, drawn over FlyGym's grey checker. Dark on purpose: the
#: floor is in view whenever dopamine arrives, so its visual code gets
#: associated with everything, and a colour sharing cells with it inherits
#: that. Measured over 16 wiring seeds, the default grey (reading 0.179) shares
#: about 1 of 5 active cells with blue or green (worst seed cosine 0.79) -- a
#: blue patch the fly never visited picked up +0.66 valence from a sugar meal.
#: At readings of 0.06 and below the overlap is zero on every seed; this paint
#: reads about 0.03.
ROOM_FLOOR_RGBA = (0.06, 0.06, 0.065, 1.0)
#: Heights. Patches sit where `flyplay.arena.ColourFloor` tiles do (top 0.02 mm,
#: measured to render cleanly) above the room floor (top 0.01); shock plates and
#: sugar just above, in group 2. Odour markers float above the fly: at 1.5 mm
#: the 0.8 mm ball sat at head height and hid the head of a fly feeding at its
#: drop. Only the marker moves; the smell source stays at 1.5 (`flyplay.sandbox`).
FLOOR_Z, PATCH_Z, SHOCK_Z, SUGAR_Z, ODOUR_Z = 0.005, 0.015, 0.03, 0.045, 3.5
HEAT_Z, LIGHT_Z = 0.035, 0.04
#: Heat and light plates are invisible in every render, like shock plates: the
#: fly feels heat and cannot see the red light CsChrimson is driven with. The
#: viewer outlines them.
ZONE_HIDDEN_RGBA = (1.0, 0.5, 0.1, 0.0)


@dataclass(frozen=True)
class Rect:
    """An axis-aligned rectangle in the floor plane."""

    cx: float
    cy: float
    hx: float
    hy: float

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return abs(x - self.cx) <= self.hx + margin and abs(y - self.cy) <= self.hy + margin

    def ray(self, ox: float, oy: float, dx: float, dy: float) -> float:
        """Distance along a unit ray to this rectangle's boundary; inf if missed.

        Slab method. A ray starting inside returns the distance to the exit,
        which is what a fly half inside a wall's footprint needs to get out.
        """
        t_near, t_far = -np.inf, np.inf
        for o, d, c, h in ((ox, dx, self.cx, self.hx), (oy, dy, self.cy, self.hy)):
            if abs(d) < 1e-12:
                if abs(o - c) > h:
                    return np.inf
                continue
            t1, t2 = (c - h - o) / d, (c + h - o) / d
            t_near, t_far = max(t_near, min(t1, t2)), min(t_far, max(t1, t2))
        if t_far < max(t_near, 0.0):
            return np.inf
        return t_near if t_near >= 0.0 else t_far


def cast_rays(origin, yaw: float, angles: np.ndarray, rects, max_range: float) -> np.ndarray:
    """Distance to the nearest solid along each ray, capped at `max_range`."""
    distances = np.full(len(angles), max_range)
    for i, angle in enumerate(angles):
        dx, dy = np.cos(yaw + angle), np.sin(yaw + angle)
        for rect in rects:
            t = rect.ray(origin[0], origin[1], dx, dy)
            if t < distances[i]:
                distances[i] = t
    return distances


def wall_rects(half: float = ROOM_HALF, thickness: float = WALL_THICKNESS) -> list[Rect]:
    """The four walls, inner faces at +/- `half`."""
    outer = half + thickness / 2
    span = half + thickness
    return [
        Rect(outer, 0.0, thickness / 2, span),
        Rect(-outer, 0.0, thickness / 2, span),
        Rect(0.0, outer, span, thickness / 2),
        Rect(0.0, -outer, span, thickness / 2),
    ]


def add_room(world, *, half: float = ROOM_HALF) -> None:
    """Add the walls and every item pool. Call before `world.add_fly()`.

    Walls and obstacle blocks join ``world.ground_geoms`` so contact pairs are
    generated for them; everything else is visual only.
    """
    body = world.mjcf_root.worldbody
    body.add_geom(
        type=GEOM_TYPES["box"], name="room_floor", size=(half, half, 0.005),
        pos=(0.0, 0.0, FLOOR_Z), rgba=ROOM_FLOOR_RGBA, contype=0, conaffinity=0,
    )
    walls = []
    for i, rect in enumerate(wall_rects(half)):
        geom = body.add_geom(
            type=GEOM_TYPES["box"], name=f"room_wall_{i}",
            size=(rect.hx, rect.hy, WALL_HEIGHT / 2),
            pos=(rect.cx, rect.cy, WALL_HEIGHT / 2),
            rgba=WALL_RGBA, contype=0, conaffinity=0,
        )
        world.ground_geoms.append(geom)
        walls.append(geom)

    def parked(kind: int, i: int, z: float):
        return (PARK_ORIGIN[0] + 40.0 * kind, PARK_ORIGIN[1] + 40.0 * i, z)

    pools: dict[str, list] = {name: [] for name in POOL_SIZES}
    for i in range(POOL_SIZES["obstacle"]):
        geom = body.add_geom(
            type=GEOM_TYPES["box"], name=f"room_obstacle_{i}",
            size=(OBSTACLE_HALF, OBSTACLE_HALF, WALL_HEIGHT / 2),
            pos=parked(0, i, WALL_HEIGHT / 2), rgba=OBSTACLE_RGBA,
            contype=0, conaffinity=0,
        )
        world.ground_geoms.append(geom)
        pools["obstacle"].append(geom)
    for i in range(POOL_SIZES["sugar"]):
        pools["sugar"].append(body.add_geom(
            type=GEOM_TYPES["cylinder"], name=f"room_sugar_{i}",
            size=(SUGAR_RADIUS, 0.01, 0), pos=parked(1, i, SUGAR_Z),
            rgba=SUGAR_MARKER_RGBA, contype=0, conaffinity=0, group=2,
        ))
    for i in range(POOL_SIZES["shock"]):
        pools["shock"].append(body.add_geom(
            type=GEOM_TYPES["box"], name=f"room_shock_{i}",
            size=(ZONE_HALF, ZONE_HALF, 0.005), pos=parked(2, i, SHOCK_Z),
            rgba=SHOCK_RGBA, contype=0, conaffinity=0, group=2,
        ))
    for i in range(POOL_SIZES["patch"]):
        pools["patch"].append(body.add_geom(
            type=GEOM_TYPES["box"], name=f"room_patch_{i}",
            size=(ZONE_HALF, ZONE_HALF, 0.005), pos=parked(3, i, PATCH_Z),
            rgba=FLOOR_COLOURS["blue"], contype=0, conaffinity=0,
        ))
    for i in range(POOL_SIZES["odour"]):
        pools["odour"].append(body.add_geom(
            type=GEOM_TYPES["sphere"], name=f"room_odour_{i}",
            size=(0.8, 0, 0), pos=parked(4, i, ODOUR_Z),
            rgba=ODOUR_MARKER_RGBA, contype=0, conaffinity=0, group=2,
        ))
    for kind, index, z in (("heat", 5, HEAT_Z), ("light", 6, LIGHT_Z)):
        for i in range(POOL_SIZES[kind]):
            pools[kind].append(body.add_geom(
                type=GEOM_TYPES["box"], name=f"room_{kind}_{i}",
                size=(ZONE_HALF, ZONE_HALF, 0.005), pos=parked(index, i, z),
                rgba=ZONE_HIDDEN_RGBA, contype=0, conaffinity=0, group=2,
            ))
    drum = body.add_body(name="room_drum", mocap=True, pos=(0.0, 0.0, MOVING_PARK_Z))
    stripes = []
    for i in range(DRUM_STRIPES):
        phi = 2.0 * np.pi * (i + 0.5) / DRUM_STRIPES
        # A little wider than the arc, so neighbouring dark stripes meet.
        half_width = DRUM_RADIUS * np.pi / DRUM_STRIPES * 1.02
        stripes.append(drum.add_geom(
            type=GEOM_TYPES["box"], name=f"room_drum_{i}",
            size=(0.5, half_width, DRUM_HEIGHT / 2),
            pos=(DRUM_RADIUS * np.cos(phi), DRUM_RADIUS * np.sin(phi), DRUM_HEIGHT / 2),
            quat=(np.cos(phi / 2), 0.0, 0.0, np.sin(phi / 2)),
            rgba=DRUM_RGBA, contype=0, conaffinity=0,
        ))
    shadow = body.add_body(name="room_shadow", mocap=True, pos=(0.0, 0.0, MOVING_PARK_Z))
    bar = shadow.add_geom(
        type=GEOM_TYPES["box"], name="room_shadow_bar",
        size=(SHADOW_WIDTH / 2, SHADOW_LENGTH / 2, 0.5), pos=(0.0, 0.0, 0.0),
        rgba=SHADOW_RGBA, contype=0, conaffinity=0,
    )
    world._flyplay_room = {"half": half, "walls": walls, "pools": pools,
                           "drum": drum, "drum_stripes": stripes, "shadow": shadow, "shadow_bar": bar}


@dataclass
class Item:
    """One thing placed in the room."""

    id: int
    kind: str  # one of POOL_SIZES
    x: float
    y: float
    slot: int
    #: Kind-specific settings: sugar type and concentration, voltage, colour,
    #: odour and intensity, half-size. Interpreted by `flyplay.sandbox`.
    params: dict = field(default_factory=dict)

    @property
    def extent(self) -> tuple[float, float]:
        """Half-extents along x and y, mm. Moving stimuli have none on the floor."""
        if self.kind in MOVING_KINDS:
            return (0.0, 0.0)
        if self.kind == "obstacle":
            return (float(self.params.get("hx", OBSTACLE_HALF)),
                    float(self.params.get("hy", OBSTACLE_HALF)))
        if self.kind == "sugar":
            r = float(self.params.get("radius", SUGAR_RADIUS))
            return (r, r)
        if "hx" in self.params:
            # Heat zones may be rectangles, to tile a floor round a cool spot.
            return (float(self.params["hx"]), float(self.params["hy"]))
        h = float(self.params.get("half", ZONE_HALF))
        return (h, h)

    @property
    def half(self) -> float:
        """The larger half-extent: a drop's radius, a square's half side."""
        return max(self.extent)

    @property
    def rect(self) -> Rect:
        hx, hy = self.extent
        return Rect(self.x, self.y, hx, hy)

    def as_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "x": round(self.x, 2),
                "y": round(self.y, 2), "half": round(self.half, 2), **self.params}


class Room:
    """Place, move and remove items at runtime. See the module docstring."""

    def __init__(self, sim, world):
        spec = world._flyplay_room
        self.sim = sim
        self.half = float(spec["half"])
        model = sim.mj_model
        ids = lambda geoms: [mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, g.name) for g in geoms]
        self.wall_ids = np.array(ids(spec["walls"]), dtype=np.int32)
        self.pool_ids = {kind: ids(geoms) for kind, geoms in spec["pools"].items()}
        self._parked = {
            kind: [model.geom_pos[g].copy() for g in pool] for kind, pool in self.pool_ids.items()
        }
        self.slots: dict[str, list[int | None]] = {
            kind: [None] * len(pool) for kind, pool in self.pool_ids.items()
        }
        self.items: dict[int, Item] = {}
        self._next_id = 1
        self.walls = wall_rects(self.half)
        self.solid_ids = np.array(list(self.wall_ids) + list(self.pool_ids["obstacle"]),
                                  dtype=np.int32)
        body = lambda element: mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, element.name)
        #: Mocap slots of the moving stimuli, and their geoms.
        self.mocap = {"drum": int(model.body_mocapid[body(spec["drum"])]),
                      "shadow": int(model.body_mocapid[body(spec["shadow"])])}
        self.drum_geoms = np.array(ids(spec["drum_stripes"]), dtype=np.int32)
        self.shadow_geom = ids([spec["shadow_bar"]])[0]
        for kind in MOVING_KINDS:
            self.slots[kind] = [None]
        # Everything the fly can collide with, for `gate_bounds`. A wall keeps
        # its compiled radius; an obstacle is resizable, so its radius is the
        # expression `_apply_params` writes, read from the size it has now.
        # Parked obstacles are gated off too: that is most of the 756 pairs.
        self._gated: list[tuple[int, float | None]] = (
            [(int(g), float(model.geom_rbound[g])) for g in self.wall_ids]
            + [(int(g), None) for g in self.pool_ids["obstacle"]])
        self._gate_on = False

    # --- editing ------------------------------------------------------------

    def place(self, kind: str, x: float, y: float, item_id: int | None = None, **params) -> Item:
        """Put an item in the room. `item_id` brings a removed item back under
        its old id, so an undo history that names it keeps pointing at it;
        ids are otherwise never reused."""
        if kind not in self.slots:
            raise ValueError(f"unknown item kind {kind!r}; choose from {sorted(self.slots)}")
        if item_id is not None and item_id in self.items:
            raise ValueError(f"item {item_id} is already in the room")
        try:
            slot = self.slots[kind].index(None)
        except ValueError:
            raise ValueError(f"no free {kind} left (the room holds {len(self.slots[kind])})")
        new_id = self._next_id if item_id is None else int(item_id)
        item = Item(new_id, kind, 0.0, 0.0, slot, dict(params))
        self._next_id = max(self._next_id, new_id + 1)
        self.slots[kind][slot] = item.id
        self.items[item.id] = item
        self._apply_params(item)
        self.move(item.id, x, y)
        return item

    def move(self, item_id: int, x: float, y: float) -> Item:
        item = self.items[item_id]
        if item.kind in MOVING_KINDS:
            # Nowhere in particular: the drum surrounds the room and the shadow
            # crosses all of it. `animate` poses them.
            item.x = item.y = 0.0
            return item
        hx, hy = item.extent
        item.x = float(np.clip(x, -(self.half - hx), self.half - hx))
        item.y = float(np.clip(y, -(self.half - hy), self.half - hy))
        gid = self.pool_ids[item.kind][item.slot]
        z = self.sim.mj_model.geom_pos[gid][2]
        self.sim.mj_model.geom_pos[gid] = (item.x, item.y, z)
        return item

    def update(self, item_id: int, **params) -> Item:
        item = self.items[item_id]
        item.params.update(params)
        self._apply_params(item)
        return self.move(item_id, item.x, item.y)  # a new size may need re-clamping

    def remove(self, item_id: int) -> None:
        item = self.items.pop(item_id)
        self.slots[item.kind][item.slot] = None
        if item.kind in MOVING_KINDS:
            self._park(item.kind)
            return
        gid = self.pool_ids[item.kind][item.slot]
        self.sim.mj_model.geom_pos[gid] = self._parked[item.kind][item.slot]

    def clear(self) -> None:
        for item_id in list(self.items):
            self.remove(item_id)

    def _park(self, kind: str) -> None:
        data = self.sim.mj_data
        data.mocap_pos[self.mocap[kind]] = (0.0, 0.0, MOVING_PARK_Z)
        data.mocap_quat[self.mocap[kind]] = (1.0, 0.0, 0.0, 0.0)

    def animate(self, seconds: dict[str, float]) -> dict:
        """Pose the moving stimuli for `seconds` since each was placed (a dict
        by kind; kinds absent are skipped). Returns what the shadow is doing,
        for counting passes: ``{"pass": k}`` while its k-th pass of the
        current burst is overhead, else ``{}``."""
        data = self.sim.mj_data
        state: dict = {}
        drum = next((i for i in self.items.values() if i.kind == "drum"), None)
        if drum is not None and "drum" in seconds:
            yaw = np.deg2rad(float(drum.params.get("offset", 0.0)) + float(drum.params.get("speed", 0.0)) * seconds["drum"])
            data.mocap_pos[self.mocap["drum"]] = (0.0, 0.0, 0.0)
            data.mocap_quat[self.mocap["drum"]] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
        shadow = next((i for i in self.items.values() if i.kind == "shadow"), None)
        if shadow is not None and "shadow" in seconds:
            p = shadow.params
            span = 2.0 * (self.half + SHADOW_WIDTH)
            crossing = span / SHADOW_SPEED
            t = seconds["shadow"] % float(p.get("interval", 10.0))
            k = int(t // float(p.get("gap", 1.0)))
            into = t - k * float(p.get("gap", 1.0))
            if k < int(p.get("passes", 1)) and into < crossing:
                x = -span / 2 + SHADOW_SPEED * into
                data.mocap_pos[self.mocap["shadow"]] = (x, 0.0, SHADOW_HEIGHT)
                state = {"pass": k, "x": x}
            else:
                data.mocap_pos[self.mocap["shadow"]] = (0.0, 0.0, MOVING_PARK_Z)
        return state

    def _apply_params(self, item: Item) -> None:
        model = self.sim.mj_model
        if item.kind == "drum":
            item.params["speed"] = float(np.clip(item.params.get("speed", 60.0), -180.0, 180.0))
            # A turn of the whole pattern, degrees: the heat maze's probe rotates
            # the panorama (Ofstad et al. 2011).
            item.params["offset"] = float(item.params.get("offset", 0.0)) % 360.0
            # A pattern with landmarks: dark sectors of 90, 10 and 30 degrees
            # with unequal gaps, so no turn of it looks like another (Ofstad et
            # al. 2011 used a panorama of distinct bars). Bars 40, 10 and 20
            # degrees wide, tried first, let a put-down compass settle 65
            # degrees off in one trial of four. "stripes" is the even grating.
            pattern = item.params.get("pattern", "stripes")
            if pattern == "landmarks":
                dark = set(LANDMARK_STRIPES)
            else:
                pattern = "stripes"
                dark = set(range(0, DRUM_STRIPES, 2))
            item.params["pattern"] = pattern
            for i, gid in enumerate(self.drum_geoms):
                model.geom_rgba[gid] = DRUM_RGBA if i in dark else (0.0, 0.0, 0.0, 0.0)
            return
        if item.kind == "shadow":
            item.params["interval"] = float(np.clip(item.params.get("interval", 10.0), 2.0, 120.0))
            item.params["passes"] = int(np.clip(item.params.get("passes", 1), 1, 10))
            item.params["gap"] = float(np.clip(item.params.get("gap", 1.0), 0.6, 5.0))
            return
        gid = self.pool_ids[item.kind][item.slot]
        if item.kind == "obstacle":
            hx = float(np.clip(item.params.get("hx", OBSTACLE_HALF), OBSTACLE_MIN_HALF, OBSTACLE_MAX_HALF))
            hy = float(np.clip(item.params.get("hy", OBSTACLE_HALF), OBSTACLE_MIN_HALF, OBSTACLE_MAX_HALF))
            item.params["hx"], item.params["hy"] = hx, hy
            model.geom_size[gid][:2] = (hx, hy)
            # Both bounds, or contacts beyond the compiled 4 mm block vanish.
            # Measured with a 40 mm wall, fly walking into it 15 mm off centre
            # with the reflex off: bounds updated, contact at 0.43 s and the fly
            # stops at the face; bounds left compiled, no contact at all and
            # the fly walks straight through to 25 mm past it.
            model.geom_rbound[gid] = float(np.linalg.norm(model.geom_size[gid]))
            model.geom_aabb[gid][3:5] = (hx, hy)
        if item.kind in ZONE_KINDS:
            half = float(np.clip(item.params.get("half", ZONE_HALF), 2.0, self.half))
            item.params["half"] = half
            model.geom_size[gid][:2] = half
            if "hx" in item.params:
                model.geom_size[gid][:2] = (item.params["hx"], item.params["hy"])
            # Kept in step with the size so nothing downstream culls the geom
            # by a stale bounding radius.
            model.geom_rbound[gid] = float(np.linalg.norm(model.geom_size[gid]))
        if item.kind == "sugar":
            # A drop shrinks as it is eaten; its footprint for tasting shrinks
            # with it (`Item.half`).
            radius = float(np.clip(item.params.get("radius", SUGAR_RADIUS), 0.3, SUGAR_MAX_RADIUS))
            item.params["radius"] = radius
            model.geom_size[gid][0] = radius
            model.geom_rbound[gid] = float(np.linalg.norm(model.geom_size[gid][:2]))
        if item.kind == "patch":
            model.geom_rgba[gid] = FLOOR_COLOURS[item.params.get("colour", "blue")]
        if "rgba" in item.params:
            model.geom_rgba[gid] = item.params["rgba"]

    # --- collision bounds ----------------------------------------------------

    def gate_bounds(self, x: float, y: float) -> None:
        """Leave a real bounding sphere only on the solids near (x, y).

        Call it right before stepping physics, with the thorax's position. See
        `GATE_MM` for why this changes nothing the fly can feel: a pair that is
        dropped could not have produced a contact before the next call.
        """
        model = self.sim.mj_model
        for gid, compiled in self._gated:
            pos, size = model.geom_pos[gid], model.geom_size[gid]
            gap_x = abs(x - pos[0]) - size[0]
            gap_y = abs(y - pos[1]) - size[1]
            if max(gap_x, 0.0) ** 2 + max(gap_y, 0.0) ** 2 < GATE_MM * GATE_MM:
                model.geom_rbound[gid] = compiled if compiled is not None else float(np.linalg.norm(size))
            else:
                model.geom_rbound[gid] = FAR_RBOUND
        self._gate_on = True

    def ungate(self) -> None:
        """Put every bounding sphere back.

        Physics that does not go through `gate_bounds` -- settling a body that
        has just been set down somewhere else -- must call this first, or the
        fly stands beside a wall whose sphere is still 1e-6 and walks through
        it without a single contact.
        """
        if not self._gate_on:
            return
        model = self.sim.mj_model
        for gid, compiled in self._gated:
            model.geom_rbound[gid] = (compiled if compiled is not None
                                      else float(np.linalg.norm(model.geom_size[gid])))
        self._gate_on = False

    # --- reading ------------------------------------------------------------

    def of_kind(self, kind: str) -> list[Item]:
        return [item for item in self.items.values() if item.kind == kind]

    def solids(self) -> list[Rect]:
        """Walls and placed obstacle blocks: what the reflex keeps the fly off."""
        return self.walls + [item.rect for item in self.of_kind("obstacle")]

    def touching_solid(self) -> bool:
        """True while any fly geom is in contact with a wall or obstacle."""
        data = self.sim.mj_data
        n = data.ncon
        if n == 0:
            return False
        c = data.contact
        hit = np.isin(c.geom1[:n], self.solid_ids) | np.isin(c.geom2[:n], self.solid_ids)
        return bool(hit.any())

    def as_list(self) -> list[dict]:
        return [item.as_dict() for item in self.items.values()]


@dataclass
class ReflexState:
    active: bool
    turn: float
    distances: np.ndarray
    reason: str


class ProximityReflex:
    """Turn away before touching a wall or obstacle.

    Seven rays from the head fan across the front. Anything within
    `front_near` of the three front rays starts a full-authority turn toward
    the freer side, and the side is held until the front is clear again --
    re-deciding every step made the fly dither into corners. Anything within
    `side_near` of a side ray gives a half turn away, which is what makes the fly
    slide along a wall rather than into it.

    Distances are the body's limits, not a fly's: see the module docstring.
    """

    ANGLES = np.deg2rad([-80.0, -50.0, -25.0, 0.0, 25.0, 50.0, 80.0])
    FRONT = np.abs(np.deg2rad([-80.0, -50.0, -25.0, 0.0, 25.0, 50.0, 80.0])) <= np.deg2rad(26.0)

    def __init__(self, *, front_near: float = 7.0, side_near: float = 2.5,
                 max_range: float = 12.0, head_offset: float = 1.2):
        self.front_near = front_near
        self.side_near = side_near
        self.max_range = max_range
        self.head_offset = head_offset
        self._side = 0.0

    def reset(self) -> None:
        self._side = 0.0

    def __call__(self, xy, yaw: float, solids) -> ReflexState:
        head = (xy[0] + self.head_offset * np.cos(yaw), xy[1] + self.head_offset * np.sin(yaw))
        d = cast_rays(head, yaw, self.ANGLES, solids, self.max_range)
        front = float(d[self.FRONT].min())
        left, right = float(d[self.ANGLES > 0].mean()), float(d[self.ANGLES < 0].mean())
        if front < self.front_near:
            if self._side == 0.0:
                # Positive turn is to the left (descending[0] < descending[1]).
                self._side = 1.0 if left >= right else -1.0
            return ReflexState(True, self._side, d, "front")
        if front > self.front_near + 1.0:
            self._side = 0.0
        closest = int(np.argmin(d))
        if d[closest] < self.side_near and not self.FRONT[closest]:
            away = -float(np.sign(self.ANGLES[closest]))
            return ReflexState(True, 0.5 * away, d, "side")
        return ReflexState(False, 0.0, d, "")

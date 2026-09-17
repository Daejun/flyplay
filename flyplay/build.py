"""Build a NeuroMechFly v2 simulation for a named terrain.

Everything the other modules need (the fly, the world, the compiled
`Simulation`, camera handles, and cached MuJoCo index lookups) is assembled
here so scripts stay short.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Callable

import mujoco as mj
import numpy as np
import yaml
from flygym import Simulation
from flygym.anatomy import (
    LEGS,
    AnatomicalJoint,
    AxisOrder,
    BodySegment,
    ContactBodiesPreset,
    Skeleton,
)
from flygym.compose import (
    ActuatorType,
    BlocksTerrainWorld,
    FlatGroundWorld,
    GappedTerrainWorld,
    MixedTerrainWorld,
)
from flygym.compose.fly.neuromechfly import DEFAULT_VISUALS_CONFIG_PATH
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import make_locomotion_fly

from flyplay.arena import (
    ColourFloor,
    add_floor_tiles,
    add_odor_marker,
    add_pillar,
    marker_geom_ids,
    pillar_geom_ids,
)
from flyplay.odor import add_odor_sensors
from flyplay.room import Room, add_room


#: NeuroMechFly's two proboscis hinges, rostrum on the head and haustellum on
#: the rostrum. `make_locomotion_fly` builds legs only, which welds both shut.
PROBOSCIS_JOINTS = (("c_head", "c_rostrum"), ("c_rostrum", "c_haustellum"))
#: Pitch of each hinge at full extension, rad, in `PROBOSCIS_JOINTS` order.
#: Folded, the proboscis is Z-shaped: the rostrum (0.37 mm) points straight
#: back under the head and the haustellum (0.37 mm) forward-down. At these
#: angles both point forward-down at 69-70 deg, 0.7 deg from collinear, and the
#: labellum's lowest point sits 0.05 mm above the floor -- a sugar drop's top
#: is 0.055. The first choice, (-1.4, 1.8), only minimised the tip's height:
#: it left a 47 deg kink with the labellum pointing back at the legs, and the
#: user found it awkward. Searched on a 0.1 rad grid for straight (<= 20 deg),
#: forward-down (50-88 deg) poses touching the floor; this reaches furthest.
PROBOSCIS_EXTENDED = (-1.9, 2.7)
#: How far the haustellum swings back when the labellum lifts between sips,
#: rad at ``lift=1``. Through the hinge's 0.1 s time constant a 0.075 s gap
#: shows only part of it.
PROBOSCIS_SIP_LIFT = 0.5
#: Hinge spring and damper. A motor torque of stiffness x angle holds an
#: angle, so the legs' position actuators (which `apply_locomotion_action`
#: fills by count) stay untouched. At the legs' 0.05 the proboscis sagged
#: under gravity to (-0.60, +0.36) rad with no torque at all; at 5 it sags
#: 0.01 rad, reaches 90% of full extension in 0.23 s and folds back as fast.
#: The speed is a model choice, not fitted. Walking is unchanged: 13.84 vs
#: 13.85 mm/s straight, 147.5 vs 147.8 deg/s turning, with and without.
PROBOSCIS_STIFFNESS = 5.0
PROBOSCIS_DAMPING = 0.5


def add_proboscis(fly) -> None:
    """Give `fly` a working proboscis: two pitch hinges and a motor on each.

    The neck stays rigid: it is listed with no axes only so the joint tree
    reaches the head. Both proboscis segments are in NeuroMechFly's
    ``hidden_segments``, so the fly's own eyes never see it move.
    """
    skeleton = Skeleton(
        axis_order=AxisOrder.YAW_PITCH_ROLL,
        anatomical_joints=[
            AnatomicalJoint("c_thorax", "c_head", []),
            *(AnatomicalJoint(parent, child, ["pitch"]) for parent, child in PROBOSCIS_JOINTS),
        ],
    )
    joints = fly.add_joints(skeleton, stiffness=PROBOSCIS_STIFFNESS, damping=PROBOSCIS_DAMPING)
    fly.add_actuators(list(joints), ActuatorType.MOTOR, forcerange=(-20.0, 20.0))


def display_colours(fly, sim) -> dict[int, np.ndarray]:
    """``{geom id: rgba}`` for painting a plain fly in a viewer's scene only.

    The colours of NeuroMechFly's own ``visuals.yaml`` (a texture's base colour
    where it has one), keyed by compiled geom id. Built into the model instead
    (``colorize=True``) the fly's eyes see its own coloured legs: measured in
    the room, 350 of 1442 ommatidia changed, VPN outputs by up to 0.42, and the
    visual Kenyon-cell code differed on 57% of samples.
    """
    with open(DEFAULT_VISUALS_CONFIG_PATH) as f:
        sets = yaml.safe_load(f)
    colours: dict[int, np.ndarray] = {}
    for segment, geoms in fly.bodyseg_to_mjcfgeom.items():
        for vis in sets.values():
            patterns = vis["apply_to"] if isinstance(vis["apply_to"], list) else [vis["apply_to"]]
            if any(fnmatchcase(segment.name, pattern) for pattern in patterns):
                material = vis["material"]
                rgb = vis["texture"]["rgb1"] if "texture" in vis else material["rgba"][:3]
                rgba = np.array([*rgb, material.get("rgba", [1, 1, 1, 1])[3]], dtype=np.float32)
                for geom in geoms:
                    gid = mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_GEOM, geom.name)
                    if gid >= 0:
                        colours[gid] = rgba
                break
    return colours


@dataclass(frozen=True)
class TerrainSpec:
    """How to build one scenario and where the fly may stand in it."""

    make_world: Callable[[int], object]
    spawn_z: float
    contact_bodies: ContactBodiesPreset
    #: (x_min, x_max, y_min, y_max) region the fly should stay inside, in mm.
    bounds: tuple[float, float, float, float]
    #: Max |bearing| of a navigation goal relative to the spawn heading, in degrees.
    goal_bearing_deg: float
    description: str


def _blocks_world(seed: int, x_range=(-5, 27), y_range=(-11, 11), height=(0.15, 0.25)):
    """`BlocksTerrainWorld` with a floor underneath the blocks.

    The stock world is only the blocks themselves -- there is nothing below or
    beyond them. A fly that walks off the patch (which it does in about two
    seconds of straight walking, since the patch is ~30 mm long) falls into
    empty space and tumbles, which looks like a locomotion failure but is not.
    The blocks sit with their bottoms at z = 0, so a plane there turns leaving
    the patch into a small step down onto flat ground.

    `height` is also lowered from the stock 0.35 mm. At 0.35 mm the hybrid
    controller barely advances -- about 5 mm in 3 s against 40 mm on flat
    ground -- which leaves an RL goal task with no reachable goal and so no
    gradient. 0.15-0.25 mm stays clearly harder than flat while remaining
    traversable.
    """
    world = BlocksTerrainWorld(
        x_range=x_range, y_range=y_range, height_range=height, rand_seed=seed
    )
    world._add_ground_plane(
        name="blocks_base_plane",
        size=(200.0, 200.0, 1.0),
        pos=(0.0, 0.0, 0.0),
        rgba=(0.25, 0.25, 0.25, 1.0),
    )
    return world


# Contact bodies: every terrain uses TIBIA_TARSUS_ONLY. Ground contact pairs are
# generated as (fly contact geom) x (ground geom), so including the thorax,
# abdomen and head roughly doubles model build time on the block terrains
# (67s -> 113s for one 699-geom world) while changing nothing about how the fly
# walks -- its body rides ~1 mm above ground and the tallest step here is
# 0.45 mm.
TERRAINS: dict[str, TerrainSpec] = {
    "flat": TerrainSpec(
        make_world=lambda seed: FlatGroundWorld(),
        spawn_z=0.8,
        contact_bodies=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
        bounds=(-60.0, 60.0, -60.0, 60.0),
        goal_bearing_deg=180.0,
        description="Flat infinite ground. Baseline for gait and steering.",
    ),
    "gapped": TerrainSpec(
        # The stock strip ends at x = 25 and the fly clears that in under 3 s of
        # straight walking, then drops off the end; 45 mm gives it room.
        make_world=lambda seed: GappedTerrainWorld(x_range=(-10, 45)),
        # 0.9 mm drops a tarsus into a gap on the first step and the fly flips;
        # 1.2 mm clears it and the retraction rule handles the rest.
        spawn_z=1.2,
        contact_bodies=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
        bounds=(-8.0, 42.0, -17.0, 17.0),
        goal_bearing_deg=75.0,
        description="Floor strips separated by 0.3 mm transverse gaps.",
    ),
    "blocks": TerrainSpec(
        make_world=_blocks_world,
        spawn_z=1.3,
        contact_bodies=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
        bounds=(-4.0, 25.0, -9.0, 9.0),
        goal_bearing_deg=75.0,
        description="Checkerboard of blocks with alternating heights (0.1/0.45 mm).",
    ),
    "mixed": TerrainSpec(
        # A fourth section, for the same reason gapped is extended to 45 mm.
        make_world=lambda seed: MixedTerrainWorld(
            x_ranges=((-4, 5), (5, 14), (14, 23), (23, 32)),
            y_range=(-12, 12),
            rand_seed=seed,
        ),
        spawn_z=1.3,
        contact_bodies=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
        bounds=(-3.0, 30.0, -10.0, 10.0),
        goal_bearing_deg=75.0,
        description="Repeating blocks / gaps / flat sections.",
    ),
}


@dataclass
class FlySim:
    """A built simulation plus the index lookups scripts keep asking for."""

    terrain: str
    spec: TerrainSpec
    fly: object
    world: object
    sim: Simulation
    cameras: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name: str = self.fly.name
        self.dof_order = self.fly.get_actuated_jointdofs_order("position")
        seg_order = self.fly.get_bodysegs_order()
        self._thorax_idx = seg_order.index(BodySegment("c_thorax"))
        body_ids = self.sim._internal_bodyids_by_fly[self.name]
        self._thorax_body_id = int(body_ids[self._thorax_idx])
        self._tarsus5 = [BodySegment(f"{leg}_tarsus5") for leg in LEGS]
        self._vel_buf = np.zeros(6, dtype=float)
        #: Set by `build` when the scene has odour sources / pillars.
        self.odor_field = None
        self.pillar_geom_ids = np.empty(0, dtype=np.int32)
        self.marker_geom_ids = np.empty(0, dtype=np.int32)
        #: Set by `build` when the scene has painted floor tiles.
        self.floor: ColourFloor | None = None
        #: Set by `build` when the scene is the sealed room (`flyplay.room`).
        self.room: Room | None = None
        #: Motor-driven proboscis hinges, empty unless built with `proboscis`.
        self._proboscis = self.fly.get_actuated_jointdofs_order(ActuatorType.MOTOR)

    @property
    def thorax_body_id(self) -> int:
        """MuJoCo body id of the thorax. Use it to aim a tracking camera."""
        return self._thorax_body_id

    # --- state readouts -------------------------------------------------

    def thorax_pos(self) -> np.ndarray:
        """World-frame thorax position (x, y, z) in mm."""
        return self.sim.get_body_positions(self.name)[self._thorax_idx].copy()

    def thorax_frame(self) -> np.ndarray:
        """3x3 rotation matrix of the thorax; columns are forward/left/up."""
        return self.sim.mj_data.xmat[self._thorax_body_id].reshape(3, 3).copy()

    def heading(self) -> np.ndarray:
        """Unit forward vector of the thorax in the world frame."""
        return self.thorax_frame()[:, 0]

    def yaw(self) -> float:
        """Heading angle in the world XY plane, in radians."""
        fwd = self.heading()
        return float(np.arctan2(fwd[1], fwd[0]))

    def upright(self) -> float:
        """z-component of the thorax up-axis: 1.0 upright, <0 upside down."""
        return float(self.thorax_frame()[2, 2])

    def _object_velocity(self, local: bool) -> np.ndarray:
        """6D thorax velocity as (wx, wy, wz, vx, vy, vz); mm/s and rad/s."""
        mj.mj_objectVelocity(
            self.sim.mj_model,
            self.sim.mj_data,
            mj.mjtObj.mjOBJ_BODY,
            self._thorax_body_id,
            self._vel_buf,
            int(local),
        )
        return self._vel_buf

    def body_velocity(self) -> np.ndarray:
        """Thorax linear velocity (forward, left, up) in mm/s, body frame."""
        return self._object_velocity(local=True)[3:].copy()

    def world_velocity(self) -> np.ndarray:
        """Thorax linear velocity (vx, vy, vz) in mm/s, world frame."""
        return self._object_velocity(local=False)[3:].copy()

    def yaw_rate(self) -> float:
        """Thorax angular velocity about the world z axis, in rad/s."""
        return float(self._object_velocity(local=False)[2])

    def leg_contacts(self, force_threshold: float = 1e-3) -> np.ndarray:
        """Boolean ground-contact flag per leg, ordered as `flygym.anatomy.LEGS`."""
        forces = self.sim.get_bodysegment_contact_forces(
            self.name, self._tarsus5, ground_only=True
        )
        return np.linalg.norm(forces, axis=1) > force_threshold

    def odor(self) -> np.ndarray:
        """Odour intensity at the four sensors, shape ``(4, n_dimensions)``."""
        if self.odor_field is None:
            raise ValueError(
                "This scene has no odour field. Pass odor_field= to build()."
            )
        return self.odor_field.read(self.sim)

    def touching_pillar(self) -> bool:
        """True while any part of the fly is in contact with an obstacle."""
        if self.pillar_geom_ids.size == 0:
            return False
        ncon = self.sim.mj_data.ncon
        if ncon == 0:
            return False
        contacts = self.sim.mj_data.contact
        geom1 = contacts.geom1[:ncon]
        geom2 = contacts.geom2[:ncon]
        active = ~contacts.exclude[:ncon].astype(bool)
        hit = np.isin(geom1, self.pillar_geom_ids) | np.isin(
            geom2, self.pillar_geom_ids
        )
        return bool((hit & active).any())

    def out_of_bounds(self, margin: float = 0.0) -> bool:
        x_min, x_max, y_min, y_max = self.spec.bounds
        x, y, _ = self.thorax_pos()
        return not (
            x_min + margin <= x <= x_max - margin
            and y_min + margin <= y <= y_max - margin
        )

    @property
    def has_proboscis(self) -> bool:
        return len(self._proboscis) > 0

    def set_proboscis(self, extension: float, lift: float = 0.0) -> None:
        """Command the proboscis: `extension` 0 folded to 1 fully extended;
        `lift` 0-1 raises the labellum by swinging the haustellum back
        (`PROBOSCIS_SIP_LIFT`). The hinges get there at their own spring-damper
        pace. No-op without a proboscis.

        Written every call, never cached: `Simulation.reset` puts the motors
        back to zero behind any cache's back."""
        if not self._proboscis:
            return
        angles = np.array(PROBOSCIS_EXTENDED) * float(extension)
        angles[1] -= PROBOSCIS_SIP_LIFT * float(lift)
        self.sim.set_actuator_inputs(self.name, ActuatorType.MOTOR, angles * PROBOSCIS_STIFFNESS)

    def proboscis_angles(self) -> np.ndarray:
        """Current hinge angles, rad, in `PROBOSCIS_JOINTS` order."""
        model, data = self.sim.mj_model, self.sim.mj_data
        out = []
        for dof in self._proboscis:
            jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, f"{self.name}/{dof.name}")
            out.append(float(data.qpos[model.jnt_qposadr[jid]]))
        return np.array(out)

    def close(self) -> None:
        self.sim.close()


def build(
    terrain: str = "flat",
    *,
    name: str = "nmf",
    spawn_xy: tuple[float, float] = (0.0, 0.0),
    spawn_yaw: float = 0.0,
    cameras: tuple[str, ...] = (),
    camera_res: tuple[int, int] = (480, 640),
    playback_speed: float = 0.2,
    output_fps: int = 30,
    vision: bool = False,
    odor: bool = False,
    odor_field=None,
    pillars: tuple[tuple[float, float], ...] = (),
    pillar_kwargs: dict | None = None,
    floor_tiles: tuple[float, int] | None = None,
    room: bool = False,
    odor_markers: bool = True,
    colorize: bool = True,
    proboscis: bool = False,
    adhesion: bool = True,
    seed: int = 0,
) -> FlySim:
    """Build a `FlySim` for one of the terrains in `TERRAINS`.

    Args:
        terrain: Key into `TERRAINS` ("flat", "gapped", "blocks", "mixed").
        name: Logical fly name used by `Simulation` lookups.
        spawn_xy: Spawn position in the ground plane, in mm.
        spawn_yaw: Spawn heading in radians (0 = +x).
        cameras: Which offscreen cameras to attach a `Renderer` to. Any of
            "body" (side-on tracking), "top" (top-down tracking), "zoom"
            (close-up on the legs). Empty means no renderer, which is what
            the real-time viewer and RL training want.
        camera_res: (height, width) in pixels for the offscreen renderer.
        playback_speed: Video speed relative to real time. Fly legs move fast;
            0.1-0.2 is much easier to follow than 1.0.
        output_fps: Frame rate of the rendered video.
        vision: Add the compound-eye cameras. Enables
            `Simulation.get_raw_vision` / `get_ommatidia_readouts`, at the cost
            of two extra offscreen renders whenever you query them -- measured
            at 25 ms per call here, which is 236 physics steps' worth.
        odor: Attach the four olfactory sensor sites (`flyplay.odor`).
            Implied by passing `odor_field`.
        odor_field: An `flyplay.odor.OdorField`. Its sources get visualisation
            markers (invisible to the fly) and it is bound to the compiled
            model, so `field.read(sim)` works straight away.
        pillars: ``(x, y)`` positions of solid, visible obstacles.
        pillar_kwargs: Extra arguments for `flyplay.arena.add_pillar`.
        floor_tiles: ``(tile_size_mm, n_side)`` of visual-only floor tiles the
            eyes can see, painted at runtime through ``FlySim.floor``.
        room: Build the sealed square room of `flyplay.room` around the spawn
            point, with its item pools, editable through ``FlySim.room``.
        odor_markers: Draw a marker per odour source. The room brings its own
            odour markers, so the sandbox turns these off.
        colorize: Give each leg segment a distinct colour. The fly's eyes see
            its own legs, so a scene that feeds vision to learning builds plain
            and paints the viewer's copy instead (`display_colours`).
        proboscis: Add the two proboscis hinges with motors
            (`FlySim.set_proboscis`). Off by default, which keeps every other
            scene's model exactly as it was.
        adhesion: Enable the tarsal adhesion actuators.
        seed: Seed for terrain randomisation.

    Returns:
        A `FlySim` with the model compiled and reset to the neutral keyframe.
    """
    if terrain not in TERRAINS:
        raise ValueError(f"Unknown terrain {terrain!r}. Choose from {sorted(TERRAINS)}.")
    spec = TERRAINS[terrain]

    fly = make_locomotion_fly(name=name, add_adhesion=adhesion, colorize=colorize)
    if proboscis:
        add_proboscis(fly)
    if vision:
        fly.add_vision()
    if odor or odor_field is not None:
        add_odor_sensors(fly)

    cam_elements: dict[str, object] = {}
    if "body" in cameras:
        cam_elements["body"] = fly.add_tracking_camera(
            name="body_cam",
            pos_offset=(-0.5, -7.5, 0.0),
            rotation=Rotation3D("euler", (1.57, 0.0, 0.0)),
            fovy=35.0,
        )
    if "top" in cameras:
        cam_elements["top"] = fly.add_tracking_camera(
            name="top_cam",
            pos_offset=(0.0, 0.0, 14.0),
            rotation=Rotation3D("euler", (0.0, 0.0, -1.5708)),
            fovy=45.0,
        )
    if "zoom" in cameras:
        cam_elements["zoom"] = fly.add_tracking_camera(
            name="zoom_cam",
            pos_offset=(-0.5, -3.2, -0.3),
            rotation=Rotation3D("euler", (1.4, 0.0, 0.0)),
            fovy=45.0,
        )

    world = spec.make_world(seed)

    # Props go in before add_fly: that call is what generates the fly-to-ground
    # contact pairs, and a pillar appended afterwards would be intangible.
    for pos in pillars:
        add_pillar(world, pos, **(pillar_kwargs or {}))
    if floor_tiles is not None:
        add_floor_tiles(world, tile_size=floor_tiles[0], n_side=floor_tiles[1])
    if room:
        add_room(world)
    if odor_field is not None and odor_markers:
        for source in odor_field.sources:
            add_odor_marker(world, source.pos, rgba=source.rgba)

    # The freejoint that attaches the fly to the world only accepts quaternions,
    # so turn the requested yaw into a rotation about +z.
    spawn_quat = (np.cos(spawn_yaw / 2), 0.0, 0.0, np.sin(spawn_yaw / 2))
    world.add_fly(
        fly,
        [spawn_xy[0], spawn_xy[1], spec.spawn_z],
        Rotation3D("quat", spawn_quat),
        bodysegs_with_ground_contact=spec.contact_bodies,
        add_ground_contact_sensors=False,
    )

    sim = Simulation(world)
    if cam_elements:
        sim.set_renderer(
            list(cam_elements.values()),
            camera_res=camera_res,
            playback_speed=playback_speed,
            output_fps=output_fps,
        )

    flysim = FlySim(
        terrain=terrain, spec=spec, fly=fly, world=world, sim=sim, cameras=cam_elements
    )
    if odor_field is not None:
        odor_field.bind(sim, fly)
        flysim.odor_field = odor_field
    flysim.pillar_geom_ids = pillar_geom_ids(sim, world)
    flysim.marker_geom_ids = marker_geom_ids(sim, world)
    if floor_tiles is not None:
        flysim.floor = ColourFloor(sim, world)
    if room:
        flysim.room = Room(sim, world)
    return flysim

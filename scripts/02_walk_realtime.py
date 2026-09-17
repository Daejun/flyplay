"""Watch the fly walk, live, and steer it with the keyboard.

This opens MuJoCo's interactive viewer and drives the physics from this script,
so the fly is actually walking while you look at it -- and you can fly the
camera around it at the same time.

    python scripts/02_walk_realtime.py
    python scripts/02_walk_realtime.py --speed 0.25          # slow motion
    python scripts/02_walk_realtime.py --terrain blocks
    python scripts/02_walk_realtime.py --auto                # scripted tour

The camera follows the fly by default, so it cannot walk out of frame.

Keyboard (click the viewer window first):

    A / left      steer left           W / up     walk faster
    D / right     steer right          S / down   walk slower
    C             centre the steering  R          reset the fly
    Z / X         zoom in / out        F          free camera (stop following)
    T             cycle playback speed (1.0 / 0.5 / 0.25 / 0.1)
    P             print status to the console

Mouse: left-drag orbits, right-drag pans, wheel zooms.
"""

import argparse

import _bootstrap  # noqa: F401
import mujoco
import mujoco.viewer
import numpy as np

from flyplay.build import TERRAINS, build
from flyplay.control import DEFAULT_DECIMATION, RealtimePacer, Walker

# Left/right descending signals stay inside this band. Outside it a tripod
# stalls and the fly pivots the wrong way -- see README "Steering response".
SIGNAL_LOW, SIGNAL_HIGH = 0.4, 1.6
MAX_TURN = 0.6
SPEED_PRESETS = (1.0, 0.5, 0.25, 0.1)

KEY_LEFT, KEY_RIGHT, KEY_DOWN, KEY_UP = 263, 262, 264, 265
KEY_A, KEY_C, KEY_D, KEY_F, KEY_P = 65, 67, 68, 70, 80
KEY_R, KEY_S, KEY_T, KEY_W, KEY_X, KEY_Z = 82, 83, 84, 87, 88, 90

# The fly is about 3 mm long and the model sets MuJoCo's `statistic extent` to
# 1 mm, so the viewer's default free camera starts almost inside the thorax.
# These frame the whole animal with room to see where it is walking.
DEFAULT_DISTANCE = 14.0  # mm from the tracked body
DEFAULT_AZIMUTH = 130.0
DEFAULT_ELEVATION = -20.0
ZOOM_STEP = 1.25


class Pilot:
    """Mutable steering state shared with the viewer's key callback."""

    def __init__(self, speed: float, distance: float):
        self.drive = 1.0
        self.turn = 0.0  # >0 turns left, <0 turns right
        self.speed = speed
        self.distance = distance
        self.follow = True
        self.camera_dirty = True
        self.reset_requested = False
        self.status_requested = False

    def signal(self) -> np.ndarray:
        """Descending command as (left tripod, right tripod)."""
        return np.clip(
            [self.drive - self.turn, self.drive + self.turn], SIGNAL_LOW, SIGNAL_HIGH
        )

    def on_key(self, keycode: int) -> None:
        if keycode in (KEY_A, KEY_LEFT):
            self.turn = min(self.turn + 0.15, MAX_TURN)
        elif keycode in (KEY_D, KEY_RIGHT):
            self.turn = max(self.turn - 0.15, -MAX_TURN)
        elif keycode in (KEY_W, KEY_UP):
            self.drive = min(self.drive + 0.15, SIGNAL_HIGH)
        elif keycode in (KEY_S, KEY_DOWN):
            self.drive = max(self.drive - 0.15, SIGNAL_LOW)
        elif keycode == KEY_C:
            self.turn = 0.0
        elif keycode == KEY_Z:
            self.distance = max(self.distance / ZOOM_STEP, 2.0)
            self.camera_dirty = True
        elif keycode == KEY_X:
            self.distance = min(self.distance * ZOOM_STEP, 200.0)
            self.camera_dirty = True
        elif keycode == KEY_F:
            self.follow = not self.follow
            self.camera_dirty = True
        elif keycode == KEY_R:
            self.reset_requested = True
        elif keycode == KEY_T:
            idx = SPEED_PRESETS.index(self.speed) if self.speed in SPEED_PRESETS else -1
            self.speed = SPEED_PRESETS[(idx + 1) % len(SPEED_PRESETS)]
        elif keycode == KEY_P:
            self.status_requested = True


#: (duration_s, drive, turn, label) for the --auto tour.
AUTO_SCHEDULE = (
    (2.0, 1.0, 0.0, "walk straight"),
    (2.5, 1.0, 0.45, "turn left"),
    (2.0, 1.0, 0.0, "straight again"),
    (2.5, 1.0, -0.45, "turn right"),
    (2.0, 1.6, 0.0, "sprint"),
    (2.0, 0.5, 0.0, "slow walk"),
)


def auto_program(sim_time: float) -> tuple[float, float, str] | None:
    """Scripted (drive, turn, label) for --auto; None once the tour is over."""
    t = 0.0
    for duration, drive, turn, label in AUTO_SCHEDULE:
        t += duration
        if sim_time < t:
            return drive, turn, label
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--terrain", default="flat", choices=sorted(TERRAINS))
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed relative to real time (default: 1.0).",
    )
    parser.add_argument(
        "--decimation",
        type=int,
        default=DEFAULT_DECIMATION,
        help="Physics steps per controller update. Lower is more faithful, "
        "higher is faster (default: %(default)s).",
    )
    parser.add_argument("--fps", type=int, default=60, help="Viewer refresh rate.")
    parser.add_argument(
        "--auto", action="store_true", help="Run a scripted tour instead of using keys."
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=DEFAULT_DISTANCE,
        help="Camera distance from the fly in mm (default: %(default)s).",
    )
    parser.add_argument("--azimuth", type=float, default=DEFAULT_AZIMUTH)
    parser.add_argument("--elevation", type=float, default=DEFAULT_ELEVATION)
    parser.add_argument(
        "--free-camera",
        action="store_true",
        help="Do not follow the fly. It will walk out of frame.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    spec = TERRAINS[args.terrain]
    print(f"terrain '{args.terrain}': {spec.description}")

    fs = build(args.terrain, seed=args.seed)
    walker = Walker(fs, decimation=args.decimation, seed=args.seed)
    walker.reset()

    pilot = Pilot(args.speed, args.distance)
    pilot.follow = not args.free_camera
    pacer = RealtimePacer(fs.sim.timestep, args.speed)

    print("Keyboard" + __doc__.split("Keyboard")[1])
    if args.auto:
        print("--auto: running the scripted tour; keys still work.")
    print("Close the viewer window to stop.\n")

    last_report = 0.0
    last_label = ""

    with mujoco.viewer.launch_passive(
        fs.sim.mj_model, fs.sim.mj_data, key_callback=pilot.on_key
    ) as viewer:
        viewer.cam.azimuth = args.azimuth
        viewer.cam.elevation = args.elevation

        while viewer.is_running():
            if pilot.camera_dirty:
                pilot.camera_dirty = False
                with viewer.lock():
                    if pilot.follow:
                        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                        viewer.cam.trackbodyid = fs._thorax_body_id
                    else:
                        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                        viewer.cam.lookat[:] = fs.thorax_pos()
                    viewer.cam.distance = pilot.distance
            if pacer.speed != pilot.speed:
                pacer.speed = pilot.speed
                pacer.reset()
                print(f"playback speed -> {pilot.speed}x")

            if pilot.reset_requested:
                pilot.reset_requested = False
                walker.reset()
                pacer.reset()
                print("fly reset")

            if args.auto:
                program = auto_program(fs.sim.time)
                if program is None:
                    print("scripted tour finished")
                    break
                pilot.drive, pilot.turn, label = program
                if label != last_label:
                    print(f"  [{fs.sim.time:5.1f}s] {label}")
                    last_label = label

            walker.descending_signal = pilot.signal()

            steps = max(1, int(pilot.speed / args.fps / fs.sim.timestep))
            for _ in range(steps):
                walker.physics_step()

            viewer.sync()
            pacer.wait(fs.sim.time)

            if pilot.status_requested or fs.sim.time - last_report >= 2.0:
                pilot.status_requested = False
                last_report = fs.sim.time
                left, right = walker.descending_signal
                pos = fs.thorax_pos()
                print(
                    f"{pacer.report(fs.sim.time)} | signal L{left:.2f}/R{right:.2f} "
                    f"| pos ({pos[0]:+6.1f}, {pos[1]:+6.1f}) mm "
                    f"| speed {np.linalg.norm(fs.body_velocity()[:2]):5.1f} mm/s"
                )

    fs.close()


if __name__ == "__main__":
    main()

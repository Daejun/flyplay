"""Render the walking fly to an mp4 you can scrub frame by frame.

The live viewer is great for looking around, but at 1x speed the legs are a
blur -- a real fly steps at roughly 10 Hz. This writes slow-motion video from
tracking cameras instead, and can also dump what the fly's compound eyes see.

    python scripts/03_record_video.py
    python scripts/03_record_video.py --terrain blocks --seconds 3 --cameras body top
    python scripts/03_record_video.py --playback-speed 0.05 --cameras zoom
"""

import argparse

import _bootstrap  # noqa: F401
import imageio.v3 as iio
import numpy as np
from tqdm import trange

from flyplay.build import TERRAINS, build
from flyplay.control import DEFAULT_DECIMATION, Walker

OUT = _bootstrap.OUT / "video"


def eye_view(fs) -> np.ndarray:
    """Left and right compound-eye readouts side by side as one uint8 frame.

    `get_ommatidia_readouts` returns, per eye, one value per ommatidium split
    across a yellow- and a pale-type channel (the other channel reads 0). The
    retina's `hex_pxls_to_human_readable` lays those hexagons back out on a
    grid so they can be watched as video.
    """
    readouts = fs.ommatidia_readouts()
    panels = [
        fs.sim.retina.hex_pxls_to_human_readable(eye.max(axis=1), color_8bit=True)
        for eye in readouts
    ]
    frame = np.concatenate(panels, axis=1)
    if frame.ndim == 2:
        frame = np.repeat(frame[:, :, None], 3, axis=2)
    # libx264 needs even width and height; the retina grid is whatever size it
    # is, so trim a row/column rather than let the encoder fail.
    h, w = frame.shape[:2]
    return frame[: h - h % 2, : w - w % 2].astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--terrain", default="flat", choices=sorted(TERRAINS))
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["body", "top"],
        choices=["body", "top", "zoom"],
        help="Tracking cameras to render (default: body top).",
    )
    parser.add_argument("--seconds", type=float, default=3.0, help="Simulated seconds.")
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=0.1,
        help="Video speed relative to real time (default: 0.1 = 10x slow motion).",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--res", nargs=2, type=int, default=[480, 640], metavar=("H", "W"))
    parser.add_argument(
        "--turn",
        type=float,
        default=0.0,
        help="Constant steering bias: >0 left, <0 right, range +-0.6.",
    )
    parser.add_argument("--drive", type=float, default=1.0, help="Walking drive.")
    parser.add_argument(
        "--vision",
        action="store_true",
        help="Also record what the two compound eyes see (721 ommatidia each).",
    )
    parser.add_argument("--decimation", type=int, default=DEFAULT_DECIMATION)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    fs = build(
        args.terrain,
        cameras=tuple(args.cameras),
        camera_res=tuple(args.res),
        playback_speed=args.playback_speed,
        output_fps=args.fps,
        vision=args.vision,
        seed=args.seed,
    )
    walker = Walker(fs, decimation=args.decimation, seed=args.seed, profile=True)
    walker.reset()
    walker.descending_signal = np.clip(
        [args.drive - args.turn, args.drive + args.turn], 0.4, 1.6
    )

    n_steps = int(round(args.seconds / fs.sim.timestep))
    print(
        f"terrain '{args.terrain}' | {args.seconds}s of simulation "
        f"({n_steps:,} physics steps) | cameras {args.cameras}"
    )

    stem = f"{args.terrain}_turn{args.turn:+.2f}"
    out_dir = OUT / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    path = np.full((n_steps, 3), np.nan)
    eye_frames: list[np.ndarray] = []
    for i in trange(n_steps, desc="simulating"):
        walker.physics_step()
        path[i] = fs.thorax_pos()
        rendered = fs.sim.render_as_needed_with_profile()
        if rendered and args.vision:
            eye_frames.append(eye_view(fs))

    fs.sim.print_performance_report(show_in_notebook=False)

    # save_video treats a bare path as a file for one camera and as a directory
    # for several, so spell out one path per camera and skip the ambiguity.
    fs.sim.renderer.save_video(
        {cam.name: out_dir / f"{name}.mp4" for name, cam in fs.cameras.items()}
    )
    for name in fs.cameras:
        print(f"  {name:6s} -> {out_dir / f'{name}.mp4'}")

    if eye_frames:
        iio.imwrite(
            out_dir / "eyes.mp4",
            np.asarray(eye_frames),
            fps=args.fps,
            codec="libx264",
            quality=8,
        )
        print(f"  {'eyes':6s} -> {out_dir / 'eyes.mp4'}  (left | right ommatidia)")

    travelled = np.linalg.norm(path[-1, :2] - path[0, :2])
    print(
        f"travelled {travelled:.1f} mm in {args.seconds}s "
        f"({travelled / args.seconds:.1f} mm/s), "
        f"final upright {fs.upright():.2f}"
    )
    fs.close()


if __name__ == "__main__":
    main()

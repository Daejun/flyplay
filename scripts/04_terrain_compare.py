"""Run the same walking controller on every terrain and compare the results.

This is the "no learning yet" baseline: one fixed descending command, four
different grounds. It shows where the hand-written hybrid controller copes and
where it starts to struggle -- which is exactly the gap the RL policy in
scripts 05-07 is asked to close.

    python scripts/04_terrain_compare.py
    python scripts/04_terrain_compare.py --seconds 4 --repeats 3 --video
"""

import argparse

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from flyplay.build import TERRAINS, build
from flyplay.control import DEFAULT_DECIMATION, Walker

OUT = _bootstrap.OUT / "terrain_compare"


def run_one(terrain: str, seconds: float, seed: int, decimation: int, video: bool):
    """Walk straight for `seconds` and record the trajectory and leg contacts."""
    fs = build(
        terrain,
        cameras=("body",) if video else (),
        playback_speed=0.1,
        output_fps=30,
        seed=seed,
    )
    walker = Walker(fs, decimation=decimation, seed=seed)
    walker.reset()
    walker.descending_signal = np.array([1.0, 1.0])

    n_steps = int(round(seconds / fs.sim.timestep))
    sample_every = decimation * 5  # log at 100 Hz
    n_samples = n_steps // sample_every + 1

    path = np.full((n_samples, 3), np.nan)
    upright = np.full(n_samples, np.nan)
    contacts = np.zeros((n_samples, 6))
    sample = 0

    for i in range(n_steps):
        walker.physics_step()
        if i % sample_every == 0 and sample < n_samples:
            path[sample] = fs.thorax_pos()
            upright[sample] = fs.upright()
            contacts[sample] = fs.leg_contacts()
            sample += 1
        if video:
            fs.sim.render_as_needed(fs.sim.mj_data)

    if video:
        (OUT / "video").mkdir(parents=True, exist_ok=True)
        fs.sim.renderer.save_video(
            {fs.cameras["body"].name: OUT / "video" / f"{terrain}_seed{seed}.mp4"}
        )

    path, upright, contacts = path[:sample], upright[:sample], contacts[:sample]
    result = {
        "terrain": terrain,
        "seed": seed,
        "path": path,
        "distance": float(np.linalg.norm(path[-1, :2] - path[0, :2])),
        "forward": float(path[-1, 0] - path[0, 0]),
        "drift": float(path[-1, 1] - path[0, 1]),
        "min_upright": float(np.nanmin(upright)),
        "final_upright": float(upright[-1]),
        "mean_legs_down": float(contacts.sum(axis=1).mean()),
        "height_std": float(np.nanstd(path[:, 2])),
    }
    fs.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--repeats", type=int, default=3, help="Seeds per terrain.")
    parser.add_argument("--decimation", type=int, default=DEFAULT_DECIMATION)
    parser.add_argument(
        "--video", action="store_true", help="Also save a slow-motion clip per run."
    )
    parser.add_argument(
        "--terrains", nargs="+", default=sorted(TERRAINS), choices=sorted(TERRAINS)
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    jobs = [(t, s) for t in args.terrains for s in range(args.repeats)]
    results = [
        run_one(t, args.seconds, s, args.decimation, args.video)
        for t, s in tqdm(jobs, desc="running")
    ]

    print(
        f"\n{'terrain':>8} | {'fwd (mm)':>9} | {'drift (mm)':>10} | "
        f"{'legs down':>9} | {'min upright':>11} | {'body z std':>10}"
    )
    print("-" * 74)
    for terrain in args.terrains:
        rows = [r for r in results if r["terrain"] == terrain]
        print(
            f"{terrain:>8} | {np.mean([r['forward'] for r in rows]):9.2f} | "
            f"{np.mean([r['drift'] for r in rows]):10.2f} | "
            f"{np.mean([r['mean_legs_down'] for r in rows]):9.2f} | "
            f"{np.min([r['min_upright'] for r in rows]):11.2f} | "
            f"{np.mean([r['height_std'] for r in rows]):10.3f}"
        )
    print(
        "\nlegs down = mean number of tarsi touching ground (6 = all);  "
        "min upright = 1.0 level, 0 on its side, <0 upside down"
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), tight_layout=True)
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(args.terrains)))
    for color, terrain in zip(colors, args.terrains):
        for k, r in enumerate([r for r in results if r["terrain"] == terrain]):
            p = r["path"]
            axes[0].plot(
                p[:, 0] - p[0, 0],
                p[:, 1] - p[0, 1],
                color=color,
                alpha=0.8,
                label=terrain if k == 0 else None,
            )
            axes[1].plot(
                np.arange(len(p)) / 100.0,
                p[:, 2],
                color=color,
                alpha=0.8,
                label=terrain if k == 0 else None,
            )
    axes[0].set(
        xlabel="forward (mm)", ylabel="lateral (mm)", title="Path, straight-ahead command"
    )
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].legend()
    axes[1].set(xlabel="time (s)", ylabel="thorax height (mm)", title="Body height")
    axes[1].legend()

    out_png = OUT / "terrain_compare.png"
    fig.savefig(out_png, dpi=140)
    print(f"\nplot -> {out_png}")


if __name__ == "__main__":
    main()

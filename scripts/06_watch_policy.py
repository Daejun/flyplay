"""Watch a trained policy steer the fly, live in the MuJoCo viewer.

A green marker is not available in a passive viewer, so the goal is printed to
the console and drawn in the trajectory plot from script 07. What you see here
is the fly itself: it should orient toward the goal within a step or two and
then run at it.

    python scripts/06_watch_policy.py --run nav_flat
    python scripts/06_watch_policy.py --run nav_blocks --speed 0.3
    python scripts/06_watch_policy.py --run nav_flat --episodes 5 --headless

`--headless` skips the viewer and just reports success rates, which is the
quick way to check whether a run is any good.
"""

import argparse

import _bootstrap  # noqa: F401
import numpy as np
from stable_baselines3 import PPO

from flyplay.build import TERRAINS
from flyplay.control import RealtimePacer
from flyplay.env import FlyNavEnv, NavConfig

OUT = _bootstrap.OUT / "rl"


def resolve(run: str, checkpoint: str | None):
    run_dir = OUT / run
    if checkpoint:
        path = run_dir / "checkpoints" / checkpoint
    else:
        path = run_dir / "model.zip"
    if not path.exists():
        available = sorted(p.name for p in OUT.glob("*") if (p / "model.zip").exists())
        raise SystemExit(
            f"No model at {path}.\n"
            f"Trained runs available: {available or '(none -- run scripts/05_train.py)'}"
        )
    return path


def terrain_of(run: str, fallback: str) -> str:
    meta = OUT / run / "meta.json"
    if meta.exists():
        import json

        return json.loads(meta.read_text(encoding="utf-8")).get("terrain", fallback)
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", default="nav_flat", help="Run name under out/rl/.")
    parser.add_argument("--checkpoint", default=None, help="e.g. ppo_100000_steps.zip")
    parser.add_argument("--terrain", default=None, choices=sorted(TERRAINS))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed.")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample from the policy instead of taking its mean action.",
    )
    parser.add_argument(
        "--headless", action="store_true", help="No viewer; just print outcomes."
    )
    args = parser.parse_args()

    model_path = resolve(args.run, args.checkpoint)
    terrain = args.terrain or terrain_of(args.run, "flat")
    print(f"model  : {model_path}")
    print(f"terrain: {terrain}")

    model = PPO.load(model_path, device="cpu")
    env = FlyNavEnv(NavConfig(terrain=terrain), seed=args.seed)

    viewer_ctx = None
    if not args.headless:
        import mujoco.viewer

        viewer_ctx = mujoco.viewer.launch_passive(
            env.fs.sim.mj_model, env.fs.sim.mj_data
        )

    pacer = RealtimePacer(env.fs.sim.timestep, args.speed)
    steps_per_sync = max(1, int(args.speed / args.fps / env.fs.sim.timestep))
    outcomes: list[str] = []

    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)
            pacer.reset()
            bearing = np.rad2deg(env.goal_bearing())
            print(
                f"\nepisode {ep + 1}/{args.episodes}: goal at "
                f"({info['goal'][0]:+.1f}, {info['goal'][1]:+.1f}) mm, "
                f"{np.linalg.norm(info['goal'] - info['start'][:2]):.1f} mm away, "
                f"bearing {bearing:+.0f} deg"
            )

            done = False
            since_sync = 0
            while not done:
                action, _ = model.predict(obs, deterministic=not args.stochastic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                since_sync += env.steps_per_action
                if viewer_ctx is not None and since_sync >= steps_per_sync:
                    since_sync = 0
                    if not viewer_ctx.is_running():
                        raise KeyboardInterrupt
                    viewer_ctx.sync()
                    pacer.wait(env.fs.sim.time)

            outcomes.append(info["outcome"])
            print(
                f"  -> {info['outcome']} after {env.elapsed:.2f}s, "
                f"{info['distance']:.1f} mm from goal"
            )
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if viewer_ctx is not None:
            viewer_ctx.close()
        env.close()

    if outcomes:
        reached = outcomes.count("goal")
        print(f"\nreached the goal in {reached}/{len(outcomes)} episodes")
        for name in ("goal", "timeout", "fell", "out_of_bounds"):
            n = outcomes.count(name)
            if n:
                print(f"  {name:14s} {n}")


if __name__ == "__main__":
    main()

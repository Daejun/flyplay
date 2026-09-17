"""Train the fly to steer toward a goal with PPO.

The policy does not learn to move legs -- walking comes from the fixed hybrid
CPG controller. What it learns is the *descending command*: the two-element
signal a fly's brain sends to its ventral nerve cord to modulate the left and
right tripods. See `flyplay/env.py` for the full task definition.

    python scripts/05_train.py                            # flat, 300k steps
    python scripts/05_train.py --terrain blocks --steps 500000
    python scripts/05_train.py --terrain flat --n-envs 8

Watch it train:

    .venv\\Scripts\\tensorboard.exe --logdir out/rl/tb

Then evaluate with scripts 06 (live) and 07 (what did it learn).
"""

import argparse
import json
import time

import _bootstrap  # noqa: F401
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from flyplay.build import TERRAINS
from flyplay.env import FlyNavEnv, NavConfig

OUT = _bootstrap.OUT / "rl"


def env_factory(terrain: str, seed: int, episode_seconds: float):
    # No per-env Monitor wrapper: VecMonitor below already records episode
    # returns, and stacking the two makes SB3 warn that the inner statistics
    # are discarded.
    def _make():
        return FlyNavEnv(
            NavConfig(terrain=terrain, episode_seconds=episode_seconds), seed=seed
        )

    return _make


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--terrain", default="flat", choices=sorted(TERRAINS))
    parser.add_argument("--steps", type=int, default=300_000, help="Total env steps.")
    parser.add_argument(
        "--n-envs",
        type=int,
        default=6,
        help="Parallel simulations. Each is one MuJoCo process (default: %(default)s).",
    )
    parser.add_argument("--episode-seconds", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=512, help="PPO rollout length.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--name", default=None, help="Run name (default: nav_<terrain>)."
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Run envs in this process. Slower, but much easier to debug.",
    )
    args = parser.parse_args()

    run_name = args.name or f"nav_{args.terrain}"
    run_dir = OUT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"terrain '{args.terrain}': {TERRAINS[args.terrain].description}")
    print(
        f"run '{run_name}' | {args.steps:,} steps | {args.n_envs} envs | "
        f"{args.episode_seconds}s episodes"
    )

    factories = [
        env_factory(args.terrain, args.seed + i, args.episode_seconds)
        for i in range(args.n_envs)
    ]
    vec_cls = DummyVecEnv if args.serial else SubprocVecEnv
    venv = VecMonitor(vec_cls(factories), info_keywords=("outcome",))

    model = PPO(
        "MlpPolicy",
        venv,
        seed=args.seed,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.005,
        policy_kwargs={"net_arch": [128, 128]},
        tensorboard_log=str(OUT / "tb"),
        verbose=1,
    )

    checkpoints = CheckpointCallback(
        save_freq=max(20_000 // args.n_envs, 1),
        save_path=str(run_dir / "checkpoints"),
        name_prefix="ppo",
    )

    t0 = time.perf_counter()
    model.learn(
        total_timesteps=args.steps,
        callback=checkpoints,
        tb_log_name=run_name,
        progress_bar=True,
    )
    elapsed = time.perf_counter() - t0

    model_path = run_dir / "model.zip"
    model.save(model_path)
    meta = {
        "terrain": args.terrain,
        "steps": args.steps,
        "n_envs": args.n_envs,
        "episode_seconds": args.episode_seconds,
        "seed": args.seed,
        "wall_seconds": round(elapsed, 1),
        "steps_per_second": round(args.steps / elapsed, 1),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    venv.close()

    print(
        f"\ntrained in {elapsed / 60:.1f} min "
        f"({args.steps / elapsed:,.0f} env steps/s)"
    )
    print(f"model -> {model_path}")
    print(f"\nnext:\n  python scripts/06_watch_policy.py --run {run_name}")
    print(f"  python scripts/07_analyze_policy.py --run {run_name}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()

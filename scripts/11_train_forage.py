"""Train the fly to smell its way to food while watching where it is going.

Three stages, each starting from the previous one's weights. Learning to
chemotax and to dodge at the same time from scratch is a much harder credit
assignment problem than learning them in order.

    stage 1  forage_odor    flat,   no pillars  -> chemotaxis
    stage 2  forage_avoid   flat,   2 pillars   -> visual avoidance on top
    stage 3  forage_rough   blocks, 2 pillars   -> the paper's full task

    python scripts/11_train_forage.py                       # all three stages
    python scripts/11_train_forage.py --stages 1 2
    python scripts/11_train_forage.py --steps 150000 250000 250000

Stopping is safe at any moment -- checkpoints land every --checkpoint-every
steps, so at most that many are lost. To pick the run back up:

    python scripts/11_train_forage.py --resume --steps 150000 250000 250000

--resume restarts each stage from its own newest checkpoint (weights *and*
optimizer state), keeps the existing checkpoints, continues the step counter
and the tensorboard curve, and skips outright any stage already at its target.
Pass the same --steps you started with, or the targets will not match.

Hand-written baselines on stage 2 for reference (scripts print the same
numbers): smell alone finds the food 3/8 times with 134 collision-steps,
smell + sight 8/8 with 3.1. A trained policy should beat the second.

Watch it train:
    .venv\\Scripts\\tensorboard.exe --logdir out/rl/tb
"""

import argparse
import json
import re
import shutil
import time
import zipfile

import _bootstrap  # noqa: F401
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from flyplay.env_multimodal import ForageConfig, ForageEnv

OUT = _bootstrap.OUT / "rl"

STAGES = (
    {"name": "forage_odor", "terrain": "flat", "n_pillars": 0, "steps": 200_000},
    {"name": "forage_avoid", "terrain": "flat", "n_pillars": 2, "steps": 300_000},
    {"name": "forage_rough", "terrain": "blocks", "n_pillars": 2, "steps": 300_000},
)


def env_factory(stage: dict, seed: int, seconds: float):
    def _make():
        return ForageEnv(
            ForageConfig(
                terrain=stage["terrain"],
                n_pillars=stage["n_pillars"],
                episode_seconds=seconds,
            ),
            seed=seed,
        )

    return _make


def newest_checkpoint(checkpoint_dir):
    """Highest-numbered ppo_*_steps.zip, or (None, 0).

    Sorted by the step count in the name rather than by mtime: a checkpoint
    copied or restored out of order would otherwise look like the latest one.
    """
    best, best_steps = None, 0
    for path in checkpoint_dir.glob("ppo_*_steps.zip"):
        match = re.search(r"ppo_(\d+)_steps", path.name)
        if match and int(match.group(1)) > best_steps:
            best, best_steps = path, int(match.group(1))
    return best, best_steps


def saved_steps(path) -> int:
    """How many steps a saved SB3 zip was trained for, or 0 if unreadable.

    Read straight out of the archive's json rather than by PPO.load(), so that
    deciding whether a stage still needs work costs no model construction.
    """
    if not path.exists():
        return 0
    try:
        with zipfile.ZipFile(path) as archive:
            return int(json.loads(archive.read("data"))["num_timesteps"])
    except (KeyError, ValueError, zipfile.BadZipFile):
        return 0


def write_meta(run_dir, stage, args, init_from, **extra) -> None:
    """Describe the run on disk.

    Written *before* training starts, not just after: anything attaching to a
    run in progress (scripts/09_web_viewer.py --follow) needs to know which
    task and terrain the checkpoints belong to, and a file that only appears at
    the end leaves it guessing.
    """
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "task": "forage",
                "terrain": stage["terrain"],
                "n_pillars": stage["n_pillars"],
                "steps": stage["steps"],
                "init_from": init_from,
                "n_envs": args.n_envs,
                "episode_seconds": args.episode_seconds,
                "seed": args.seed,
                **extra,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_stage(stage: dict, args, init_from: str | None) -> str:
    run_dir = OUT / stage["name"]
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Resume from whichever of the two is further along: a finished stage's
    # model.zip is ahead of its last checkpoint, an interrupted one is behind.
    resume_from, done_steps = None, 0
    if args.resume:
        ckpt, ckpt_steps = newest_checkpoint(checkpoint_dir)
        final_steps = saved_steps(run_dir / "model.zip")
        if final_steps > ckpt_steps:
            resume_from, done_steps = run_dir / "model.zip", final_steps
        else:
            resume_from, done_steps = ckpt, ckpt_steps
    remaining = stage["steps"] - done_steps

    print(f"\n{'=' * 64}")
    print(
        f"{stage['name']}: terrain '{stage['terrain']}', "
        f"{stage['n_pillars']} pillars, {stage['steps']:,} steps"
    )

    if remaining <= 0:
        # Nothing left to do. Still make sure the next stage has weights to
        # inherit: a run killed after its last checkpoint but before
        # model.save() reached the target without ever writing one.
        if not (run_dir / "model.zip").exists():
            shutil.copy2(resume_from, run_dir / "model.zip")
            print(f"  {resume_from.name} -> model.zip")
        print(f"  already at {done_steps:,} steps, skipping")
        print("=" * 64)
        return stage["name"]

    if resume_from is not None:
        print(f"  resuming {resume_from.name} at {done_steps:,} steps, "
              f"{remaining:,} to go")
    elif init_from:
        print(f"  starting from {init_from}")
    print("=" * 64)
    write_meta(
        run_dir, stage, args, init_from, status="running",
        resumed_from=resume_from.name if resume_from is not None else None,
    )

    factories = [
        env_factory(stage, args.seed + i, args.episode_seconds)
        for i in range(args.n_envs)
    ]
    vec_cls = DummyVecEnv if args.serial else SubprocVecEnv
    venv = VecMonitor(vec_cls(factories), info_keywords=("outcome",))

    if resume_from is not None:
        # A checkpoint carries the optimizer state, not just the weights, so
        # this continues the same optimisation instead of restarting it warm.
        model = PPO.load(
            resume_from, env=venv, device="cpu", tensorboard_log=str(OUT / "tb")
        )
        model.set_env(venv)
    elif init_from and (OUT / init_from / "model.zip").exists():
        # Every stage shares one observation and action space, so the weights
        # transfer directly; only the environment around them changes.
        model = PPO.load(
            OUT / init_from / "model.zip",
            env=venv,
            device="cpu",
            tensorboard_log=str(OUT / "tb"),
        )
        model.set_env(venv)
    else:
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
            device="cpu",
        )

    # Frequent checkpoints: scripts/09_web_viewer.py --follow reloads the newest
    # one while training runs, so this interval is how often the fly you are
    # watching gets any better.
    if resume_from is None:
        stale = list(checkpoint_dir.glob("ppo_*_steps.zip"))
        if stale:
            # Otherwise --follow, and the viewer's attribution history, would
            # read this run's checkpoints and the previous run's as one series.
            print(f"  deleting {len(stale)} checkpoints from an earlier run "
                  f"(--resume continues it instead)")
            for path in stale:
                path.unlink()
    checkpoints = CheckpointCallback(
        save_freq=max(args.checkpoint_every // args.n_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix="ppo",
    )
    t0 = time.perf_counter()
    model.learn(
        # With reset_num_timesteps=False SB3 reads total_timesteps as the
        # *additional* budget and keeps writing to the same tensorboard run,
        # so both the checkpoint numbering and the curve stay continuous.
        total_timesteps=remaining,
        callback=checkpoints,
        tb_log_name=stage["name"],
        progress_bar=True,
        reset_num_timesteps=resume_from is None,
    )
    elapsed = time.perf_counter() - t0

    model.save(run_dir / "model.zip")
    write_meta(
        run_dir, stage, args, init_from,
        status="done", wall_seconds=round(elapsed, 1),
    )
    venv.close()
    print(
        f"-> {run_dir / 'model.zip'}  ({elapsed / 60:.1f} min, "
        f"{remaining / elapsed:,.0f} steps/s)"
    )
    return stage["name"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        choices=[1, 2, 3],
        help="Which stages to run (default: all).",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=None,
        help="Override the per-stage step counts.",
    )
    parser.add_argument("--n-envs", type=int, default=6)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10_000,
        help="Total env steps between checkpoints (default: %(default)s).",
    )
    parser.add_argument("--episode-seconds", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--serial", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue each stage from its newest checkpoint instead of "
             "restarting it, and skip stages already at their target. Pass the "
             "same --steps as the original run.",
    )
    parser.add_argument(
        "--no-transfer",
        action="store_true",
        help="Train each stage from scratch instead of carrying weights over.",
    )
    args = parser.parse_args()

    stages = [dict(STAGES[i - 1]) for i in args.stages]
    if args.steps:
        if len(args.steps) != len(stages):
            raise SystemExit(
                f"--steps needs one value per stage ({len(stages)} given "
                f"{len(args.steps)})"
            )
        for stage, steps in zip(stages, args.steps):
            stage["steps"] = steps

    previous = None
    for stage in stages:
        previous = run_stage(stage, args, None if args.no_transfer else previous)

    print("\nnext:")
    print(f"  python scripts/12_analyze_forage.py --run {previous}")
    print(f"  python scripts/09_web_viewer.py --policy {previous} --terrain "
          f"{stages[-1]['terrain']}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()

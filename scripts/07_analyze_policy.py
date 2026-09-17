"""Answer "what did the fly actually learn?" for a trained policy.

Four things get produced, as one PNG plus a printed summary:

1. **The learned steering law.** The policy's whole job is a map from "where
   is the goal, relative to me" to "how hard do I drive each tripod". Sweeping
   the goal bearing across +-180 deg while holding the rest of the observation
   at states actually visited during rollouts draws that map as a curve. A
   trained fly produces a monotonic curve through the origin: goal on the left
   -> drive the right tripod harder -> turn left.

2. **Trajectories.** Where the fly goes, for goals placed all around it.

3. **Outcome by bearing.** Which goal directions it handles and which it does
   not -- a fly that only ever learned to walk straight shows up immediately.

4. **One episode, unrolled.** Bearing and descending command against time, so
   you can see the feedback loop closing.

    python scripts/07_analyze_policy.py --run nav_flat
    python scripts/07_analyze_policy.py --run nav_flat --episodes 40
    python scripts/07_analyze_policy.py --run nav_flat nav_blocks   # compare
"""

import argparse
import json

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from stable_baselines3 import PPO
from tqdm import trange

from flyplay.env import SIGNAL_HIGH, SIGNAL_LOW, FlyNavEnv, NavConfig

OUT = _bootstrap.OUT / "rl"
ANALYSIS = _bootstrap.OUT / "analysis"

#: Distance held fixed while sweeping the bearing, in mm.
SWEEP_DISTANCE = 10.0


def load_run(run: str, checkpoint: str | None = None):
    run_dir = OUT / run
    model_path = (
        run_dir / "checkpoints" / checkpoint if checkpoint else run_dir / "model.zip"
    )
    if not model_path.exists():
        available = sorted(p.name for p in OUT.glob("*") if (p / "model.zip").exists())
        raise SystemExit(
            f"No model at {model_path}.\n"
            f"Trained runs: {available or '(none -- run scripts/05_train.py)'}"
        )
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return PPO.load(model_path, device="cpu"), meta.get("terrain", "flat"), meta


def rollout(model, env: FlyNavEnv, episodes: int, seed: int):
    """Run episodes with goals spread evenly around the fly."""
    max_bearing = env.max_goal_bearing
    bearings = np.linspace(-max_bearing, max_bearing, episodes)
    episodes_out = []
    observations = []

    for i in trange(episodes, desc="rollouts"):
        obs, info = env.reset(
            seed=seed + i, options={"bearing": float(bearings[i]), "distance": 14.0}
        )
        trace = {"bearing0": float(bearings[i]), "obs": [], "action": [], "bearing": []}
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            trace["obs"].append(obs.copy())
            trace["action"].append(np.asarray(action, dtype=float).copy())
            trace["bearing"].append(env.goal_bearing())
            obs, _reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        observations.extend(trace["obs"][:: max(1, len(trace["obs"]) // 20)])
        trace.update(
            path=info["path"],
            goal=info["goal"],
            outcome=info["outcome"],
            seconds=env.elapsed,
            final_distance=info["distance"],
        )
        trace["action"] = np.asarray(trace["action"])
        trace["bearing"] = np.asarray(trace["bearing"])
        episodes_out.append(trace)

    return episodes_out, np.asarray(observations, dtype=np.float32)


def steering_law(model, states: np.ndarray, n_points: int = 73):
    """Marginal policy response to goal bearing, averaged over visited states.

    `states` are real observations from rollouts, so the CPG phase, speed and
    leg-contact channels keep realistic values; only the two goal-direction
    channels and the distance channel are overwritten.
    """
    bearings = np.linspace(-np.pi, np.pi, n_points)
    mean = np.zeros((n_points, 2))
    std = np.zeros((n_points, 2))
    batch = states.copy()
    batch[:, 2] = np.tanh(SWEEP_DISTANCE / 10.0)

    for i, bearing in enumerate(bearings):
        batch[:, 0] = np.cos(bearing)
        batch[:, 1] = np.sin(bearing)
        actions, _ = model.predict(batch, deterministic=True)
        signals = FlyNavEnv.action_to_signal(actions)
        mean[i] = signals.mean(axis=0)
        std[i] = signals.std(axis=0)
    return np.rad2deg(bearings), mean, std


def plot(runs: dict[str, dict], out_png):
    n = len(runs)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), tight_layout=True)
    colors = plt.cm.tab10(np.arange(n))

    # 1. learned steering law
    ax = axes[0, 0]
    for color, (name, r) in zip(colors, runs.items()):
        deg, mean, std = r["law"]
        turn = mean[:, 1] - mean[:, 0]  # right drive - left drive; >0 turns left
        ax.plot(deg, turn, color=color, lw=2, label=f"{name} ({r['terrain']})")
        ax.fill_between(
            deg,
            turn - (std[:, 0] + std[:, 1]),
            turn + (std[:, 0] + std[:, 1]),
            color=color,
            alpha=0.15,
        )
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set(
        xlabel="goal bearing (deg)   [0 = straight ahead, + = to the left]",
        ylabel="turn command\n(right drive - left drive)",
        title="1. The learned steering law",
        xticks=np.arange(-180, 181, 60),
    )
    ax.legend(fontsize=8)
    ax.text(
        0.02,
        0.02,
        "positive = turn left",
        transform=ax.transAxes,
        fontsize=8,
        color="0.35",
    )

    # 2. trajectories (first run only, to stay readable)
    ax = axes[0, 1]
    name, r = next(iter(runs.items()))
    outcome_style = {
        "goal": ("tab:green", "reached the goal"),
        "timeout": ("tab:orange", "ran out of time"),
        "fell": ("tab:red", "fell over"),
        "out_of_bounds": ("tab:purple", "left the terrain"),
    }
    seen = set()
    for ep in r["episodes"]:
        p = ep["path"]
        c, label = outcome_style.get(ep["outcome"], ("0.5", ep["outcome"]))
        ax.plot(
            p[:, 0],
            p[:, 1],
            color=c,
            alpha=0.8,
            lw=1.2,
            label=label if ep["outcome"] not in seen else None,
        )
        seen.add(ep["outcome"])
        ax.plot(*ep["goal"], marker="*", color=c, ms=9, mec="k", mew=0.4)
    ax.plot(0, 0, marker="o", color="k", ms=7)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set(
        xlabel="x (mm)",
        ylabel="y (mm)",
        title=f"2. Trajectories - {name} ({r['terrain']})\ncircle = start, star = goal",
    )
    ax.legend(fontsize=8)

    # 3. outcome and time-to-goal against bearing
    ax = axes[1, 0]
    for color, (name, r) in zip(colors, runs.items()):
        deg = np.rad2deg([ep["bearing0"] for ep in r["episodes"]])
        secs = [ep["seconds"] if ep["outcome"] == "goal" else np.nan for ep in r["episodes"]]
        ax.plot(deg, secs, "o-", color=color, ms=4, lw=1, label=f"{name} (reached)")
        miss = [ep for ep in r["episodes"] if ep["outcome"] != "goal"]
        if miss:
            ax.plot(
                np.rad2deg([ep["bearing0"] for ep in miss]),
                [r["env_seconds"]] * len(miss),
                "x",
                color=color,
                ms=7,
                label=f"{name} (missed)",
            )
    ax.set(
        xlabel="initial goal bearing (deg)",
        ylabel="time to reach goal (s)",
        title="3. Which directions it handles",
    )
    ax.legend(fontsize=8)

    # 4. one episode unrolled
    ax = axes[1, 1]
    name, r = next(iter(runs.items()))
    ep = max(r["episodes"], key=lambda e: abs(e["bearing0"]))
    t = np.arange(len(ep["bearing"])) / r["action_hz"]
    signals = FlyNavEnv.action_to_signal(ep["action"])
    ax.plot(t, np.rad2deg(ep["bearing"]), color="k", lw=2, label="goal bearing (deg)")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set(xlabel="time (s)", ylabel="bearing (deg)")
    ax2 = ax.twinx()
    ax2.plot(t, signals[:, 0], color="tab:blue", lw=1.2, label="left tripod drive")
    ax2.plot(t, signals[:, 1], color="tab:red", lw=1.2, label="right tripod drive")
    ax2.set_ylabel("descending signal")
    ax2.set_ylim(SIGNAL_LOW - 0.1, SIGNAL_HIGH + 0.1)
    lines = ax.get_lines()[:1] + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc="upper right")
    ax.set_title(
        f"4. One episode - {name}, goal {np.rad2deg(ep['bearing0']):+.0f} deg "
        f"to the side ({ep['outcome']})"
    )

    fig.savefig(out_png, dpi=140)
    return out_png


def summarise(name: str, r: dict) -> None:
    eps = r["episodes"]
    reached = [e for e in eps if e["outcome"] == "goal"]
    deg, mean, _std = r["law"]
    turn = mean[:, 1] - mean[:, 0]

    # Slope of the steering law near straight-ahead: how hard it corrects.
    near = np.abs(deg) <= 45
    slope = np.polyfit(deg[near], turn[near], 1)[0]
    # Rank correlation rather than a step-by-step "always increasing" test: a
    # good law saturates past +-90 deg, so its successive differences there sit
    # at ~0 and noise flips half of them negative even when the curve is clean.
    front = np.abs(deg) <= 150
    monotone = float(spearmanr(deg[front], turn[front]).statistic)

    print(f"\n--- {name}  (terrain: {r['terrain']}) ---")
    print(f"  reached the goal   : {len(reached)}/{len(eps)} episodes")
    if reached:
        print(
            f"  time to goal       : {np.mean([e['seconds'] for e in reached]):.2f} s "
            f"(min {min(e['seconds'] for e in reached):.2f}, "
            f"max {max(e['seconds'] for e in reached):.2f})"
        )
    for outcome in ("timeout", "fell", "out_of_bounds"):
        n = sum(1 for e in eps if e["outcome"] == outcome)
        if n:
            print(f"  {outcome:19s}: {n}")
    print(f"  steering gain      : {slope:+.4f} signal per degree of bearing error")
    print(f"  steering monotonic : rho = {monotone:+.3f} over +-150 deg")
    print(
        f"  turn command at -90/0/+90 deg: "
        f"{np.interp(-90, deg, turn):+.2f} / "
        f"{np.interp(0, deg, turn):+.2f} / "
        f"{np.interp(90, deg, turn):+.2f}"
    )
    if slope > 0.002 and monotone > 0.9:
        print("  => learned a clean 'turn toward the goal' law.")
    elif slope > 0.0:
        print("  => turning the right way, but the law is noisy. Train longer.")
    else:
        print("  => has not learned to steer yet. Train longer or check the reward.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run", nargs="+", default=["nav_flat"], help="One or more run names."
    )
    parser.add_argument("--episodes", type=int, default=21)
    parser.add_argument("--seed", type=int, default=500)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Analyse a mid-training checkpoint instead of the final model, "
        "e.g. ppo_120000_steps.zip. Applies to every --run.",
    )
    args = parser.parse_args()

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    runs: dict[str, dict] = {}

    for name in args.run:
        model, terrain, meta = load_run(name, args.checkpoint)
        env = FlyNavEnv(NavConfig(terrain=terrain), seed=args.seed)
        print(f"\n{name}: terrain '{terrain}', trained {meta.get('steps', '?')} steps")
        episodes, states = rollout(model, env, args.episodes, args.seed)
        runs[name] = {
            "terrain": terrain,
            "meta": meta,
            "episodes": episodes,
            "law": steering_law(model, states),
            "action_hz": env.config.action_hz,
            "env_seconds": env.config.episode_seconds,
        }
        env.close()

    for name, r in runs.items():
        summarise(name, r)

    out_png = ANALYSIS / ("_vs_".join(args.run) + ".png")
    plot(runs, out_png)
    print(f"\nplot -> {out_png}")


if __name__ == "__main__":
    main()

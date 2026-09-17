"""What did the fly learn, and does it actually need both senses?

Two things get measured.

**The ablation.** The same policy is run three ways: with both senses, with
vision blanked, and with smell blanked. If the task really is multimodal, each
lesion should break it in its own characteristic way -- the blinded fly should
still head the right way but collide, the anosmic fly should dodge cleanly and
never arrive. Hand-written controllers already show this pattern (smell alone:
3/8 found, 134 collision-steps; smell + sight: 8/8, 3.1), so the trained policy
has a reference to beat.

**The two control laws.** The policy is a map from senses to a turn command.
Sweeping one sense while holding the other at states the fly actually visited
draws each law as a curve: turn-vs-odour-asymmetry should rise through the
origin (turn toward the smell), turn-vs-pillar-side should fall through it
(turn away from the obstacle). Opposite signs on one axis, which is the whole
point of the task.

    python scripts/12_analyze_forage.py --run forage_rough
    python scripts/12_analyze_forage.py --run forage_avoid --episodes 24
"""

import argparse
import json

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from tqdm import trange

from flyplay.env_multimodal import OBS_SLICES, ForageConfig, ForageEnv

OUT = _bootstrap.OUT / "rl"
ANALYSIS = _bootstrap.OUT / "analysis"

CONDITIONS = (
    ("both", {}),
    ("smell only", {"blind": True}),
    ("sight only", {"deaf_to_odor": True}),
)


def load_run(run: str, checkpoint: str | None = None):
    run_dir = OUT / run
    path = run_dir / "checkpoints" / checkpoint if checkpoint else run_dir / "model.zip"
    if not path.exists():
        available = sorted(
            p.name for p in OUT.glob("*") if (p / "model.zip").exists()
        )
        raise SystemExit(f"No model at {path}. Trained runs: {available}")
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return PPO.load(path, device="cpu"), meta


def evaluate(model, meta, episodes, seed, overrides, collect_states=False):
    config = ForageConfig(
        terrain=meta.get("terrain", "flat"),
        n_pillars=meta.get("n_pillars", 2),
        **overrides,
    )
    env = ForageEnv(config, seed=seed)
    episodes_out, states = [], []
    for i in trange(episodes, desc=f"  {overrides or 'both'}", leave=False):
        obs, _info = env.reset(seed=seed + i)
        done = False
        trace = []
        while not done:
            if collect_states:
                trace.append(obs.copy())
            action, _ = model.predict(obs, deterministic=True)
            obs, _r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        path = info["path"]
        straight = np.linalg.norm(info["source"] - path[0])
        walked = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        episodes_out.append(
            {
                "outcome": info["outcome"],
                "seconds": env.elapsed,
                "collisions": info["collisions"],
                "final_distance": info["distance"],
                "path": path,
                "source": info["source"],
                "pillars": info["pillars"],
                "detour": walked / max(straight, 1e-6),
            }
        )
        if collect_states and trace:
            states.extend(trace[:: max(1, len(trace) // 25)])
    env.close()
    return episodes_out, np.asarray(states, dtype=np.float32)


def control_law(model, states, field, values):
    """Marginal turn command while one sensory channel is swept.

    Everything except the swept channel keeps the values it had in real
    rollouts, so the curve is the policy's response in states it actually
    reaches rather than in a made-up one.
    """
    turns = np.zeros((len(values), 2))
    batch = states.copy()
    for i, value in enumerate(values):
        probe = batch.copy()
        if field == "odor":
            probe[:, OBS_SLICES["odor_asymmetry"]] = value
        else:
            vis = OBS_SLICES["vision"]
            # vision block: [l_mass, r_mass, l_az, r_az, mass_asym, total]
            magnitude = abs(value)
            probe[:, vis.start + 0] = magnitude if value > 0 else 0.0
            probe[:, vis.start + 1] = magnitude if value < 0 else 0.0
            probe[:, vis.start + 4] = value
            probe[:, vis.start + 5] = magnitude
        actions, _ = model.predict(probe, deterministic=True)
        turns[i] = [actions[:, 1].mean() - actions[:, 0].mean(), actions.std()]
    return turns


def summarise(name, episodes):
    found = [e for e in episodes if e["outcome"] == "found"]
    return {
        "condition": name,
        "found": len(found),
        "n": len(episodes),
        "seconds": np.mean([e["seconds"] for e in found]) if found else np.nan,
        "collisions": float(np.mean([e["collisions"] for e in episodes])),
        "detour": float(np.mean([e["detour"] for e in found])) if found else np.nan,
        "final": float(np.mean([e["final_distance"] for e in episodes])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", default="forage_rough")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=700)
    args = parser.parse_args()

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    model, meta = load_run(args.run, args.checkpoint)
    print(
        f"{args.run}: terrain '{meta.get('terrain')}', "
        f"{meta.get('n_pillars')} pillars, {meta.get('steps', '?')} steps"
    )

    print("\nablation")
    results, all_states = {}, None
    for name, overrides in CONDITIONS:
        episodes, states = evaluate(
            model, meta, args.episodes, args.seed, overrides,
            collect_states=(name == "both"),
        )
        results[name] = episodes
        if name == "both":
            all_states = states

    print(f"\n{'condition':<12} | {'found':>7} | {'time':>6} | {'collisions':>10} "
          f"| {'detour':>7} | {'final gap':>9}")
    print("-" * 68)
    rows = []
    for name, _ in CONDITIONS:
        s = summarise(name, results[name])
        rows.append(s)
        print(
            f"{s['condition']:<12} | {s['found']:>3}/{s['n']:<3} | "
            f"{s['seconds']:>5.2f}s | {s['collisions']:>10.1f} | "
            f"{s['detour']:>7.2f} | {s['final']:>7.1f}mm"
        )
    print(
        "\ncollisions = physics steps in contact with a pillar;  "
        "detour = path length / straight-line distance"
    )

    # --- plots ----------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), tight_layout=True)

    ax = axes[0, 0]
    odor_values = np.linspace(-1.5, 1.5, 41)
    odor_turn = control_law(model, all_states, "odor", odor_values)
    ax.plot(odor_values, odor_turn[:, 0], lw=2, color="tab:orange")
    ax.fill_between(
        odor_values,
        odor_turn[:, 0] - odor_turn[:, 1],
        odor_turn[:, 0] + odor_turn[:, 1],
        color="tab:orange",
        alpha=0.15,
    )
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set(
        xlabel="odour asymmetry   [+ = food smells stronger on the left]",
        ylabel="turn command  (+ = turn left)",
        title="1. Smell law: turn toward the food",
    )

    ax = axes[0, 1]
    vis_values = np.linspace(-1.0, 1.0, 41)
    vis_turn = control_law(model, all_states, "vision", vis_values)
    ax.plot(vis_values, vis_turn[:, 0], lw=2, color="tab:green")
    ax.fill_between(
        vis_values,
        vis_turn[:, 0] - vis_turn[:, 1],
        vis_turn[:, 0] + vis_turn[:, 1],
        color="tab:green",
        alpha=0.15,
    )
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set(
        xlabel="visual mass asymmetry   [+ = pillar seen on the left]",
        ylabel="turn command  (+ = turn left)",
        title="2. Sight law: turn away from the obstacle",
    )

    ax = axes[1, 0]
    labels = [r["condition"] for r in rows]
    x = np.arange(len(labels))
    found_rate = [r["found"] / r["n"] * 100 for r in rows]
    bars = ax.bar(x, found_rate, color=["tab:blue", "tab:orange", "tab:green"])
    ax.bar_label(bars, fmt="%.0f%%")
    ax2 = ax.twinx()
    ax2.plot(x, [r["collisions"] for r in rows], "ko--", label="collisions")
    ax2.set_ylabel("mean collision steps")
    ax.set(
        xticks=x,
        xticklabels=labels,
        ylabel="food found (%)",
        ylim=(0, 115),
        title="3. Each sense breaks the task its own way",
    )
    ax2.legend(fontsize=8, loc="upper right")

    ax = axes[1, 1]
    colors = {"found": "tab:green", "timeout": "tab:orange",
              "fell": "tab:red", "out_of_bounds": "tab:purple"}
    seen = set()
    for ep in results["both"]:
        p = ep["path"]
        c = colors.get(ep["outcome"], "0.5")
        ax.plot(p[:, 0], p[:, 1], color=c, lw=1.1, alpha=0.8,
                label=ep["outcome"] if ep["outcome"] not in seen else None)
        seen.add(ep["outcome"])
        ax.plot(*ep["source"], marker="*", color="tab:orange", ms=11, mec="k", mew=0.4)
        for px, py in ep["pillars"]:
            ax.add_patch(plt.Circle((px, py), 0.8, color="0.25", alpha=0.55))
    ax.plot(0, 0, "ko", ms=7)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set(
        xlabel="x (mm)", ylabel="y (mm)",
        title="4. Paths with both senses\ncircle = start, star = food, grey = pillars",
    )
    ax.legend(fontsize=8)

    out_png = ANALYSIS / f"{args.run}_forage.png"
    fig.savefig(out_png, dpi=140)
    print(f"\nplot -> {out_png}")


if __name__ == "__main__":
    main()

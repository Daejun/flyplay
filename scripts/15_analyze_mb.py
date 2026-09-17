"""Did the fly learn which odour to avoid, and does it need compartments to?

Reads the sweep `scripts/14_run_conditioning.py` wrote and measures two things
that only make sense next to each other.

**The behaviour.** The preference index is where the fly ended up relative to
the two sources, +1 for odour B and -1 for odour A, scored over the second half
of each test trial. Three blocks of it: naive, after training on CS+ = A, and
after the contingency is reversed. Learning is not a single number here, it is
a *shape* -- naive near zero, positive after A is punished, negative again
after B is. A script that reported only the middle block could not tell an
association apart from a fly that happens to like odour B.

It is a noisy readout and the noise is structural, not a bug: a fly commits to
whichever gradient it catches first and then walks more or less straight, so
one trial is close to a binary choice (measured: sd 0.58 within a naive fly).
Every mean below therefore carries its standard error and the number of trials
behind it, and a difference smaller than the error bar is not a result.

**The circuit.** The mushroom-body valence separation -- unpaired odour minus
paired odour -- says whether the memory that should be driving that behaviour
exists at all, which is the difference between "the fly did not learn" and
"the fly learned and did not act on it". The two ablations pin it down:

    shared dopamine   one global signal instead of per-compartment. Both
                      compartments then get the same scalar and the same
                      Kenyon cells, so their weights cannot diverge and the
                      separation is exactly zero -- an identity, not noise.
    frozen            no plasticity. The floor, and the control for the
                      innate chemotaxis law on its own.

Both ablations therefore steer on innate attraction alone, and for the same
seed they walk the same path to the last written digit (checked on the smoke
sweep). They are not the same circuit, though. Frozen changes nothing; shared
dopamine changes a great deal -- it depresses the paired odour's synapses in
*both* compartments at once, identically, so the difference between them that
valence is made of never appears. Panel 4 draws that. It also shows up in
behaviour during training, which the preference index never scores: a fly that
cannot learn avoidance keeps walking into the punished odour, and the dopamine
column counts how long it was punished for.

**The extinction sweep** (`--dir out/mb_extinction`) asks a different question
of the same readouts: can an omission-of-punishment signal make reversal as
strong as acquisition? Its measure is the *symmetry* column, |post-reversal| /
|post-train| -- 1.0 means the fly unlearns as well as it learns. Panel 4 then
draws both compartments for odour A in the no-extinction control against the
extinction condition: extinction should leave the approach synapses exactly
where passive decay leaves them and pull avoid down to meet them, two opposing
memories side by side rather than one memory erased.

**The colour sweeps** (`--dir out/mb_colour`, `--dir out/mb_brightness`) are
the floor-colour assay of Vogt et al. (2014, 2016) and are scored their way:
one test per fly, two reciprocal groups, and the learning index
LI = [PI(CS+ second) - PI(CS+ first)] / 2 with PI the preference for the second
stimulus -- negative for avoiding what was punished, positive for seeking what
was rewarded. Conditions that leave the learned visual valence at exactly zero
must walk exactly the frozen fly's path, and the table says whether they did.

    python scripts/15_analyze_mb.py
    python scripts/15_analyze_mb.py --dir out/mb_extinction
    python scripts/15_analyze_mb.py --dir out/mb_colour

Writes out/analysis/mushroom_body.png, or mushroom_body_<dir>.png for any
directory other than out/mb.
"""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = _bootstrap.OUT / "mb"
ANALYSIS = _bootstrap.OUT / "analysis"

#: Test blocks, in the order `run_session` produces them, with column headings.
TEST_BLOCKS = (("naive", "naive PI"), ("trained", "post-train"),
               ("reversed", "post-reversal"))
#: Fixed per condition so the three panels agree on which line is which.
COLORS = {"baseline": "tab:blue", "shared_dan": "tab:orange", "frozen": "tab:green",
          "differential": "tab:blue", "extinction": "tab:purple",
          "extinction_wrong": "tab:red"}
#: (linestyle, linewidth) per condition. Not decoration: both ablations leave
#: valence at exactly zero, so both flies steer on innate attraction alone and
#: their trajectories come out identical -- verified bit for bit on every trial
#: of the smoke sweep. Drawn with the same style, whichever is plotted last
#: would simply hide the other and look like missing data.
STYLES = {"baseline": ("-", 1.8), "shared_dan": ("-", 1.6), "frozen": ("--", 1.3),
          "differential": ("-", 1.6), "extinction": ("-", 1.9),
          "extinction_wrong": ("--", 1.4)}
#: Short block names for the x-axis: naive / train / test / reversal / test.
SHORT = {"naive": "naive", "trained": "test", "reversed": "test"}
#: Room above the axes for the block names `mark_blocks` writes there.
TITLE_PAD = 20


def load(directory: Path) -> tuple[list[dict], dict]:
    """Every fly in a sweep directory, plus its meta.json if there is one."""
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*_seed*.json"))
    ]
    if not records:
        raise SystemExit(
            f"No results in {directory}. Run:\n"
            f"  python scripts/14_run_conditioning.py"
        )
    meta_path = directory / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return records, meta


def by_condition(records: list[dict]) -> dict[str, list[dict]]:
    """Group flies by condition, keeping the order they were first seen."""
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record["condition"], []).append(record)
    return grouped


def blocks_of(record: dict) -> list[tuple[str, int, int]]:
    """(name, first trial index, last trial index) for each block, in order."""
    spans: list[tuple[str, int, int]] = []
    for trial in record["trials"]:
        if spans and spans[-1][0] == trial["block"]:
            spans[-1] = (spans[-1][0], spans[-1][1], trial["index"])
        else:
            spans.append((trial["block"], trial["index"], trial["index"]))
    return spans


def sem(values) -> float:
    """Standard error of the mean. NaN for a single sample, on purpose."""
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return float("nan")
    return float(values.std(ddof=1) / np.sqrt(values.size))


def pooled_preference(records: list[dict], block: str) -> np.ndarray:
    """Every test trial's preference in one block, pooled across flies.

    Pooled rather than averaged per fly because the per-trial spread is the
    dominant term -- with sd ~0.6 a six-trial block leaves a s.e.m. of 0.24 on
    a single fly, so the fly means being averaged are themselves noise. The
    cost is that trials within a fly share one memory and are not independent,
    which makes this s.e.m. a lower bound; the flies column is there so that is
    visible.
    """
    return np.array(
        [
            trial["preference"]
            for record in records
            for trial in record["trials"]
            if trial["block"] == block and trial["preference"] is not None
        ],
        dtype=float,
    )


def fly_separation(records: list[dict], block: str) -> np.ndarray:
    """Per fly: valence of the unpaired odour minus the paired one, over a block.

    One value per fly rather than per trial. The valence readout is a property
    of the synapses, not of where the fly walked, so it barely moves inside a
    block -- pooling trials here would divide by a sqrt(n) that was never
    earned. Signed by the contingency, so positive always means "held apart in
    the direction it was taught", whichever odour that is.
    """
    values = []
    for record in records:
        paired = record["cs_plus"]
        other = "B" if paired == "A" else "A"
        per_trial = [
            trial["valence"][other] - trial["valence"][paired]
            for trial in record["trials"]
            if trial["block"] == block
        ]
        if per_trial:
            values.append(float(np.mean(per_trial)))
    return np.array(values, dtype=float)


def preference_series(records: list[dict]) -> dict[int, np.ndarray]:
    """{trial index: preference across flies}, test trials only.

    Keyed by trial index so the three conditions land on the same x-axis: every
    fly in a sweep runs the same protocol, so index 7 is the same point in the
    experiment for all of them.
    """
    series: dict[int, list[float]] = {}
    for record in records:
        for trial in record["trials"]:
            if trial["preference"] is not None:
                series.setdefault(trial["index"], []).append(trial["preference"])
    return {k: np.array(v, dtype=float) for k, v in sorted(series.items())}


def mbon_series(records: list[dict], odour: str, compartment: str = "approach"):
    """(indices, mean, s.e.m.) of one MBON's output across flies, per trial."""
    series: dict[int, list[float]] = {}
    for record in records:
        for trial in record["trials"]:
            series.setdefault(trial["index"], []).append(
                trial["mbon"][odour][compartment]
            )
    indices = np.array(sorted(series), dtype=float)
    values = [np.array(series[int(i)], dtype=float) for i in indices]
    return (
        indices,
        np.array([v.mean() for v in values]),
        np.array([sem(v) for v in values]),
    )


def mark_blocks(ax, spans) -> None:
    """Vertical rules at the block boundaries, named along the top.

    The names go just above the axes, so every panel that uses this needs its
    title padded out of the way by `TITLE_PAD`.
    """
    for name, lo, hi in spans:
        if lo > 1:
            ax.axvline(lo - 0.5, color="0.6", lw=0.8, ls=":")
        ax.text(
            0.5 * (lo + hi), 1.01,
            SHORT.get(name, name.replace("train CS+", "train ")),
            transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=8, color="0.35",
        )


#: Where the colour sweep writes; the brightness sweep borrows its intact hue
#: baseline from here rather than running the same flies twice.
COLOUR_DIR = _bootstrap.OUT / "mb_colour"
#: Conditions whose paths must equal another's exactly (see 14_run_conditioning).
IDENTITY_OF = {
    "colour_shared_dan": "colour_frozen",
    "colour_gd_blocked": "colour_frozen",
    "colour_punish_dan_blocked": "colour_frozen",
    "colour_reward_dan_blocked": "colour_frozen",
}
#: Swatches for the path panel, close to the tiles' paint.
SWATCH = {"blue": "#2f5fd6", "green": "#3aa843", "dim_blue": "#1d3a80",
          "dim_green": "#236b29", "green_tenth": "#0d2a10", "grey": "#808080"}


def visual_scale() -> float:
    """Normalisation of visual valence: full depression of a colour's code.

    `flyplay.mushroom_body` scales the visual code by k_visual / k_olfactory,
    so a colour whose every synapse is depressed reads this, not 1.
    """
    from flyplay.olfactory import OlfactoryFrontEnd
    from flyplay.visual_pathway import VisualFrontEnd

    return VisualFrontEnd().k / OlfactoryFrontEnd(2).k


def colour_under(record: dict, trial: dict) -> list[str]:
    """The colour the fly stood on at every step of a colour trial.

    Recovered from the stored floor pattern and path, with the same rule
    `flyplay.arena.ColourFloor.colour_at` uses.
    """
    pattern = trial["floor"]
    path = np.asarray(trial["path"], dtype=float)
    if pattern[0] == "uniform":
        return [pattern[1]] * len(path)
    tile = record["config"]["tile_size"]
    parity = (np.floor(path[:, 0] / tile).astype(int)
              + np.floor(path[:, 1] / tile).astype(int)) % 2
    return [pattern[1 + p] for p in parity]


def test_trial(record: dict) -> dict:
    return next(t for t in record["trials"] if t["kind"] == "test")


def learning_index(flies: list[dict]) -> tuple[float, float, np.ndarray]:
    """(LI, s.e.m., per-fly preference toward its own CS+)."""
    first, second = flies[0]["stimuli"]
    to_second = [test_trial(r)["preference"] for r in flies if r["cs_plus"] == second]
    to_first = [test_trial(r)["preference"] for r in flies if r["cs_plus"] == first]
    toward = np.array(to_second + [-v for v in to_first], dtype=float)
    if not to_second or not to_first:
        return float("nan"), sem(toward), toward
    return float((np.mean(to_second) - np.mean(to_first)) / 2), sem(toward), toward


def analyze_colour(args, records: list[dict], meta: dict) -> None:
    """The colour and brightness sweeps: LI table and four panels."""
    grouped = by_condition(records)
    order = [c for c in meta.get("conditions", []) if c in grouped]
    order += [c for c in grouped if c not in order]
    # The brightness sweep's hue task has no intact baseline of its own.
    if "colour" not in grouped and any(c.startswith("hue") for c in grouped):
        borrowed = [r for r in load(COLOUR_DIR)[0] if r["condition"] == "colour"] \
            if COLOUR_DIR.exists() and any(COLOUR_DIR.glob("colour_seed*.json")) else []
        if borrowed:
            grouped = {"colour": borrowed, **grouped}
            order = ["colour"] + order
    labels = {c: rs[0]["label"] for c, rs in grouped.items()}
    scale = visual_scale()

    print(f"{args.dir}: {len(records)} flies"
          + (f", sweep status '{meta['status']}'" if "status" in meta else ""))
    name_width = max(24, max(len(labels[c]) for c in order))
    print(f"\n{'condition':<{name_width}} | {'stimuli':<19} | {'LI':>13} | {'flies':>9} | "
          f"{'CS+ valence':>12} | {'dopamine/CS+':>12} | identity")
    print("-" * (name_width + 100))
    summary = {}
    for condition in order:
        flies = grouped[condition]
        li, spread, toward = learning_index(flies)
        first, second = flies[0]["stimuli"]
        n_second = sum(r["cs_plus"] == second for r in flies)
        # Valence of each fly's own CS+ at the test, in units of full depression.
        cs_valence = np.array([test_trial(r)["valence"][r["cs_plus"]] / scale for r in flies])
        dopamine = [t["dopamine_seconds"] for r in flies for t in r["trials"] if t["kind"] == "train"]
        reference = IDENTITY_OF.get(condition)
        identity = ""
        if reference and reference in grouped:
            theirs = {r["seed"]: test_trial(r)["path"] for r in grouped[reference]}
            same = [theirs.get(r["seed"]) == test_trial(r)["path"] for r in flies]
            identity = (f"paths = {labels[reference]}: "
                        + ("identical" if all(same) else f"{sum(same)}/{len(same)} identical"))
        summary[condition] = {"li": li, "sem": spread, "toward": toward}
        print(f"{labels[condition]:<{name_width}} | {first + ' / ' + second:<19} | "
              f"{f'{li:+.2f} +/-{spread:.2f}':>13} | {f'{n_second}+{len(flies) - n_second}':>9} | "
              f"{f'{cs_valence.mean():+.2f}':>12} | {f'{np.mean(dopamine):.1f}s':>12} | {identity}")
    print(
        "\nLI = [PI(CS+ second) - PI(CS+ first)] / 2, PI = time on the second stimulus minus\n"
        "     time on the first over the test (1-90 s). Negative = avoids the punished colour,\n"
        "     positive = seeks the rewarded one. Spread: s.e.m. of each fly's preference toward\n"
        "     its own CS+. flies = CS+ second + CS+ first.\n"
        "CS+ valence = learned valence of each fly's CS+ at the test, in units of full\n"
        "     depression of that colour's visual code (-1 = every synapse silent)."
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), tight_layout=True)

    ax = axes[0, 0]
    x = np.arange(len(order))
    means = [summary[c]["li"] for c in order]
    errors = [summary[c]["sem"] for c in order]
    bars = ax.bar(x, means, 0.6, yerr=errors, capsize=4, color="0.55", alpha=0.85)
    ax.bar_label(bars, fmt="%+.2f", fontsize=8, padding=3)
    for i, condition in enumerate(order):
        toward = summary[condition]["toward"]
        # Each dot is one fly's preference toward its own CS+; the bar is their
        # LI. Jittered sideways only so that identical flies do not stack.
        jitter = np.linspace(-0.18, 0.18, toward.size) if toward.size > 1 else [0.0]
        ax.scatter(i + np.asarray(jitter), toward, s=14, color="black", zorder=3, alpha=0.7)
    ax.axhline(0, color="0.3", lw=1.0)
    ax.set(xticks=x, ylabel="learning index   [- avoid CS+, + seek CS+]", ylim=(-1.05, 1.05))
    ax.set_xticklabels([labels[c] for c in order], rotation=25, ha="right", fontsize=8)
    ax.set_title("1. Learning index per condition (bar) and per fly (dots)")

    ax = axes[0, 1]
    # Vogt et al. (2014) found choices settle within about 20 s; this is the
    # same readout over time: fraction of flies on their CS+ minus on their CS-.
    for condition in order:
        flies = grouped[condition]
        curves = []
        for record in flies:
            trial = test_trial(record)
            under = colour_under(record, trial)
            cs_minus = next(s for s in record["stimuli"] if s != record["cs_plus"])
            curves.append([(u == record["cs_plus"]) - (u == cs_minus) for u in under])
        length = min(len(c) for c in curves)
        curve = np.mean([c[:length] for c in curves], axis=0)
        # One-second running mean: the per-step value is +/-1 per fly.
        window = 100
        smooth = np.convolve(curve, np.ones(window) / window, mode="valid")
        seconds = np.arange(smooth.size) / 100.0 + window / 200.0
        ax.plot(seconds, smooth, lw=1.6, label=labels[condition])
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set(xlabel="time in test (s)", ylabel="on CS+ minus on CS-   (fraction of flies)",
           ylim=(-1.05, 1.05))
    ax.set_title("2. The choice over the 90 s test (1 s running mean)")
    ax.legend(fontsize=7, loc="lower right", ncol=2)

    ax = axes[1, 0]
    # The memory itself, trial by trial: learned valence of each fly's CS+ and
    # CS-. Training trials are 60 s with a dopamine pulse every 5 s, so this is
    # sampled at the end of each trial.
    for condition in order:
        flies = grouped[condition]
        n = min(len(r["trials"]) for r in flies)
        plus = np.array([[t["valence"][r["cs_plus"]] / scale for t in r["trials"][:n]] for r in flies])
        minus = np.array([[t["valence"][next(s for s in r["stimuli"] if s != r["cs_plus"])] / scale
                           for t in r["trials"][:n]] for r in flies])
        line, = ax.plot(range(1, n + 1), plus.mean(axis=0), "o-", ms=4, lw=1.5,
                        label=f"{labels[condition]}: CS+")
        ax.plot(range(1, n + 1), minus.mean(axis=0), "s--", ms=3, lw=1.0,
                color=line.get_color(), alpha=0.7, label=f"{labels[condition]}: CS-")
    ax.axhline(0, color="0.6", lw=0.8)
    kinds = [t["kind"] for t in grouped[order[0]][0]["trials"]]
    ax.set_xticks(range(1, len(kinds) + 1), [k[:5] for k in kinds], fontsize=8)
    ax.set(xlabel="trial (end of)", ylabel="learned valence   [-1 = colour fully depressed]",
           ylim=(-1.05, 1.05))
    ax.set_title("3. What the mushroom body holds for CS+ and CS-")
    ax.legend(fontsize=6, ncol=2, loc="lower left")

    ax = axes[1, 1]
    # One fly's test, drawn over the floor it walked on.
    showcase = next((c for c in order if c in ("colour", "colour_reward", "brightness")), order[0])
    record = grouped[showcase][0]
    trial = test_trial(record)
    path = np.asarray(trial["path"], dtype=float)
    tile = record["config"]["tile_size"]
    lo = np.floor(path.min(axis=0) / tile) * tile - tile
    hi = np.ceil(path.max(axis=0) / tile) * tile + tile
    _, even, odd = trial["floor"]
    for gx in np.arange(lo[0], hi[0], tile):
        for gy in np.arange(lo[1], hi[1], tile):
            parity = (int(np.floor(gx / tile)) + int(np.floor(gy / tile))) % 2
            colour = (even, odd)[parity]
            ax.add_patch(plt.Rectangle((gx, gy), tile, tile, color=SWATCH.get(colour, "0.5"),
                                       alpha=0.55 if colour == record["cs_plus"] else 0.25, lw=0))
    ax.plot(path[:, 0], path[:, 1], color="black", lw=1.2)
    ax.plot(*path[0], "o", color="white", mec="black", ms=6)
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), aspect="equal", xlabel="x (mm)", ylabel="y (mm)")
    ax.set_title(f"4. {labels[showcase]}, seed {record['seed']}: CS+ = {record['cs_plus']} "
                 f"(darker squares), PI toward it {-(1 if record['cs_plus'] == record['stimuli'][0] else -1) * trial['preference']:+.2f}",
                 fontsize=10)

    suffix = f"_{args.dir.name}"
    out_png = ANALYSIS / f"mushroom_body{suffix}.png"
    fig.savefig(out_png, dpi=140)
    print(f"\nplot -> {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dir", type=Path, default=OUT)
    args = parser.parse_args()

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    records, meta = load(args.dir)
    if records[0]["config"].get("modality") == "colour":
        analyze_colour(args, records, meta)
        return
    grouped = by_condition(records)
    labels = {c: rs[0]["label"] for c, rs in grouped.items()}
    order = [c for c in meta.get("conditions", []) if c in grouped]
    order += [c for c in grouped if c not in order]

    first = records[0]
    print(
        f"{args.dir}: {len(records)} flies, "
        f"{len(first['trials'])} trials each, CS+ = {first['cs_plus']}, "
        f"n_train={first['n_train']} n_test={first['n_test']}"
        + (f", sweep status '{meta['status']}'" if "status" in meta else "")
    )
    lengths = {len(r["trials"]) for r in records}
    if len(lengths) > 1:
        print(
            f"  WARNING: mixed protocols in this directory ({sorted(lengths)} "
            f"trials); the trial-indexed panels will not line up"
        )

    # --- table -----------------------------------------------------------
    name_width = max(16, max(len(labels[c]) for c in order))
    print(f"\n{'condition':<{name_width}} | {'naive PI':>13} | {'post-train':>13} | "
          f"{'post-reversal':>14} | {'symmetry':>8} | {'val sep train':>15} | "
          f"{'val sep rev':>15} | {'trials':>6} | {'flies':>5}")
    print("-" * (name_width + 112))
    table = {}
    for condition in order:
        flies = grouped[condition]
        table[condition] = {
            "pi": {b: pooled_preference(flies, b) for b, _ in TEST_BLOCKS},
            "separation": fly_separation(flies, "trained"),
            "separation_rev": fly_separation(flies, "reversed"),
        }
        pi = table[condition]["pi"]
        cells = [
            f"{v.mean():+.2f} +/-{sem(v):.2f}" if v.size else "-"
            for v in (pi[b] for b, _ in TEST_BLOCKS)
        ]
        # How well the fly unlearns relative to how well it learned. Undefined
        # when it never learned: dividing by a post-train PI near zero is noise.
        acquired, reversed_ = pi["trained"], pi["reversed"]
        symmetry = (
            f"{abs(reversed_.mean()) / abs(acquired.mean()):.2f}"
            if acquired.size and reversed_.size and abs(acquired.mean()) >= 0.1
            else "-"
        )
        spread = table[condition]["separation"]
        spread_rev = table[condition]["separation_rev"]
        print(
            f"{labels[condition]:<{name_width}} | {cells[0]:>13} | {cells[1]:>13} | "
            f"{cells[2]:>14} | {symmetry:>8} | "
            f"{f'{spread.mean():+.3f} +/-{sem(spread):.3f}':>15} | "
            f"{f'{spread_rev.mean():+.3f} +/-{sem(spread_rev):.3f}':>15} | "
            f"{max(v.size for v in pi.values()):>6} | {len(flies):>5}"
        )
    print(
        "\nPI = preference index: +1 the fly ended at odour B, -1 at odour A, "
        "scored over the\n"
        f"     second half of each test trial. CS+ is odour {first['cs_plus']}, "
        "so learning reads as post-train > 0\n"
        "     and post-reversal < 0. Spread is s.e.m. over pooled test trials "
        "(the trials column);\n"
        "     trials inside one fly share a memory, so it is a lower bound."
    )
    print(
        "symmetry = |post-reversal| / |post-train|. 1.0 means the fly unlearns "
        "as well as it learned."
    )
    print(
        "val sep = mushroom-body valence of the first CS- minus the first CS+ "
        "(B - A), s.e.m. across\n"
        "     flies, over the post-training and post-reversal test blocks. "
        "Positive after training and\n"
        "     negative after reversal is a memory that followed the contingency; "
        "zero means the circuit\n"
        "     never represented which odour was punished, whatever the fly did "
        "with its legs."
    )

    # What the circuit holds for the first CS+ when the reversal test starts --
    # the last trial of reversal training. This is where passive forgetting and
    # extinction differ: residual depression of approach, and whether avoid has
    # come down to cancel it.
    paired = first["cs_plus"]
    print(f"\ncircuit for odour {paired} when the reversal test starts "
          f"(mean over flies):")
    print(f"{'condition':<{name_width}} | {'valence':>8} | {'approach':>8} | "
          f"{'avoid':>6} | {'punishment/train':>16} | {'omission/expose':>15}")
    for condition in order:
        rows, punish, omit = [], [], []
        for record in grouped[condition]:
            training = [t for t in record["trials"]
                        if t["block"].startswith("train") and t["block"].endswith(
                            "B" if paired == "A" else "A")]
            if training:
                last = training[-1]
                rows.append((last["valence"][paired],
                             last["mbon"][paired]["approach"],
                             last["mbon"][paired]["avoid"]))
            punish += [t["dopamine_seconds"] for t in record["trials"]
                       if t["kind"] == "train"]
            omit += [t.get("omission_dopamine", 0.0) for t in record["trials"]
                     if t["kind"] == "expose"]
        if not rows:
            continue
        v, app, avd = np.mean(rows, axis=0)
        omit_cell = f"{np.mean(omit):.2f}" if omit else "-"
        print(f"{labels[condition]:<{name_width}} | {v:>+8.3f} | {app:>8.3f} | "
              f"{avd:>6.3f} | {np.mean(punish):>15.2f}s | {omit_cell:>15}")
    print("punishment/train = seconds of punishment dopamine per training trial; "
          "omission/expose =\n     omission dopamine-seconds per unpunished "
          "exposure (differential protocols only).")

    # --- plots -----------------------------------------------------------
    spans = blocks_of(first)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), tight_layout=True)

    ax = axes[0, 0]
    # Baseline last. The naive block runs before anything is taught, so all
    # three conditions walk identical paths there and whichever is drawn last
    # is the only colour visible -- with baseline first, its naive trials
    # disappeared under frozen and looked like missing data.
    for condition in sorted(order, key=lambda c: c == "baseline"):
        series = preference_series(grouped[condition])
        color = COLORS.get(condition, "0.4")
        style, lw = STYLES.get(condition, ("-", 1.5))
        drawn = False
        # One line segment per test block: a single line would join the two
        # blocks straight across the training between them, drawing a learning
        # curve through trials that were never scored.
        for _name, lo, hi in spans:
            keys = [k for k in series if lo <= k <= hi]
            if not keys:
                continue
            mean = np.array([series[k].mean() for k in keys])
            spread = np.array([sem(series[k]) for k in keys])
            ax.plot(keys, mean, marker="o", ls=style, ms=4, lw=lw, color=color,
                    label=labels[condition] if not drawn else None)
            ax.fill_between(keys, mean - spread, mean + spread,
                            color=color, alpha=0.15)
            drawn = True
    ax.axhline(0, color="0.6", lw=0.8)
    mark_blocks(ax, spans)
    ax.set(
        xlabel="trial", ylabel="preference index   [+ = toward odour B]",
        ylim=(-1.05, 1.05),
    )
    ax.set_title("1. Preference through the protocol", pad=TITLE_PAD)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9)

    ax = axes[0, 1]
    x = np.arange(len(order))
    width = 0.38
    for offset, (block, heading) in zip((-width / 2, width / 2), TEST_BLOCKS[1:]):
        means = [table[c]["pi"][block].mean() for c in order]
        errors = [sem(table[c]["pi"][block]) for c in order]
        bars = ax.bar(x + offset, means, width, yerr=errors, capsize=4,
                      label=heading,
                      color="0.35" if offset < 0 else "tab:red", alpha=0.85)
        ax.bar_label(bars, fmt="%+.2f", fontsize=8, padding=2)
    ax.axhline(0, color="0.3", lw=1.0)
    ax.set(
        xticks=x, xticklabels=[labels[c] for c in order],
        ylabel="preference index",
        title="2. Reversal: the preference follows the contingency\n"
              "CS+ = A, then CS+ = B",
    )
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    # Same shape as panel 2, one level down: the memory rather than the
    # behaviour. After reversal a memory that followed the contingency reads
    # negative, so the red bar is the one extinction should push down.
    for offset, key, heading, color in (
        (-0.19, "separation", "post-train", "0.35"),
        (+0.19, "separation_rev", "post-reversal", "tab:red"),
    ):
        means = [table[c][key].mean() for c in order]
        errors = [sem(table[c][key]) for c in order]
        bars = ax.bar(x + offset, means, 0.38, yerr=errors, capsize=4,
                      label=heading, color=color, alpha=0.85)
        ax.bar_label(bars, fmt="%+.3f", fontsize=8, padding=2)
    ax.axhline(0, color="0.3", lw=1.0)
    ax.set(
        xticks=x, xticklabels=[labels[c] for c in order],
        ylabel="valence(B) - valence(A)",
        title="3. What the mushroom body holds\n"
              "valence difference in each test block, mean over flies",
    )
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    # Both compartments' response to the first CS+, for baseline against
    # shared dopamine. This is the panel that says *why* shared dopamine fails,
    # and the answer is not that nothing happens. Its synapses move as far as
    # baseline's or further -- measured on the smoke sweep, approach to odour A
    # fell to 0.28 against baseline's 0.56, because a fly that never learns to
    # avoid keeps walking into the odour and keeps being punished (2.2-3.4 s of
    # dopamine a trial against 1.1 s). But the avoid compartment falls with it,
    # to the same value on every trial, so approach minus avoid never leaves
    # zero. The gap between the solid lines is the valence; the dashed lines
    # have none. The full sweep bears it out: approach to A bottoms out at
    # 0.02-0.03 per shared-dopamine fly against 0.45-0.52 for baseline, on
    # 2.30 s of dopamine per training trial against 0.33 s.
    #
    # In the extinction sweep the same panel shows the opposite point. The
    # dashed lines are the no-extinction control; the solid lines are
    # extinction, where omission dopamine pulls avoid down to meet the still-
    # depressed approach -- a second memory cancelling the first, not the first
    # one erased.
    if {"baseline", "shared_dan"} <= set(grouped):
        shown, title, filled = ["baseline", "shared_dan"], (
            f"4. One dopamine signal moves both compartments together "
            f"(odour {paired})"), "baseline"
    elif {"differential", "extinction"} <= set(grouped):
        shown, title, filled = ["extinction", "differential"], (
            f"4. Extinction adds an opposing memory, it does not erase "
            f"(odour {paired})"), "extinction"
    else:
        shown, title, filled = order[:2], f"4. Both compartments (odour {paired})", None
    for condition, (style, lw) in zip(shown, (("-", 1.9), ("--", 1.3))):
        curves = {}
        for compartment, color in (("approach", "tab:orange"), ("avoid", "tab:blue")):
            indices, mean, _spread = mbon_series(
                grouped[condition], paired, compartment
            )
            curves[compartment] = mean
            coincident = condition == "shared_dan" and compartment == "approach"
            # Under shared dopamine the two compartments are equal on every
            # trial, so a plain dashed line for each draws one line and the
            # legend promises two. The approach curve goes underneath as a wide
            # translucent band, so the orange showing round the blue dashes is
            # the visible statement that they coincide.
            ax.plot(indices, mean, "-" if coincident else style,
                    lw=5 if coincident else lw, color=color,
                    alpha=0.35 if coincident else 1.0,
                    label=f"{labels[condition]}: {compartment}"
                    + (" (under avoid, identical)" if coincident else ""))
        if condition == filled:
            ax.fill_between(indices, curves["approach"], curves["avoid"],
                            color="tab:orange", alpha=0.12,
                            label=f"{labels[condition]} valence (gap)")
    mark_blocks(ax, spans)
    ax.set(
        xlabel="trial",
        ylabel=f"MBON response to odour {paired}   [1.0 = resting weight]",
        ylim=(0, 1.08),
    )
    ax.set_title(title, pad=TITLE_PAD)
    ax.legend(fontsize=7, loc="lower left", ncol=2, framealpha=0.9)

    # One figure per sweep directory, so analysing the extinction sweep does not
    # overwrite the compartment sweep's figure.
    suffix = "" if args.dir.resolve() == OUT.resolve() else f"_{args.dir.name}"
    out_png = ANALYSIS / f"mushroom_body{suffix}.png"
    fig.savefig(out_png, dpi=140)
    print(f"\nplot -> {out_png}")


if __name__ == "__main__":
    main()

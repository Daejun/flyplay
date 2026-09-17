"""Run the conditioning protocols -- odour and colour -- over conditions and seeds.

One fly is one `ConditioningExperiment`: a body, a mushroom body, and a memory
that survives the trials. `flyplay.conditioning.run_session` walks it through
naive test / train / test / reversal train / test, and this script runs the
whole grid of (condition, seed) flies and writes the raw trial stream to disk
for `scripts/15_analyze_mb.py` to read.

Four experiments: two on smell, two on colour. The default is the first.

**Compartments** -- are separate dopamine inputs necessary?

    baseline     per-compartment dopamine, plasticity on.
    shared_dan   one global dopamine signal instead of per-compartment. Both
                 compartments then see the same scalar and the same Kenyon
                 cells, so their weights cannot diverge, valence stays at
                 exactly zero (an identity, not noise -- scripts/13_mb_check.py
                 measures 0.000 separation) and the fly cannot tell which
                 odour was punished.
    frozen       eta = recovery = 0, no plasticity at all. The floor: whatever
                 preference index this scores is what the innate chemotaxis
                 law produces on its own, with no mushroom body contributing.

Measured: acquisition +0.49, but reversal only -0.30, because the only way this
circuit unlearns is passive decay.

**Extinction** -- does an omission-of-punishment signal fix reversal? All
three use differential training (CS+ and CS- alternating), so the CS- trials
are the one thing that differs from the first experiment.

    differential      differential training, no omission signal. Controls for
                      the protocol change on its own.
    extinction        omission signal into the avoid compartment -- the wiring
                      Felsenberg et al. (2018) describe.
    extinction_wrong  the same signal into the approach compartment. Offline it
                      deepens the aversion instead of cancelling it
                      (13_mb_check.py), so compartment identity decides whether
                      the signal extinguishes or reinforces.

    python scripts/14_run_conditioning.py --experiment extinction

**Colour** -- the floor-colour assay of Vogt et al. (2014), same mushroom body
with visual Kenyon cells added (`flyplay.visual_pathway`). Whole-floor CS+ and
CS- periods of 60 s, four of each, then one 90 s checkerboard test. Seeds
alternate the contingency (even seeds: CS+ is the first stimulus; odd: the
second), so an even number of seeds gives Vogt's two reciprocal groups, and the
score is their learning index, LI = [PI(CS+ second) - PI(CS+ first)] / 2 with
PI the preference for the second stimulus -- negative when the flies avoid
what was punished.

    colour            punishment into the approach compartment (PPL1)
    colour_reward     sugar: reward into the avoid compartment (PAM)
    colour_frozen     no plasticity: the floor
    colour_shared_dan one dopamine signal for both compartments
    colour_gd_blocked visual Kenyon-cell output blocked (gamma-d, Vogt 2016)
    colour_punish_dan_blocked / colour_reward_dan_blocked
                      the reinforcing DAN population silenced (Vogt 2014)

The last four leave the eyes' learned valence at exactly zero, so the fly
steers exactly as a frozen one does and its path must match colour_frozen's on
the same seed to the last digit. Ten noisy flies would say less than that, so
they run on two seeds (`IDENTITY_OF`) and the summary checks the match.

    python scripts/14_run_conditioning.py --experiment colour --seeds 10

**Brightness** -- Vogt et al. (2016): colour memory needs one VPN type,
brightness memory another. The hue task (blue vs green, equally bright to this
model) runs with colour VPNs or brightness VPNs silenced -- its intact baseline
is the colour experiment's `colour` condition, not run twice -- and the
brightness task (green vs green at a tenth) with both families, with colour
VPNs silenced, and with brightness VPNs silenced. None of the silenced ones is
an exact null: rendered blue and green agree in brightness only to three
decimals, and a tenth-bright green renders slightly desaturated (balance 0.84
against 0.90), so each runs on every seed.

    python scripts/14_run_conditioning.py --experiment brightness --seeds 10

Seeds are subjects, not repeats. A seed redraws the Kenyon-cell wiring, the
spawn pose and the per-trial side randomisation, so the spread across seeds is
between-fly variation -- which is what the error bars in the analysis are made
of. The per-trial preference index is very noisy on its own (measured: sd 0.58
within a naive fly, because the fly commits to whichever gradient it catches
first and then walks in a straight line), so a single fly proves nothing and
the number of trials behind every mean is reported everywhere.

**Parallelism.** A trial costs 4.3 s of wall time (measured: 5 s of simulated
walking, 100 Hz control over 10 kHz physics), and the default sweep is
3 conditions x 5 seeds x 58 trials = 870 trials, about 62 minutes serially.
Every (condition, seed) fly is fully independent -- its own MuJoCo model, its
own mushroom body, its own RNG -- so they go out to a process pool. The default
sweep is 15 flies and this machine has 16 cores, so it fits in a single wave.
It does not cost what one fly costs, though: measured, 15 at once took 8.6 min,
each fly 440-510 s against about 250 s running alone -- still a 7x speedup
over the 62 min serial estimate. Why each worker ran at half speed was not
measured. This is a hybrid-core laptop part (Core Ultra 7 356H) and the web
viewer was also running, so core type, memory bandwidth and contention are all
candidates; do not read it as a fixed property of the code.

    python scripts/14_run_conditioning.py
    python scripts/14_run_conditioning.py --seeds 8 --jobs 12
    python scripts/14_run_conditioning.py --conditions baseline frozen
    python scripts/14_run_conditioning.py --eta 1.0 --recovery 0.01
    python scripts/14_run_conditioning.py --seeds 1 --n-train 3 --n-test 2

`--jobs 1` runs in this process and prints a line per trial instead, which is
what you want when a single fly is behaving oddly.

Writes <out>/meta.json (at the *start*, status "running", so anything opening
the directory mid-sweep knows what it is looking at) and one
<out>/<condition>_seed<n>.json per fly:

    {"condition", "label", "seed", "cs_plus", "n_train", "n_test", "reversal",
     "naive", "stimuli", "config": {every ConditioningConfig field},
     "wall_seconds",
     "trials": [{"kind": "train"|"expose"|"test", "index", "block", "cs_plus",
                 "odours", "time_near", "preference", "valence", "mbon",
                 "dopamine_seconds", "omission_dopamine", "path": [[x, y], ...],
                 "sources": {"A": [x, y], "B": [x, y]},
                 "floor": null | ["uniform", colour] | ["checker", even, odd]}]}

For a colour fly, "odours" and the keys of "time_near", "valence" and "mbon"
are the colours, "time_near" is seconds spent on each, and "sources" is empty.

Nothing in there is a numpy array. `path` is a plain list of [x, y] pairs
rounded to 3 decimals, which is a micrometre on a 20 mm arena. No summary
statistics are stored: the trial stream is the only source of truth and
15_analyze_mb.py derives everything from it.

The trial stream says what each fly *did*. What it *knows* -- the KC->MBON
weights themselves -- goes to <out>/memory/<condition>_seed<n>/after_<block>.npz
at the end of every block, and `ConditioningExperiment.load_memory` puts a
trained fly back together from one.
"""

import argparse
import dataclasses
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from flyplay.conditioning import (
    ODOUR_NAMES,
    ConditioningConfig,
    ConditioningExperiment,
    TrialResult,
    colour_config,
    preference_index,
    run_session,
)

OUT = _bootstrap.OUT / "mb"

#: Condition name -> ConditioningConfig overrides, plus a label for printing.
CONDITIONS = {
    "baseline": ({}, "baseline"),
    "shared_dan": ({"shared_dan": True}, "shared dopamine"),
    "frozen": ({"frozen": True}, "frozen"),
    "differential": ({"differential": True}, "differential"),
    "extinction": ({"differential": True, "extinction": "avoid"}, "extinction"),
    "extinction_wrong": (
        {"differential": True, "extinction": "approach"}, "extinction->approach"
    ),
    # Colour: entries whose overrides say "modality": "colour" are built with
    # `colour_config`, which supplies the Vogt protocol around them.
    "colour": ({"modality": "colour"}, "colour"),
    "colour_reward": ({"modality": "colour", "reinforcer": "reward"}, "colour reward"),
    "colour_frozen": ({"modality": "colour", "frozen": True}, "colour frozen"),
    "colour_shared_dan": ({"modality": "colour", "shared_dan": True}, "colour shared DA"),
    "colour_gd_blocked": ({"modality": "colour", "blocked": ("visual_kc",)}, "gamma-d blocked"),
    "colour_punish_dan_blocked": (
        {"modality": "colour", "blocked": ("punish_dan",)}, "PPL1 blocked"
    ),
    "colour_reward_dan_blocked": (
        {"modality": "colour", "reinforcer": "reward", "blocked": ("reward_dan",)},
        "PAM blocked (reward)",
    ),
    "hue_colour_vpn_off": (
        {"modality": "colour", "blocked": ("colour_vpn",)}, "hue, colour VPN off"
    ),
    "hue_brightness_vpn_off": (
        {"modality": "colour", "blocked": ("brightness_vpn",)}, "hue, bright VPN off"
    ),
    "brightness": (
        {"modality": "colour", "stimuli": ("green", "green_tenth")}, "brightness task"
    ),
    "brightness_colour_vpn_off": (
        {"modality": "colour", "stimuli": ("green", "green_tenth"),
         "blocked": ("colour_vpn",)},
        "bright, colour VPN off",
    ),
    "brightness_brightness_vpn_off": (
        {"modality": "colour", "stimuli": ("green", "green_tenth"),
         "blocked": ("brightness_vpn",)},
        "bright, bright VPN off",
    ),
}
#: Protocol shapes: training trials per block, test trials per block, and
#: whether the naive test and the reversal run.
ODOUR_PROTOCOL = {"n_train": 20, "n_test": 6, "naive": True, "reversal": True}
#: Vogt et al. (2014): four CS+ and four CS- periods, then one test. No naive
#: block -- the reciprocal groups cancel innate bias -- and no reversal.
COLOUR_PROTOCOL = {"n_train": 8, "n_test": 1, "naive": False, "reversal": False}
#: Named sets of conditions, where each writes by default, and its protocol.
EXPERIMENTS = {
    "compartments": (("baseline", "shared_dan", "frozen"), OUT, ODOUR_PROTOCOL),
    "extinction": (
        ("differential", "extinction", "extinction_wrong"),
        _bootstrap.OUT / "mb_extinction",
        ODOUR_PROTOCOL,
    ),
    "colour": (
        ("colour", "colour_reward", "colour_frozen", "colour_shared_dan",
         "colour_gd_blocked", "colour_punish_dan_blocked", "colour_reward_dan_blocked"),
        _bootstrap.OUT / "mb_colour",
        COLOUR_PROTOCOL,
    ),
    "brightness": (
        ("hue_colour_vpn_off", "hue_brightness_vpn_off",
         "brightness", "brightness_colour_vpn_off", "brightness_brightness_vpn_off"),
        _bootstrap.OUT / "mb_brightness",
        COLOUR_PROTOCOL,
    ),
}
#: Conditions whose behaviour is exactly that of another condition by
#: construction, and which one. They run on `IDENTITY_SEEDS` seeds and the
#: summary compares test paths digit for digit.
IDENTITY_OF = {
    "colour_shared_dan": "colour_frozen",
    "colour_gd_blocked": "colour_frozen",
    "colour_punish_dan_blocked": "colour_frozen",
    "colour_reward_dan_blocked": "colour_frozen",
}
IDENTITY_SEEDS = 2
#: Test blocks in the order `run_session` produces them.
TEST_BLOCKS = ("naive", "trained", "reversed")
#: `flyplay.conditioning.PROTOCOL` block keys -> the names written to disk.
#: The on-disk names predate the protocol table and 15_analyze_mb.py reads
#: them, so they are kept rather than churning every saved result.
TEST_BLOCK_NAMES = {"naive": "naive", "acquisition": "trained", "reversal": "reversed"}
#: Measured on this machine: wall seconds per simulated second, one process and
#: with a pool running. Odour, alone 0.86 (4.3 s for a 5 s trial) and about 1.7
#: per fly with 15 at once (the 440-510 s flies above). Colour, with the eyes
#: rendered at 10 Hz: 1.2-2.1 alone depending on the run, 3.7 per fly with 15 at
#: once (measured on 10 s trials) -- worse scaling than odour, and the eye
#: renders sharing one integrated GPU are the likely reason, not measured.
#: Estimates only.
WALL_PER_SIM_SECOND = {"odour": 0.86, "colour": 1.22}
WALL_PER_SIM_SECOND_POOLED = {"odour": 1.7, "colour": 3.7}


def sem(values) -> float:
    """Standard error of the mean. NaN for a single sample, on purpose."""
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return float("nan")
    return float(values.std(ddof=1) / np.sqrt(values.size))


def block_names(results: list[TrialResult], cs_plus: str, names=ODOUR_NAMES) -> list[str]:
    """Name the block each trial belongs to, from the block `run_session` set.

    This used to be inferred from where `kind` changed, which broke the moment
    differential training put "train" and "expose" trials in one block: every
    alternation looked like a new block. The protocol knows its blocks, so the
    results carry them now. A training block is named after the odour it
    punishes.
    """
    other = names[1] if cs_plus == names[0] else names[0]
    training = {"train": f"train CS+{cs_plus}", "reversal_train": f"train CS+{other}"}
    names = []
    for result in results:
        if result.block in TEST_BLOCK_NAMES:
            names.append(TEST_BLOCK_NAMES[result.block])
        elif result.block in training:
            names.append(training[result.block])
        else:
            raise ValueError(
                f"trial {result.index} has no protocol block ({result.block!r}); "
                f"results must come from run_session"
            )
    return names


def to_record(result: TrialResult, block: str) -> dict:
    """One trial as plain JSON types.

    Rounding is not cosmetic. An un-rounded path is 17 significant digits per
    coordinate and 500 samples per trial, which triples the file for precision
    the simulation does not have.
    """
    return {
        "kind": result.kind,
        "index": result.index,
        "block": block,
        "cs_plus": result.cs_plus,
        "odours": list(result.odours),
        "time_near": {k: round(float(v), 4) for k, v in result.time_near.items()},
        "preference": (
            None if result.preference is None else round(float(result.preference), 6)
        ),
        "valence": {k: round(float(v), 6) for k, v in result.valence.items()},
        "mbon": {
            odour: {name: round(float(value), 6) for name, value in per.items()}
            for odour, per in result.mbon.items()
        },
        "dopamine_seconds": round(float(result.dopamine_seconds), 4),
        "omission_dopamine": round(float(result.omission_dopamine), 4),
        "path": [[round(float(x), 3), round(float(y), 3)] for x, y in result.path],
        "sources": {
            k: [round(float(v[0]), 3), round(float(v[1]), 3)]
            for k, v in result.sources.items()
        },
        "floor": None if result.floor is None else list(result.floor),
    }


def separation(results: list[TrialResult], block: str, names: list[str],
               stimuli=ODOUR_NAMES) -> float:
    """Valence of the unpaired odour minus the paired one, over one block.

    Signed by the contingency rather than by odour name, so it reads the same
    way after a reversal: positive means the mushroom body holds the two
    odours apart in the direction it was taught.
    """
    paired = next((r.cs_plus for r in results if r.cs_plus is not None), stimuli[0])
    other = stimuli[1] if paired == stimuli[0] else stimuli[0]
    values = [
        r.valence[other] - r.valence[paired]
        for r, name in zip(results, names)
        if name == block
    ]
    return float(np.mean(values)) if values else float("nan")


def run_job(spec: dict, on_trial=None) -> dict:
    """Run one fly end to end and write its JSON. Returns a summary to print.

    This is the process-pool payload, so it takes plain dicts and returns a
    plain dict: the trial stream never crosses the process boundary, only the
    handful of numbers the parent needs for its progress line.
    """
    config = ConditioningConfig(**spec["config"])
    experiment = ConditioningExperiment(config, seed=spec["seed"])
    stimuli = experiment.stimulus_names
    memory_dir = (
        Path(spec["out"]) / "memory" / f"{spec['condition']}_seed{spec['seed']:02d}"
    )
    memory_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        results = run_session(
            experiment,
            n_train=spec["n_train"],
            n_test=spec["n_test"],
            cs_plus=spec["cs_plus"],
            reversal=spec["reversal"],
            naive=spec["naive"],
            on_trial=on_trial,
            memory_dir=memory_dir,
        )
    finally:
        experiment.close()
    elapsed = time.perf_counter() - started

    names = block_names(results, spec["cs_plus"], stimuli)
    path = Path(spec["out"]) / f"{spec['condition']}_seed{spec['seed']:02d}.json"
    path.write_text(
        json.dumps(
            {
                "condition": spec["condition"],
                "label": spec["label"],
                "seed": spec["seed"],
                "cs_plus": spec["cs_plus"],
                "n_train": spec["n_train"],
                "n_test": spec["n_test"],
                "reversal": spec["reversal"],
                "naive": spec["naive"],
                "stimuli": list(stimuli),
                "config": spec["config"],
                "wall_seconds": round(elapsed, 1),
                "trials": [to_record(r, n) for r, n in zip(results, names)],
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    by_block = {
        block: preference_index([r for r, n in zip(results, names) if n == block])
        for block in TEST_BLOCKS
    }
    return {
        "condition": spec["condition"],
        "label": spec["label"],
        "seed": spec["seed"],
        "file": path.name,
        "wall_seconds": elapsed,
        "n_trials": len(results),
        "pi": by_block,
        "separation": separation(results, "trained", names, stimuli),
        # Digest of every test path at full float precision, for the identity
        # conditions: equal digests mean the fly walked the same path exactly.
        "test_paths": hashlib.sha1(b"".join(
            np.ascontiguousarray(r.path).tobytes() for r in results if r.kind == "test"
        )).hexdigest(),
        "modality": config.modality,
        "cs_plus": spec["cs_plus"],
        "stimuli": list(stimuli),
    }


def progress_printer(spec: dict, total: int):
    """A serial-mode `on_trial` callback. Only used when --jobs 1."""
    state = {"n": 0}

    def report(result: TrialResult) -> None:
        state["n"] += 1
        score = (
            "     -" if result.preference is None else f"{result.preference:+.3f}"
        )
        print(
            f"  {spec['label']:<15} seed {spec['seed']:<2} "
            f"trial {state['n']:>3}/{total}  {result.kind:<5} PI {score}",
            flush=True,
        )

    return report


def write_meta(
    out: Path, args, specs: list[dict], trials_per_fly: int, **extra
) -> None:
    """Describe the sweep on disk.

    Written before the first fly starts, exactly as 11_train_forage.py does it:
    a sweep is tens of minutes long and anything that opens out/mb/ while it is
    running needs to know which conditions and seeds the half-written directory
    is going to contain.
    """
    (out / "meta.json").write_text(
        json.dumps(
            {
                "task": "conditioning",
                "conditions": list(dict.fromkeys(s["condition"] for s in specs)),
                "seeds": sorted({s["seed"] for s in specs}),
                "n_train": args.n_train,
                "n_test": args.n_test,
                "cs_plus": specs[0]["cs_plus"],
                "reversal": specs[0]["reversal"],
                "trials_per_fly": trials_per_fly,
                "n_flies": len(specs),
                "jobs": args.jobs,
                "eta": args.eta,
                "recovery": args.recovery,
                **extra,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def learning_index(rows: list[dict]) -> tuple[float, float, int, int]:
    """Vogt et al.'s LI over reciprocal groups, with its spread.

    PI is each fly's preference for the second stimulus. LI =
    [mean PI(CS+ second) - mean PI(CS+ first)] / 2: negative when flies avoid
    what was paired with punishment, positive when they seek what was paired
    with sugar. The spread is the s.e.m. of each fly's preference toward its
    own CS+, which is what the LI averages.
    """
    if not rows:
        return float("nan"), float("nan"), 0, 0
    first, second = rows[0]["stimuli"]
    to_second = [r["pi"]["trained"] for r in rows if r["cs_plus"] == second]
    to_first = [r["pi"]["trained"] for r in rows if r["cs_plus"] == first]
    li = (np.mean(to_second) - np.mean(to_first)) / 2 if to_second and to_first else float("nan")
    toward_cs_plus = to_second + [-v for v in to_first]
    return float(li), sem(toward_cs_plus), len(to_second), len(to_first)


def print_colour_summary(args, summaries: list[dict]) -> None:
    print(f"\n{'condition':<26} | {'LI':>14} | {'groups':>17} | {'valence sep':>17}")
    print("-" * 84)
    for condition in args.conditions:
        rows = [s for s in summaries if s["condition"] == condition]
        if not rows:
            continue
        li, spread, n_second, n_first = learning_index(rows)
        sep = np.array([r["separation"] for r in rows], dtype=float)
        first, second = rows[0]["stimuli"]
        line = (
            f"{rows[0]['label']:<26} | {f'{li:+.2f} +/-{spread:.2f}':>14} | "
            f"{f'{n_second}x{second[:5]}+ {n_first}x{first[:5]}+':>17} | "
            f"{f'{sep.mean():+.4f} +/-{sem(sep):.4f}':>17}"
        )
        reference = IDENTITY_OF.get(condition)
        if reference:
            theirs = {s["seed"]: s["test_paths"] for s in summaries
                      if s["condition"] == reference}
            matched = [theirs.get(r["seed"]) == r["test_paths"] for r in rows]
            verdict = ("identical" if all(matched) else "DIFFERENT") if theirs else "no reference run"
            line += f"  paths vs {reference}: {verdict}"
        print(line)
    print(
        "\nLI = [PI(CS+ second) - PI(CS+ first)] / 2, PI = preference for the\n"
        "     second stimulus over the test (Vogt et al. 2014). Negative = the\n"
        "     flies avoid what was punished; positive = seek what was rewarded.\n"
        "     Spread is s.e.m. of each fly's preference toward its own CS+.\n"
        "     Valence sep is in visual-code units: full depression is 0.05."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Flies per condition; seed 0..N-1 (default: %(default)s).",
    )
    parser.add_argument(
        "--n-train", type=int, default=None,
        help="Training trials per block (default: 20 odour, 8 colour).",
    )
    parser.add_argument(
        "--n-test", type=int, default=None,
        help="Test trials per block (default: 6 odour, 1 colour).",
    )
    parser.add_argument(
        "--experiment",
        choices=list(EXPERIMENTS),
        default="compartments",
        help="Which set of conditions to run, its default output directory "
             "and its protocol (default: %(default)s).",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        choices=list(CONDITIONS),
        help="Run these conditions instead of the experiment's set.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Flies to run at once. Default: one per core minus one, capped at "
             "the number of flies. --jobs 1 runs in this process and prints "
             "per-trial progress.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory. Default: out/mb for compartments, "
             "out/mb_extinction for extinction -- separate, so neither sweep "
             "deletes the other's results.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=None,
        help="Override the depression rate. Default is ConditioningConfig.eta, "
             "not mushroom_body.DEFAULT_ETA -- the task delivers far less "
             "dopamine per trial than the module was calibrated on.",
    )
    parser.add_argument(
        "--recovery",
        type=float,
        default=None,
        help="Override the recovery rate (default ConditioningConfig.recovery). "
             "Ignored by the frozen condition, which pins both to zero.",
    )
    parser.add_argument(
        "--no-reversal",
        action="store_true",
        help="Stop after the acquisition test block.",
    )
    args = parser.parse_args()

    default_conditions, default_out, protocol = EXPERIMENTS[args.experiment]
    args.conditions = args.conditions or list(default_conditions)
    out = args.out = args.out or default_out
    out.mkdir(parents=True, exist_ok=True)
    args.n_train = protocol["n_train"] if args.n_train is None else args.n_train
    args.n_test = protocol["n_test"] if args.n_test is None else args.n_test
    reversal = protocol["reversal"] and not args.no_reversal
    naive = protocol["naive"]

    specs = []
    estimate = 0.0
    pooled_estimate = 0.0
    for condition in args.conditions:
        overrides, label = CONDITIONS[condition]
        config = dict(overrides)
        if args.eta is not None:
            config["eta"] = args.eta
        if args.recovery is not None:
            config["recovery"] = args.recovery
        colour = config.get("modality") == "colour"
        built = colour_config(**config) if colour else ConditioningConfig(**config)
        n_seeds = min(args.seeds, IDENTITY_SEEDS) if condition in IDENTITY_OF else args.seeds
        for seed in range(n_seeds):
            # Colour flies alternate the contingency by seed: Vogt's two
            # reciprocal groups. Odour flies all punish A, as before.
            cs_plus = built.stimuli[seed % 2] if colour else "A"
            specs.append(
                {
                    "condition": condition,
                    "label": label,
                    "seed": seed,
                    "n_train": args.n_train,
                    "n_test": args.n_test,
                    "cs_plus": cs_plus,
                    "reversal": reversal,
                    "naive": naive,
                    "config": dataclasses.asdict(built),
                    "out": str(out),
                }
            )
            tests = args.n_test * (1 + int(naive) + int(reversal))
            trains = args.n_train * (1 + int(reversal))
            test_seconds = built.test_seconds or built.trial_seconds
            simulated = trains * built.trial_seconds + tests * test_seconds
            estimate += simulated * WALL_PER_SIM_SECOND[built.modality]
            pooled_estimate += simulated * WALL_PER_SIM_SECOND_POOLED[built.modality]

    per_fly = args.n_train * (1 + int(reversal)) + args.n_test * (
        1 + int(naive) + int(reversal)
    )
    # One core left free so the console and the viewer stay responsive; the
    # default 15-fly sweep still fits in one wave on this machine's 16 cores.
    if args.jobs is None:
        args.jobs = min(len(specs), max(1, (os.cpu_count() or 2) - 1))
    args.jobs = max(1, min(args.jobs, len(specs)))

    serial = estimate
    identity = [c for c in args.conditions if c in IDENTITY_OF]
    print(f"conditioning sweep -> {out}")
    print(
        f"  {len(args.conditions)} conditions, {len(specs)} flies "
        f"({args.seeds} seeds"
        + (f"; {len(identity)} identity conditions on {min(args.seeds, IDENTITY_SEEDS)}"
           if identity else "")
        + f"), {per_fly} trials each ({args.n_test} test / {args.n_train} train per block)"
    )
    # Pooled flies each run slower than a lone one, so dividing the serial time
    # by the job count promises far too much -- it said 29 min for a colour
    # sweep measured at about 2 h.
    print(
        f"  {args.jobs} jobs, estimate {serial / 60:.0f} min serial -> "
        f"~{pooled_estimate / args.jobs / 60:.0f} min pooled"
    )
    if args.eta is not None or args.recovery is not None:
        print(f"  plasticity overrides: eta={args.eta}, recovery={args.recovery}")

    # Results from an earlier sweep with a different --seeds or --n-train would
    # be read back by 15_analyze_mb.py as if they belonged to this one. Same
    # reasoning as the stale-checkpoint sweep in 11_train_forage.py.
    stale = [
        p
        for condition in args.conditions
        for p in out.glob(f"{condition}_seed*.json")
    ]
    if stale:
        print(f"  deleting {len(stale)} result files from an earlier sweep")
        for path in stale:
            path.unlink()

    write_meta(out, args, specs, status="running", trials_per_fly=per_fly,
               experiment=args.experiment)

    # Each worker builds its own MuJoCo model and its own numpy state; without
    # this a BLAS thread pool per worker would oversubscribe the 16 cores.
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(variable, "1")

    # Flushed: piped to a file, stdout is block-buffered, and the next thing
    # that would flush it is a job finishing several minutes from now.
    print(flush=True)
    started = time.perf_counter()
    summaries = []

    def report(summary: dict) -> None:
        summaries.append(summary)
        pi = summary["pi"]
        if summary["modality"] == "colour":
            scores = (f"CS+ {summary['cs_plus']:<11} PI toward "
                      f"{summary['stimuli'][1]} {pi['trained']:+.2f}")
        else:
            scores = (f"PI naive {pi['naive']:+.2f} trained {pi['trained']:+.2f} "
                      f"reversed {pi['reversed']:+.2f}")
        print(
            f"[{len(summaries):>2}/{len(specs)}] {summary['label']:<24} "
            f"seed {summary['seed']:<2} {scores}  "
            f"sep {summary['separation']:+.4f}  "
            f"({summary['wall_seconds']:.0f}s)",
            flush=True,
        )

    if args.jobs == 1:
        for spec in specs:
            report(run_job(spec, on_trial=progress_printer(spec, per_fly)))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(run_job, spec): spec for spec in specs}
            for future in as_completed(futures):
                report(future.result())

    elapsed = time.perf_counter() - started
    write_meta(
        out, args, specs, status="done", wall_seconds=round(elapsed, 1),
        trials_per_fly=per_fly,
    )

    # --- summary ---------------------------------------------------------
    if summaries and summaries[0]["modality"] == "colour":
        print_colour_summary(args, summaries)
        print(f"\n{len(specs)} flies in {elapsed / 60:.1f} min -> {out}")
        print("\nnext:")
        print(f"  python scripts/15_analyze_mb.py --dir {out}")
        return
    print(f"\n{'condition':<16} | {'naive':>14} | {'trained':>14} | "
          f"{'reversed':>14} | {'valence sep':>15} | {'flies':>5}")
    print("-" * 93)
    for condition in args.conditions:
        rows = [s for s in summaries if s["condition"] == condition]
        if not rows:
            continue
        cells = []
        for block in TEST_BLOCKS:
            values = np.array([r["pi"][block] for r in rows], dtype=float)
            cells.append(f"{values.mean():+.2f} +/-{sem(values):.2f}")
        sep = np.array([r["separation"] for r in rows], dtype=float)
        print(
            f"{rows[0]['label']:<16} | {cells[0]:>14} | {cells[1]:>14} | "
            f"{cells[2]:>14} | "
            f"{f'{sep.mean():+.3f} +/-{sem(sep):.3f}':>15} | "
            f"{len(rows):>5}"
        )
    print(
        "\nPI = preference index: +1 the fly ended at odour B, -1 at odour A.\n"
        "     CS+ is odour A, so learning reads as trained > 0, reversed < 0.\n"
        f"     Spread is s.e.m. across flies (n={args.seeds}); the per-trial\n"
        "     spread is far wider. 15_analyze_mb.py reports both."
    )
    print(f"\n{len(specs)} flies in {elapsed / 60:.1f} min -> {out}")
    print("\nnext:")
    print("  python scripts/15_analyze_mb.py")


if __name__ == "__main__":
    main()

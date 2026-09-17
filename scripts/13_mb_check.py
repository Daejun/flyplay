"""Check the olfactory front end and the mushroom body, before any simulation.

Both are pure numpy, so this runs in seconds and needs no physics. If any of it
fails, the conditioning experiment cannot work -- a Kenyon-cell code that does
not separate two odours gives the mushroom body nothing to associate, and a
plasticity rule that leaks across compartments cannot represent two opposite
contingencies no matter how many trials you run.

Section 6 checks extinction: the MBON -> DAN feedback that turns a predicted
punishment which never arrives into dopamine for the avoid compartment.

Sections 7 and 8 check vision: the visual Kenyon-cell code on synthetic
compound-eye readouts built from measured floor readings, then those cells
sharing the compartments with smell.

    python scripts/13_mb_check.py
    python scripts/13_mb_check.py --no-plot

Writes out/mb/front_end.png and prints a pass/fail summary.
"""

import argparse

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flyplay.mushroom_body import Compartment, MushroomBody
from flyplay.olfactory import OlfactoryFrontEnd, pattern_correlation
from flyplay.visual_pathway import FLOOR_READINGS, VisualFrontEnd, floor_readouts

OUT = _bootstrap.OUT / "mb"

#: Concentrations the fly actually walks through: inverse-square falloff from a
#: peak-1.0 source, 25 mm away down to 3 mm away.
FAR, NEAR = 0.0016, 0.11
#: Mid working range, used wherever one concentration is enough.
MID = 0.05
A = np.array([1.0, 0.0])
B = np.array([0.0, 1.0])

DT = 0.01  # 100 Hz, the policy rate the conditioning runner uses
TRIAL_S = 8.0
US_FRACTION = 0.5  # dopamine covers the second half of a CS+ trial
N_SEEDS = 8


def run_trial(mb, odour, dan_name=None, dan=1.0):
    """One synthetic trial: odour throughout, dopamine over the second half."""
    steps = int(TRIAL_S / DT)
    us_from = int(steps * (1 - US_FRACTION))
    for i in range(steps):
        signal = {dan_name: dan} if (dan_name and i >= us_from) else {}
        mb.step(odour, signal, DT)


def fresh_mb(seed=0, **compartment_kwargs):
    front_end = OlfactoryFrontEnd(2, seed=seed)
    return MushroomBody(
        front_end,
        [
            Compartment("approach", sign=+1.0, n_kc=front_end.n_kc, **compartment_kwargs),
            Compartment("avoid", sign=-1.0, n_kc=front_end.n_kc, **compartment_kwargs),
        ],
        shared_dan=compartment_kwargs.pop("shared_dan", False),
    )


def seeing_mb(seed=0, shared_dan=False):
    """A mushroom body with both olfactory and visual Kenyon cells."""
    front_end = OlfactoryFrontEnd(2, seed=seed)
    return MushroomBody(front_end, shared_dan=shared_dan, visual=VisualFrontEnd(seed=seed))


def run_visual_trial(mb, readouts, dan_name=None, dan=1.0, odour=None):
    """`run_trial` for a floor colour: seen throughout, dopamine over the second half."""
    steps = int(TRIAL_S / DT)
    us_from = int(steps * (1 - US_FRACTION))
    smell = np.zeros(2) if odour is None else odour
    for i in range(steps):
        signal = {dan_name: dan} if (dan_name and i >= us_from) else {}
        mb.step(smell, signal, DT, readouts=readouts)


#: Synthetic floors. The measured blue and green readings are mirror images, so
#: the model's brightness for them is exactly equal; the tenth-intensity copy
#: scales every reading exactly, which the rendered one only nearly does.
ISO_BLUE, ISO_GREEN = FLOOR_READINGS["blue"], FLOOR_READINGS["green"]
GREEN_FULL = FLOOR_READINGS["green"]
GREEN_TENTH = tuple(0.1 * v for v in GREEN_FULL)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    passed = []

    # --- 1. Kenyon-cell code --------------------------------------------
    print(f"1. Kenyon-cell code ({N_SEEDS} wiring seeds, "
          f"concentration {FAR} to {NEAR}, {NEAR / FAR:.0f}x)")
    invariance, separation, overlap, sparsity = [], [], [], []
    for seed in range(N_SEEDS):
        front_end = OlfactoryFrontEnd(2, seed=seed)
        far, near, other = (
            front_end(A * FAR), front_end(A * NEAR), front_end(B * NEAR)
        )
        invariance.append(pattern_correlation(far.kc, near.kc))
        separation.append(pattern_correlation(near.kc, other.kc))
        shared = np.intersect1d(near.active, other.active).size
        overlap.append(shared / max(near.active.size, 1))
        sparsity.append(near.active.size)
    print(f"   same odour across concentration : {np.mean(invariance):.3f} "
          f"(worst {np.min(invariance):.3f})")
    print(f"   odour A against odour B         : {np.mean(separation):.3f} "
          f"(worst {np.max(separation):.3f})")
    print(f"   Kenyon cells shared by A and B  : {np.mean(overlap):.1%}")
    print(f"   active cells                    : {int(np.mean(sparsity))} "
          f"of {front_end.n_kc} ({np.mean(sparsity) / front_end.n_kc:.1%})")
    passed.append(("KC code survives a concentration change", min(invariance) > 0.9))
    passed.append(("KC code separates two odours", max(separation) < 0.2))
    passed.append(
        ("KC code is sparse", all(n == front_end.k for n in sparsity))
    )

    # A silent fly must produce no code at all, or dopamine arriving while it
    # smells nothing would land on a pattern conjured out of numerical dust.
    silent = OlfactoryFrontEnd(2, seed=0)(np.zeros(2))
    print(f"   no odour -> active cells        : {silent.active.size} (want 0)")
    passed.append(("no odour produces no code", silent.active.size == 0))

    # --- 2. acquisition --------------------------------------------------
    print("\n2. acquisition: odour A paired with punishment")
    mb = fresh_mb()
    curve = []
    for trial in range(1, 21):
        run_trial(mb, A * MID, "approach")
        curve.append((mb.read(A * MID), mb.read(B * MID)))
        if trial in (1, 3, 5, 10, 20):
            ra, rb = curve[-1]
            print(f"   trial {trial:>2} -> MBON_approach  A {ra.mbon['approach']:.3f}  "
                  f"B {rb.mbon['approach']:.3f}   valence A {ra.valence:+.3f}")
    final_a, final_b = curve[-1]
    drop_a = 1.0 - final_a.mbon["approach"]
    drop_b = 1.0 - final_b.mbon["approach"]
    leak = 1.0 - final_a.mbon["avoid"]
    print(f"   depression after 20 trials      : A {drop_a:.1%}, B {drop_b:.1%}")
    print(f"   other compartment              : {leak:.1%} (want 0)")
    passed.append(("the paired odour is depressed", drop_a > 0.5))
    passed.append(("the unpaired odour is not", drop_b < 0.1))
    passed.append(("the untargeted compartment is not", abs(leak) < 1e-9))

    # --- 3. reversal ------------------------------------------------------
    print("\n3. reversal: punish B instead, starting from that memory")
    reversal = []
    for trial in range(1, 21):
        run_trial(mb, B * MID, "approach")
        reversal.append((mb.valence_of(A * MID), mb.valence_of(B * MID)))
        if trial in (1, 5, 10, 20):
            va, vb = reversal[-1]
            print(f"   trial {trial:>2} -> valence  A {va:+.3f}  B {vb:+.3f}")
    crossover = next(
        (i + 1 for i, (va, vb) in enumerate(reversal) if vb < va), None
    )
    print(f"   valence crosses over at trial   : {crossover}")
    passed.append(("reversal flips the preference", crossover is not None))

    # --- 4. forgetting ----------------------------------------------------
    print("\n4. recovery: does the memory decay without dopamine?")
    mb2 = fresh_mb()
    for _ in range(20):
        run_trial(mb2, A * MID, "approach")
        before = mb2.read(A * MID).mbon["approach"]
    rest = []
    for seconds in (30, 60, 120):
        mb3 = fresh_mb()
        for _ in range(20):
            run_trial(mb3, A * MID, "approach")
        for _ in range(int(seconds / DT)):
            mb3.step(np.zeros(2), {}, DT)
        rest.append(mb3.read(A * MID).mbon["approach"])
        print(f"   {seconds:>3} s of no odour -> {rest[-1]:.3f}")
    print(f"   (from {before:.3f} right after training)")
    passed.append(("a depression-only rule does not ratchet", rest[-1] > 0.95))

    # Dopamine with nothing to associate it with must be inert.
    mb4 = fresh_mb()
    for _ in range(int(20 * TRIAL_S / DT)):
        mb4.step(np.zeros(2), {"approach": 1.0}, DT)
    unpaired = mb4.read(A * MID).mbon["approach"]
    print(f"   dopamine with no odour present  : {unpaired:.3f} (want 1.000)")
    passed.append(("unpaired dopamine changes nothing", abs(unpaired - 1.0) < 1e-9))

    # --- 5. compartments --------------------------------------------------
    print("\n5. compartments: two contingencies at once "
          "(punish A, reward B)")
    spread = {}
    for shared in (False, True):
        mb5 = fresh_mb()
        mb5.shared_dan = shared
        for _ in range(20):
            run_trial(mb5, A * MID, "approach")
            run_trial(mb5, B * MID, "avoid")
        va, vb = mb5.valence_of(A * MID), mb5.valence_of(B * MID)
        spread["shared" if shared else "separate"] = vb - va
        label = "one shared dopamine" if shared else "per-compartment"
        print(f"   {label:<20} valence A {va:+.3f}  B {vb:+.3f}  "
              f"spread {vb - va:+.3f}")
    print("   a scalar signal drives both compartments identically, so their")
    print("   weights cannot diverge -- the zero spread is an identity, not noise")
    passed.append(("compartments represent opposite valences", spread["separate"] > 0.5))
    passed.append(("one global signal cannot", abs(spread["shared"]) < 1e-9))

    # --- 6. extinction ----------------------------------------------------
    print("\n6. extinction: odour A again, without punishment "
          "(3 exposures after 20 CS+ trials)")

    def omission(target):
        # The rule flyplay.conditioning uses: dopamine in proportion to the
        # punishment the circuit predicts for what it is smelling.
        return lambda mbon, valence: {target: max(0.0, -valence)}

    def expose(mb, odour, feedback, n=3):
        for _ in range(n):
            for _ in range(int(TRIAL_S / DT)):
                mb.step(odour, {}, DT, feedback=feedback)

    outcome = {}
    for label, target in (("passive", None), ("to avoid", "avoid"),
                          ("to approach", "approach")):
        mb6 = fresh_mb()
        for _ in range(20):
            run_trial(mb6, A * MID, "approach")
        start = mb6.read(A * MID).valence
        expose(mb6, A * MID, omission(target) if target else None)
        state = mb6.read(A * MID)
        outcome[label] = state
        print(f"   {label:<12} valence {start:+.3f} -> {state.valence:+.3f}   "
              f"approach {state.mbon['approach']:.3f}  avoid {state.mbon['avoid']:.3f}")
    passive, extinct, wrong = outcome["passive"], outcome["to avoid"], outcome["to approach"]
    passed.append((
        "omission of predicted punishment cancels the memory",
        abs(extinct.valence) < 0.5 * abs(passive.valence),
    ))
    # Extinction writes only to avoid, so approach must have followed exactly
    # the same path as with no signal at all: the original memory is untouched
    # and a second one sits beside it.
    passed.append((
        "the original memory survives underneath",
        abs(extinct.mbon["approach"] - passive.mbon["approach"]) < 1e-9
        and extinct.mbon["avoid"] < 1.0,
    ))
    passed.append((
        "the same signal in the wrong compartment deepens it",
        wrong.valence < start,
    ))

    novel = {}
    for label, target in (("passive", None), ("feedback", "avoid")):
        mb7 = fresh_mb()
        expose(mb7, B * MID, omission(target) if target else None)
        novel[label] = mb7.read(B * MID).valence
    print(f"   novel odour B, exposed with and without feedback: "
          f"{novel['feedback']:+.6f} vs {novel['passive']:+.6f} (want identical)")
    passed.append((
        "no predicted punishment, no omission signal",
        abs(novel["feedback"] - novel["passive"]) < 1e-12,
    ))

    # --- 7. visual Kenyon cells --------------------------------------------
    n_visual_seeds = 16
    print(f"\n7. visual Kenyon cells ({n_visual_seeds} wiring seeds, synthetic "
          f"readouts from measured floor readings)")
    blue_r, green_r = floor_readouts(FLOOR_READINGS["blue"]), floor_readouts(FLOOR_READINGS["green"])
    split_r = floor_readouts(FLOOR_READINGS["blue"], FLOOR_READINGS["green"])
    hue, brightness_code, hue_across, identity_colour, identity_brightness = [], [], [], [], []
    bilateral, active, blind = [], [], []
    for seed in range(n_visual_seeds):
        visual = VisualFrontEnd(seed=seed)
        blue, green = visual(blue_r), visual(green_r)
        hue.append(pattern_correlation(blue.kc[0], green.kc[0]))
        brightness_code.append(pattern_correlation(
            visual(floor_readouts(GREEN_FULL)).kc[0], visual(floor_readouts(GREEN_TENTH)).kc[0]))
        hue_across.append(pattern_correlation(
            blue.kc[0], visual(floor_readouts(FLOOR_READINGS["dim_blue"])).kc[0]))
        split = visual(split_r)
        bilateral.append(min(pattern_correlation(split.kc[0], blue.kc[0]),
                             pattern_correlation(split.kc[1], green.kc[1])))
        active.extend([np.count_nonzero(blue.kc[0]), np.count_nonzero(green.kc[1])])
        # Vogt et al. (2016): colour memory needs VPN-MB1, brightness memory
        # VPN-MB2. With one family silenced, stimuli differing only along the
        # other dimension must become the same code -- exactly, not nearly.
        visual.silenced = {"colour"}
        identity_colour.append(np.abs(visual(floor_readouts(ISO_BLUE)).kc
                                      - visual(floor_readouts(ISO_GREEN)).kc).max())
        visual.silenced = {"brightness"}
        identity_brightness.append(np.abs(visual(floor_readouts(GREEN_FULL)).kc
                                          - visual(floor_readouts(GREEN_TENTH)).kc).max())
        visual.silenced = {"colour", "brightness"}
        blind.append(np.count_nonzero(visual(blue_r).kc))
    print(f"   blue floor against green floor  : {np.mean(hue):.3f} (worst {np.max(hue):.3f})")
    print(f"   green against a tenth as bright : {np.mean(brightness_code):.2f} "
          f"(seeds {np.min(brightness_code):.2f}-{np.max(brightness_code):.2f}; recorded, no criterion)")
    print(f"   blue against measured dim blue  : {np.mean(hue_across):.2f} (recorded)")
    print(f"   colour VPNs off, isoluminant    : max difference {max(identity_colour):.2e}")
    print(f"   brightness VPNs off, 10:1 green : max difference {max(identity_brightness):.2e}")
    print(f"   left eye on blue, right on green: each eye's code matches (worst cosine "
          f"{min(bilateral):.6f})")
    print(f"   active cells                    : {min(active)}-{max(active)} of 100; "
          f"everything silenced -> {max(blind)}")
    passed.append(("visual code separates blue from green", max(hue) < 0.2))
    passed.append(("colour VPNs off: blue and green become one code", max(identity_colour) < 1e-9))
    passed.append(("brightness VPNs off: brightness is invisible", max(identity_brightness) < 1e-9))
    passed.append(("each eye codes its own side", min(bilateral) > 1 - 1e-9))
    passed.append(("visual code is sparse, and silent when undriven",
                   set(active) == {VisualFrontEnd().k} and max(blind) == 0))

    # --- 8. vision in the mushroom body --------------------------------------
    print("\n8. vision shares the compartments (blue floor paired with punishment)")
    mb8 = seeing_mb()
    odour_before = mb8.valence_of(A * MID)
    olfactory_before = [c.weights[: mb8.n_olfactory].copy() for c in mb8.compartments]
    for _ in range(20):
        run_visual_trial(mb8, blue_r, "approach")
    blue_state, green_state = mb8.read(np.zeros(2), blue_r), mb8.read(np.zeros(2), green_r)
    print(f"   after 20 trials: valence blue {blue_state.valence:+.4f}  green "
          f"{green_state.valence:+.4f}  odour A {mb8.valence_of(A * MID):+.4f} "
          f"(was {odour_before:+.4f})")
    untouched = all(np.array_equal(c.weights[: mb8.n_olfactory], w)
                    for c, w in zip(mb8.compartments, olfactory_before))
    avoid_visual = mb8["avoid"].weights[mb8.n_olfactory :]
    # Scale: an active visual cell should carry about what an active olfactory
    # cell does, or one sense would learn twenty times faster than the other.
    vis_code = mb8.embed(visual_kc=mb8.visual(blue_r).mean_kc)[mb8.n_olfactory :]
    olf_code = OlfactoryFrontEnd(2, seed=0)(A * MID).kc
    per_cell = vis_code[vis_code > 0].mean() / olf_code[olf_code > 0].mean()
    print(f"   olfactory synapses untouched: {untouched}; avoid compartment's visual "
          f"synapses at rest: {bool(np.all(avoid_visual == 1.0))}")
    print(f"   activity per active cell, visual / olfactory: {per_cell:.2f}")
    left_right = mb8.eye_valences(mb8.visual(split_r))
    print(f"   blue on the left, green on the right -> eye valences "
          f"{left_right[0]:+.4f} / {left_right[1]:+.4f}")
    # The whole memory lives in the visual share of the population, which
    # caps what vision can move the valence by; recorded here for the steering
    # gain. Full depression of the blue code would give -visual_scale.
    passed.append(("a punished colour loses valence, the other colour does not",
                   blue_state.valence < -0.5 * mb8.visual_scale
                   and abs(green_state.valence) < 0.1 * abs(blue_state.valence)))
    passed.append(("a colour memory leaves smell exactly as it was",
                   untouched and mb8.valence_of(A * MID) == odour_before))
    passed.append(("the untargeted compartment's visual synapses are not touched",
                   bool(np.all(avoid_visual == 1.0))))
    passed.append(("an active visual cell weighs what an olfactory one does",
                   0.5 < per_cell < 2.0))
    passed.append(("the eye that sees the punished colour reads worse",
                   left_right[0] < left_right[1]))

    spread8 = {}
    for shared in (False, True):
        mb8b = seeing_mb(shared_dan=shared)
        for _ in range(20):
            run_visual_trial(mb8b, blue_r, "approach")
            run_visual_trial(mb8b, green_r, "avoid")
        vb = mb8b.valence_of(np.zeros(2), blue_r)
        vg = mb8b.valence_of(np.zeros(2), green_r)
        spread8[shared] = vg - vb
        print(f"   {'one shared dopamine' if shared else 'per-compartment':<20} valence "
              f"blue {vb:+.4f}  green {vg:+.4f}  spread {vg - vb:+.4f}")
    passed.append(("one global signal cannot tell colours apart either",
                   spread8[False] > 0 and abs(spread8[True]) < 1e-9))

    blocked = seeing_mb()
    blocked.visual_blocked = True
    for _ in range(20):
        run_visual_trial(blocked, blue_r, "approach")
    print(f"   visual cells blocked: blue valence {blocked.valence_of(np.zeros(2), blue_r):+.6f}, "
          f"visual synapses at rest: "
          f"{all(np.all(c.weights[blocked.n_olfactory:] == 1.0) for c in blocked.compartments)}")
    passed.append((
        "blocked visual cells learn nothing and report nothing",
        all(np.all(c.weights[blocked.n_olfactory:] == 1.0) for c in blocked.compartments)
        and blocked.valence_of(np.zeros(2), blue_r) == 0.0,
    ))

    # Smell learned by a seeing fly that sees nothing must match a smell-only
    # fly synapse for synapse: adding eyes may not change the nose.
    smell_only, with_eyes = fresh_mb(), seeing_mb()
    for _ in range(5):
        run_trial(smell_only, A * MID, "approach")
        run_trial(with_eyes, A * MID, "approach")
    same = all(np.array_equal(a.weights, b.weights[: with_eyes.n_olfactory])
               for a, b in zip(smell_only.compartments, with_eyes.compartments))
    print(f"   odour learning with eyes closed matches a smell-only fly exactly: {same}")
    passed.append(("vision changes nothing about smell when nothing is seen", same))

    # --- summary ----------------------------------------------------------
    print("\n" + "=" * 58)
    for label, ok in passed:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print("=" * 58)
    if not all(ok for _, ok in passed):
        print("Fix these before running the experiment: the conditioning task "
              "cannot compensate for a circuit that does not discriminate.")

    if args.no_plot:
        return

    fig, axes = plt.subplots(1, 4, figsize=(18, 4), tight_layout=True)

    ax = axes[0]
    front_end = OlfactoryFrontEnd(2, seed=0)
    scales = np.geomspace(FAR, NEAR, 12)
    ref = front_end(A * NEAR)
    ax.plot(scales, [pattern_correlation(ref.kc, front_end(A * s).kc) for s in scales],
            "o-", label="A vs A")
    ax.plot(scales, [pattern_correlation(ref.kc, front_end(B * s).kc) for s in scales],
            "s--", color="tab:red", label="A vs B")
    ax.axhline(0.9, color="0.6", lw=0.8, ls=":")
    ax.set(xscale="log", xlabel="concentration (source intensity)",
           ylabel="KC pattern correlation", ylim=(-0.05, 1.05),
           title="1. Identity survives concentration")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(range(1, 21), [c[0].mbon["approach"] for c in curve], "o-",
            label="odour A (paired)")
    ax.plot(range(1, 21), [c[1].mbon["approach"] for c in curve], "s--",
            color="0.5", label="odour B (unpaired)")
    ax.set(xlabel="CS+ trial", ylabel="MBON_approach", ylim=(0, 1.05),
           title="2. Dopamine depresses the paired odour")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(range(1, 21), [v[0] for v in reversal], "o-", label="odour A")
    ax.plot(range(1, 21), [v[1] for v in reversal], "s-", color="tab:red",
            label="odour B (now punished)")
    ax.axhline(0, color="0.6", lw=0.8)
    if crossover:
        ax.axvline(crossover, color="0.4", lw=0.8, ls=":")
    ax.set(xlabel="reversal trial", ylabel="valence",
           title="3. Reversal flips it back")
    ax.legend(fontsize=8)

    ax = axes[3]
    visual = VisualFrontEnd(seed=0)
    floors = ["blue", "dim_blue", "green", "dim_green", "grey", "white"]
    codes = [visual(floor_readouts(FLOOR_READINGS[f])).kc[0] for f in floors]
    matrix = np.array([[pattern_correlation(a, b) for b in codes] for a in codes])
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(floors)), floors, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(floors)), floors, fontsize=8)
    for i in range(len(floors)):
        for j in range(len(floors)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if matrix[i, j] < 0.6 else "black")
    ax.set_title("7. Visual KC code, floor against floor (seed 0)")
    fig.colorbar(image, ax=ax, fraction=0.046)

    out_png = OUT / "front_end.png"
    fig.savefig(out_png, dpi=140)
    print(f"\nplot -> {out_png}")


if __name__ == "__main__":
    main()

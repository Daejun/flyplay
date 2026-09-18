"""Check that smell and sight are wired up correctly, before training anything.

If either sense has its left and right swapped, or reads a constant, RL will
fail in a way that looks like a reward-shaping problem and will cost hours to
diagnose. This script answers three questions directly:

1. Does odour intensity rise as the fly approaches a source, and is the
   left-right asymmetry positive when the source is on the fly's left?
2. Does a pillar to the left darken the left eye more than the right?
3. Are the two senses independent -- does the odour source (marker in geom
   group 2) stay invisible to the fly, and the pillar odourless?
4. Can the eyes read the colour of the floor, on the correct side? The colour
   conditioning task (`flyplay.conditioning`, Vogt et al. 2014) stands on it.

    python scripts/10_sense_check.py
    python scripts/10_sense_check.py --no-plot

Writes out/sense_check/sense_check.png and prints a pass/fail summary.
"""

import argparse

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flyplay.build import build
from flyplay.odor import SENSOR_NAMES, OdorField, OdorSource
from flyplay.control import Walker
from flyplay.vision import FEATURE_NAMES, RetinaFeatures
from flyplay.visual_pathway import eye_layout, photoreceptors, vpn_field

OUT = _bootstrap.OUT / "sense_check"


def sweep_odor(distances, lateral):
    """Odour at a static fly while the source is moved around it."""
    rows = []
    for dist in distances:
        field = OdorField(
            sources=[OdorSource(pos=(dist, lateral, 1.5), peak=(1.0,))]
        )
        fs = build("flat", odor_field=field, colorize=False)
        fs.sim.reset()
        fs.sim.warmup(0.2)
        intensity = fs.odor()  # (4, 1)
        rows.append(
            {
                "distance": dist,
                "sensors": intensity[:, 0].copy(),
                "left_right": OdorField.left_right(intensity)[:, 0].copy(),
                "asymmetry": float(OdorField.asymmetry(intensity)[0]),
            }
        )
        fs.close()
    return rows


def sweep_pillar(bearings_deg, distance=6.0):
    """Visual features while a pillar is moved around the fly."""
    retina = RetinaFeatures()

    # Calibrate the horizon on an empty scene, then reuse that calibration.
    empty = build("flat", vision=True, colorize=False)
    empty.sim.reset()
    empty.sim.warmup(0.2)
    baseline = empty.ommatidia_readouts()
    retina.calibrate_horizon(baseline)
    base_features = retina.extract(baseline)
    empty.close()

    rows = []
    for bearing in bearings_deg:
        angle = np.deg2rad(bearing)
        pos = (distance * np.cos(angle), distance * np.sin(angle))
        fs = build("flat", vision=True, pillars=(pos,), colorize=False)
        fs.sim.reset()
        fs.sim.warmup(0.2)
        readouts = fs.ommatidia_readouts()
        rows.append(
            {
                "bearing": bearing,
                "features": retina.extract(readouts),
                "n_pairs": int(fs.sim.mj_model.npair),
            }
        )
        fs.close()
    return retina, base_features, rows


def crosstalk_check():
    """The marker must be invisible; the pillar must be odourless."""
    field = OdorField(sources=[OdorSource(pos=(8.0, 0.0, 1.5), peak=(1.0,))])
    retina = RetinaFeatures()

    empty = build("flat", vision=True, colorize=False)
    empty.sim.reset()
    empty.sim.warmup(0.2)
    retina.calibrate_horizon(empty.ommatidia_readouts())
    base = retina.extract(empty.ommatidia_readouts())
    empty.close()

    # Odour source in front. Its marker sits in geom group 2.
    with_odor = build("flat", vision=True, odor_field=field, colorize=False)
    with_odor.sim.reset()
    with_odor.sim.warmup(0.2)
    odor_view = retina.extract(with_odor.ommatidia_readouts())
    with_odor.close()

    # Pillar in front, no odour source -- sensors should read zero.
    empty_field = OdorField(sources=[])
    with_pillar = build(
        "flat", vision=True, odor_field=empty_field, pillars=((8.0, 0.0),), colorize=False
    )
    with_pillar.sim.reset()
    with_pillar.sim.warmup(0.2)
    pillar_odor = with_pillar.odor()
    pillar_view = retina.extract(
        with_pillar.ommatidia_readouts()
    )
    with_pillar.close()

    return base, odor_view, pillar_view, pillar_odor


def floor_colour_check():
    """Type means over the VPN band for painted floors, both eyes.

    The fly spawns on a tile corner facing +x along an edge, so a checker
    puts one colour on its left and the other on its right.
    """
    pale = eye_layout()[0]
    band = vpn_field()
    fs = build("flat", vision=True, colorize=False, floor_tiles=(20.0, 12))
    fs.sim.reset()
    fs.sim.warmup(0.2)

    def means():
        own = photoreceptors(fs.ommatidia_readouts())
        # (eye, yellow-type mean, pale-type mean)
        return [(own[e][band & ~pale].mean(), own[e][band & pale].mean()) for e in (0, 1)]

    readings = {}
    for label, paint in (
        ("uniform blue", lambda: fs.floor.paint_uniform("blue")),
        ("uniform green", lambda: fs.floor.paint_uniform("green")),
        ("checker blue|green", lambda: fs.floor.paint_checker("blue", "green")),
        ("checker green|blue", lambda: fs.floor.paint_checker("green", "blue")),
    ):
        paint()
        readings[label] = means()
    fs.floor.paint_checker("blue", "green")
    left_after = fs.floor.colour_at((0.5, 5.0))
    fs.close()
    return readings, left_after


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    passed = []

    # --- 1. odour vs distance, source straight ahead ---------------------
    print("1. odour intensity vs distance (source straight ahead)")
    distances = np.array([3.0, 5.0, 8.0, 12.0, 18.0, 25.0])
    ahead = sweep_odor(distances, lateral=0.0)
    for r in ahead:
        print(
            f"   {r['distance']:5.1f} mm -> "
            + "  ".join(
                f"{n.split('_')[0][0]}{n.split('_')[-1][:4]}={v:7.4f}"
                for n, v in zip(SENSOR_NAMES, r["sensors"])
            )
        )
    means = [r["sensors"].mean() for r in ahead]
    monotonic = all(a > b for a, b in zip(means, means[1:]))
    passed.append(("odour falls off with distance", monotonic))

    # --- 2. odour asymmetry vs lateral offset ---------------------------
    print("\n2. odour asymmetry vs source side (source 8 mm ahead)")
    laterals = np.array([-6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0])
    asym = []
    for lateral in laterals:
        field = OdorField(sources=[OdorSource(pos=(8.0, lateral, 1.5), peak=(1.0,))])
        fs = build("flat", odor_field=field, colorize=False)
        fs.sim.reset()
        fs.sim.warmup(0.2)
        a = float(OdorField.asymmetry(fs.odor())[0])
        asym.append(a)
        side = "left" if lateral > 0 else "right" if lateral < 0 else "centre"
        print(f"   y={lateral:+5.1f} mm ({side:>6}) -> asymmetry {a:+.4f}")
        fs.close()
    asym = np.array(asym)
    correct_sign = bool(
        np.all(asym[laterals > 0] > 0.01) and np.all(asym[laterals < 0] < -0.01)
    )
    passed.append(("odour asymmetry points at the source", correct_sign))

    # --- 3. vision vs pillar bearing ------------------------------------
    print("\n3. visual features vs pillar bearing (pillar 6 mm away)")
    bearings = np.array([-60, -40, -20, 0, 20, 40, 60])
    retina, base_features, vis_rows = sweep_pillar(bearings)
    print(f"   horizon calibrated to row {retina.horizon_row:.0f} "
          f"({retina.n_upper}/{retina.n_ommatidia} ommatidia above it)")
    print(f"   empty scene: {retina.describe(base_features)}")
    for r in vis_rows:
        f = r["features"]
        print(
            f"   {r['bearing']:+4.0f} deg -> L={f[0]:.3f} R={f[1]:.3f} "
            f"asym={f[4]:+.3f}  pairs={r['n_pairs']}"
        )
    vis_asym = np.array([r["features"][4] for r in vis_rows])
    vis_sign = bool(
        np.all(vis_asym[bearings > 10] > 0) and np.all(vis_asym[bearings < -10] < 0)
    )
    passed.append(("visual asymmetry points at the pillar", vis_sign))
    passed.append(
        ("pillars generate contact pairs", all(r["n_pairs"] > 0 for r in vis_rows))
    )

    # --- 4. crosstalk ---------------------------------------------------
    print("\n4. the senses do not leak into each other")
    base, odor_view, pillar_view, pillar_odor = crosstalk_check()
    marker_invisible = bool(abs(odor_view[5] - base[5]) < 0.01)
    pillar_seen = bool(pillar_view[5] - base[5] > 0.01)
    pillar_odourless = bool(np.allclose(pillar_odor, 0.0))
    print(f"   odour marker in view -> total_mass change {odor_view[5] - base[5]:+.4f}"
          f"   (want ~0: the fly must not see it)")
    print(f"   pillar in view       -> total_mass change {pillar_view[5] - base[5]:+.4f}"
          f"   (want >0)")
    print(f"   pillar odour reading -> {pillar_odor.ravel()}   (want all zero)")
    passed.append(("odour marker invisible to the fly", marker_invisible))
    passed.append(("pillar visible to the fly", pillar_seen))
    passed.append(("pillar has no smell", pillar_odourless))

    # --- 5. floor colour --------------------------------------------------
    print("\n5. floor colour, over the VPN band (yellow-type / pale-type per eye)")
    readings, left_is = floor_colour_check()
    for label, eyes in readings.items():
        print(f"   {label:<19} -> left {eyes[0][0]:.3f}/{eyes[0][1]:.3f}   "
              f"right {eyes[1][0]:.3f}/{eyes[1][1]:.3f}")
    blue, green = readings["uniform blue"], readings["uniform green"]
    # Pale-type ommatidia read blue and yellow-type read green: a blue floor
    # must light the pale type in both eyes and a green one the yellow type.
    colour_read = all(b[1] > 3 * b[0] for b in blue) and all(g[0] > 3 * g[1] for g in green)
    # Repainting at runtime must reach the eyes, or the protocol's training and
    # test floors would look the same.
    recoloured = abs(blue[0][1] - green[0][1]) > 0.2
    # checker blue|green paints even squares blue; the square on the fly's left
    # at spawn is even ("colour_at" says so), so the left eye reads blue.
    lateral = readings["checker blue|green"]
    swapped = readings["checker green|blue"]
    sided = (
        left_is == "blue"
        and lateral[0][1] > lateral[0][0] and lateral[1][0] > lateral[1][1]
        and swapped[0][0] > swapped[0][1] and swapped[1][1] > swapped[1][0]
    )
    print(f"   square on the fly's left at spawn is {left_is} for checker blue|green")
    passed.append(("eyes read floor colour by ommatidium type", bool(colour_read)))
    passed.append(("repainting the floor reaches the eyes", bool(recoloured)))
    passed.append(("each eye reads the colour on its own side", bool(sided)))

    # --- summary ---------------------------------------------------------
    print("\n" + "=" * 58)
    for label, ok in passed:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print("=" * 58)
    if not all(ok for _, ok in passed):
        print("Fix the failures before training: RL cannot compensate for a "
              "sense that is wired backwards.")

    if args.no_plot:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), tight_layout=True)
    axes[0].plot(distances, [r["sensors"].mean() for r in ahead], "o-")
    axes[0].set(
        xlabel="distance to source (mm)",
        ylabel="mean sensor intensity",
        title="1. Odour falls off with distance",
        yscale="log",
    )
    axes[1].plot(laterals, asym, "o-", color="tab:orange")
    axes[1].axhline(0, color="0.6", lw=0.8)
    axes[1].axvline(0, color="0.6", lw=0.8)
    axes[1].set(
        xlabel="source lateral offset (mm)   [+ = fly's left]",
        ylabel=r"$(I_L - I_R)\,/\,\bar{I}$",
        title="2. Odour asymmetry",
    )
    axes[2].plot(bearings, vis_asym, "o-", color="tab:green", label="mass asymmetry")
    axes[2].plot(
        bearings,
        [r["features"][5] - base_features[5] for r in vis_rows],
        "s--",
        color="0.5",
        label="total mass above empty",
    )
    axes[2].axhline(0, color="0.6", lw=0.8)
    axes[2].axvline(0, color="0.6", lw=0.8)
    axes[2].set(
        xlabel="pillar bearing (deg)   [+ = fly's left]",
        ylabel="feature value",
        title="3. Vision localises the pillar",
    )
    axes[2].legend(fontsize=8)

    out_png = OUT / "sense_check.png"
    fig.savefig(out_png, dpi=140)
    print(f"\nplot -> {out_png}")


if __name__ == "__main__":
    main()

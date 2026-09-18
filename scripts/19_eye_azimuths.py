"""Measure which way each ommatidium looks, by rendering, for the central complex.

    python scripts/19_eye_azimuths.py

FlyGym's eyes are fisheye cameras mapped onto a hexagonal lattice; nothing in
the package says which azimuth an ommatidium faces. This puts the fly in the
middle of the sandbox room facing +x, darkens one 10-degree stripe of the drum
(`flyplay.room`, radius 70 mm, 0-40 mm high) at a time, and records which
ommatidia darken. An ommatidium's azimuth is the circular mean of the stripe
angles weighted by how much each darkened it; ommatidia the drum never reaches
(the floor below, the sky above it) get NaN.

Writes ``flyplay/data/ommatidia_azimuth.npz``: ``azimuth_deg`` (2, 721), left
eye first, counterclockwise from straight ahead; ``weight`` (2, 721), the
summed darkening, for choosing reliable ommatidia; and the stripe readings.
"""

import os

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_BLOCKTIME", "0")
for _threads in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_threads, "1")

import _bootstrap  # noqa: F401
import mujoco
import numpy as np

from flyplay.room import DRUM_RGBA, DRUM_STRIPES
from flyplay.sandbox import Sandbox, SandboxConfig, load_preset
from flyplay.visual_pathway import photoreceptors

OUT = _bootstrap.ROOT / "flyplay" / "data" / "ommatidia_azimuth.npz"


def main() -> None:
    sb = Sandbox(SandboxConfig(), seed=0)
    load_preset(sb, "empty")
    sb.reset_fly(yaw=0.0)
    drum = sb.place("drum", 0.0, 0.0, speed=0.0)
    room, model, data = sb.room, sb.fs.sim.mj_model, sb.fs.sim.mj_data
    room.animate({"drum": 0.0})

    def render(dark: int | None) -> np.ndarray:
        for i, gid in enumerate(room.drum_geoms):
            model.geom_rgba[gid] = DRUM_RGBA if i == dark else (0.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(model, data)
        return photoreceptors(sb.fs.ommatidia_readouts())

    base = render(None)
    angles = np.deg2rad((np.arange(DRUM_STRIPES) + 0.5) * 360.0 / DRUM_STRIPES)
    drops = np.stack([np.maximum(base - render(k), 0.0) for k in range(DRUM_STRIPES)])  # (36, 2, 721)
    weight = drops.sum(axis=0)
    vector = (drops * np.exp(1j * angles)[:, None, None]).sum(axis=0)
    azimuth = np.degrees(np.angle(vector))
    reliable = weight > 0.05 * weight.max()
    azimuth = np.where(reliable, azimuth, np.nan)
    np.savez_compressed(OUT, azimuth_deg=azimuth.astype(np.float32), weight=weight.astype(np.float32),
                        stripe_drop=drops.astype(np.float32))
    for eye, name in enumerate(("left", "right")):
        a = azimuth[eye][reliable[eye]]
        print(f"{name} eye: {reliable[eye].sum()} ommatidia see the drum, azimuth "
              f"{np.nanmin(a):+.0f} to {np.nanmax(a):+.0f} deg (median {np.nanmedian(a):+.0f})")
    print(f"wrote {OUT.relative_to(_bootstrap.ROOT)}")
    sb.close()


if __name__ == "__main__":
    main()

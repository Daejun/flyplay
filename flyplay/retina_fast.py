"""The compound eyes' readout in one pass instead of three.

`Simulation.get_ommatidia_readouts` renders each eye, straightens the image
(`Retina.correct_fisheye`) and then pools it into ommatidia
(`Retina.raw_image_to_hex_pxls`). The straightening recomputes the same
512x450 coordinate transform on every call and writes a whole image that the
pooling immediately reads back. Both of those are fixed by the retina's
geometry, so the source pixel of every destination pixel can be worked out
once and the readout becomes a single gather.

Measured on one eye (`eyes_bench.py`, 22:50 code, unchanged since):

    idle machine, 16 threads   fisheye 0.42 ms  pooling 0.35  gather 0.31
    one numba thread           fisheye 1.82 ms  pooling 0.35  gather 0.32
    pinned to one P core       fisheye 1.90 ms  pooling 0.34  gather 0.31

so both eyes with the render take 7.07 ms against 1.96 on a pinned worker,
which is what an experiment worker is. In a 10 Hz room that is 98.0 ms of
each simulated second down to 44.6, nearly all of what is left being the
render itself; in a room where something moves the eyes run at 100 Hz and it
is worth about ten times as much.

The gather is bit-identical to the two stock passes, not merely close:

  * `_correct_fisheye` copies pixels, it does not blend them, so each
    destination pixel holds exactly one source pixel;
  * destination pixels whose source falls outside the image stay 0 in the
    stock path and are summed as `0 / hex_pxl_size`, which adds nothing, so
    leaving them out gives the same sum;
  * the remaining pixels are visited in ascending flat order, as
    `_raw_image_to_hex_pxls` visits them (its `prange` is `parallel=False`),
    so the additions happen in the same order.

`EyeReadout` checks that against FlyGym's own path when it is built, on a
random image, and raises rather than quietly drifting after an upgrade.
"""

from __future__ import annotations

import numba as nb
import numpy as np


@nb.njit(cache=True)
def _gather(raw_flat, src, hexel, channel, pixels, n_ommatidia):
    """One eye's readout from its raw render. See the module docstring."""
    vals = np.zeros((n_ommatidia, 2))
    for j in range(src.size):
        h = hexel[j]
        c = channel[j]
        vals[h, c] += raw_flat[src[j], c + 1] / pixels[h]
    return vals / 255


def fisheye_source_index(retina) -> np.ndarray:
    """For each destination pixel, the flat index of the source pixel
    `Retina._correct_fisheye` copies into it, or -1 when it has none.

    The arithmetic below is that function's, term for term.
    """
    nrows, ncols = retina.nrows, retina.ncols
    zoom, k = retina.zoom, retina.distortion_coefficient
    src = np.full(nrows * ncols, -1, dtype=np.int64)
    for dst_row in range(nrows):
        for dst_col in range(ncols):
            dst_row_norm = ((2 * dst_row - nrows) / nrows) / zoom
            dst_col_norm = ((2 * dst_col - ncols) / ncols) / zoom
            denom = 1 - (k * (dst_col_norm**2 + dst_row_norm**2)) + 1e-6
            src_row = int((((dst_row_norm / denom) + 1) * nrows) / 2)
            src_col = int((((dst_col_norm / denom) + 1) * ncols) / 2)
            if 0 <= src_row < nrows and 0 <= src_col < ncols:
                src[dst_row * ncols + dst_col] = src_row * ncols + src_col
    return src


class EyeReadout:
    """`Simulation.get_ommatidia_readouts` for one fly, as a gather.

    Build it once per simulation and call it instead; the answer is the same
    float32 ``(2, n_ommatidia, 2)`` array, left eye first.
    """

    def __init__(self, sim, fly_name: str, *, check: bool = True):
        self.sim = sim
        self.fly_name = fly_name
        if sim.retina is None:
            from flygym.vision.retina import Retina

            sim.retina = Retina()
        retina = sim.retina
        src = fisheye_source_index(retina)
        ids = retina.ommatidia_id_map.ravel().astype(np.int64)
        use = np.flatnonzero((ids > 0) & (src >= 0))
        self._src = np.ascontiguousarray(src[use])
        self._hexel = np.ascontiguousarray(ids[use] - 1)
        self._pixels = np.ascontiguousarray(retina.num_pixels_per_ommatidia.astype(np.int64))
        self._channel = np.ascontiguousarray(np.asarray(retina.pale_type_mask).astype(np.int64)[self._hexel])
        self._n = int(self._pixels.size)
        if check:
            self.check()

    def check(self, seed: int = 0) -> None:
        """Raise unless the gather matches FlyGym's two passes, value for value."""
        retina = self.sim.retina
        rng = np.random.default_rng(seed)
        image = rng.integers(0, 256, size=(retina.nrows, retina.ncols, 3), dtype=np.uint8)
        stock = retina.raw_image_to_hex_pxls(retina.correct_fisheye(image))
        ours = self._one_eye(image)
        if not np.array_equal(stock, ours):
            worst = float(np.abs(stock - ours).max())
            raise RuntimeError(
                "flyplay.retina_fast no longer reproduces FlyGym's ommatidia readout "
                f"(largest difference {worst:g}). The retina's fisheye or pooling has "
                "changed; see the module docstring before trusting any result.")

    def _one_eye(self, image: np.ndarray) -> np.ndarray:
        return _gather(image.reshape(-1, 3), self._src, self._hexel, self._channel, self._pixels, self._n)

    def __call__(self) -> np.ndarray:
        sim = self.sim
        if sim.eye_renderer is None:
            # Builds the renderer and its scene option, on this thread -- which
            # is the whole point of `FlyServer.build` running on the sim thread.
            sim.get_raw_vision(self.fly_name)
        eyes = [self._one_eye(self._render(camera))
                for camera in sim._intern_eye_camera_ids_by_fly[self.fly_name]]
        return np.array(eyes, dtype=np.float32)

    def _render(self, camera) -> np.ndarray:
        sim = self.sim
        sim.eye_renderer.update_scene(sim.mj_data, camera, scene_option=sim.eye_renderer_scene_option)
        return sim.eye_renderer.render()

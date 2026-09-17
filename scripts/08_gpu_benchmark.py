"""Measure the MJWarp GPU backend against the CPU one, on this machine.

FlyGym advertises a large GPU speed-up, but two things have to be true before
you see it, and both are easy to miss:

1. **Batch size.** The GPU wins by running thousands of worlds at once, not by
   running one world faster. FlyGym's own scaling test puts peak throughput at
   2,000-17,000 parallel worlds; below a few hundred the GPU loses to the CPU.
2. **Keeping the loop on the GPU.** A Python loop that uploads control inputs,
   steps, and waits every timestep spends most of its time on CPU-GPU
   synchronisation. Capturing the inner loop as a CUDA graph and feeding the
   controls from a Warp kernel removes that.

This script measures both, so you can see the gap rather than take it on faith.
Reference points from the FlyGym docs: RTX 3080 Ti ~30x real time, L40S / H100
~60x real time.

    python scripts/08_gpu_benchmark.py
    python scripts/08_gpu_benchmark.py --worlds 512 2048 8192
    python scripts/08_gpu_benchmark.py --naive          # also time the slow path

Run it with nothing else heavy going on -- even the graph-captured loop is
launched from Python, and a saturated CPU drags the numbers down. The first GPU
run compiles Warp kernels (~70 s here) into %LOCALAPPDATA%\\NVIDIA\\warp.
"""

import argparse
import time
import warnings

import _bootstrap  # noqa: F401
import numpy as np

from flyplay.build import build

SIM_DT = 1e-4


def cpu_benchmark(steps: int) -> float:
    """Single-world CPU throughput, as the baseline everything else divides by."""
    fs = build("flat", colorize=False)
    fs.sim.warmup()
    t0 = time.perf_counter()
    for _ in range(steps):
        fs.sim.step()
    elapsed = time.perf_counter() - t0
    fs.close()
    rate = steps / elapsed
    print(
        f"{'CPU, 1 world':>26} | {rate:12,.0f} steps/s | "
        f"{steps * SIM_DT / elapsed:7.2f}x | {'1.00':>8}x"
    )
    return rate


def _make_gpu_sim(n_worlds: int):
    from flygym.warp import GPUSimulation

    fs = build("flat", colorize=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = GPUSimulation(fs.world, n_worlds)
    return fs, sim


def gpu_naive(n_worlds: int, steps: int, baseline: float) -> None:
    """Python-driven loop: upload controls, step, synchronise, repeat."""
    fs, sim = _make_gpu_sim(n_worlds)
    sim.step()  # compile
    t0 = time.perf_counter()
    for _ in range(steps):
        sim.step()
    elapsed = time.perf_counter() - t0
    total = steps * n_worlds
    rate = total / elapsed
    print(
        f"{f'GPU naive, {n_worlds} worlds':>26} | {rate:12,.0f} steps/s | "
        f"{total * SIM_DT / elapsed:7.2f}x | {rate / baseline:8.2f}x"
    )
    fs.close()


def gpu_graph(n_worlds: int, steps: int, baseline: float) -> None:
    """CUDA-graph-captured loop with controls written by a Warp kernel.

    Everything inside the capture is replayed on the GPU with no Python in
    between. Note that CPU code inside the capture block runs once, at capture
    time, and never again -- which is why the profiled step variants cannot be
    used here and the timing is a single wall-clock measurement outside.
    """
    import warp as wp
    from flygym.compose import ActuatorType

    @wp.kernel
    def hold_pose_kernel(
        neutral_gpu: wp.array(dtype=wp.float32),  # type: ignore
        out_gpu: wp.array2d(dtype=wp.float32),  # type: ignore
    ):
        world_id, dof_id = wp.tid()
        out_gpu[world_id, dof_id] = neutral_gpu[dof_id]

    fs, sim = _make_gpu_sim(n_worlds)
    n_dofs = len(fs.dof_order)

    from flygym_demo.complex_terrain import PreprogrammedSteps

    neutral = PreprogrammedSteps().default_pose_by_dof_order(fs.dof_order)
    neutral_gpu = wp.array(neutral.astype(np.float32))
    ctrl_gpu = wp.zeros((n_worlds, n_dofs), dtype=wp.float32)

    sim.set_leg_adhesion_states(fs.name, np.ones((n_worlds, 6), dtype=np.float32))
    sim.warmup()

    with wp.ScopedCapture() as capture:
        wp.launch(
            hold_pose_kernel,
            dim=(n_worlds, n_dofs),
            inputs=[neutral_gpu],
            outputs=[ctrl_gpu],
        )
        sim.set_actuator_inputs(fs.name, ActuatorType.POSITION, ctrl_gpu)
        sim.step()

    t0 = time.perf_counter()
    for _ in range(steps):
        wp.capture_launch(capture.graph)
    wp.synchronize()
    elapsed = time.perf_counter() - t0

    total = steps * n_worlds
    rate = total / elapsed
    print(
        f"{f'GPU graph, {n_worlds} worlds':>26} | {rate:12,.0f} steps/s | "
        f"{total * SIM_DT / elapsed:7.2f}x | {rate / baseline:8.2f}x"
    )
    fs.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--worlds",
        nargs="+",
        type=int,
        default=[512, 2048, 8192],
        help="Parallel world counts to sweep (default: %(default)s).",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument(
        "--naive",
        action="store_true",
        help="Also time the Python-driven loop, to see what the graph buys.",
    )
    parser.add_argument("--skip-gpu", action="store_true")
    args = parser.parse_args()

    print(
        "'steps/s' counts world-steps: one step of a 2048-world batch is 2048.\n"
        "'x real' is total simulated seconds per wall-clock second.\n"
        "'vs CPU' divides by single-world CPU throughput.\n"
    )
    print(f"{'backend':>26} | {'throughput':>19} | {'x real':>7} | {'vs CPU':>9}")
    print("-" * 70)

    baseline = cpu_benchmark(args.steps * 4)
    if args.skip_gpu:
        return

    try:
        from flygym.warp.utils import check_gpu

        check_gpu()
    except ImportError:
        print("\nwarp-lang / mujoco_warp not installed; skipping the GPU sweep.")
        print('  uv pip install "mujoco_warp>=3.9,<3.10" "warp-lang>=1.14,<1.15"')
        return
    except Exception as e:
        print(f"\nNo usable GPU: {type(e).__name__}: {e}")
        return

    for n in args.worlds:
        for label, fn in (("naive", gpu_naive), ("graph", gpu_graph)):
            if label == "naive" and not args.naive:
                continue
            try:
                fn(n, args.steps, baseline)
            except Exception as e:
                # Out of memory is the expected failure as the batch grows.
                print(f"{f'GPU {label}, {n} worlds':>26} | {type(e).__name__}: {str(e)[:60]}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()

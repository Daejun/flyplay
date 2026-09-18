"""Run an A/B experiment set in the background: many flies at once, one CPU core each.

    python scripts/17_run_experiment.py --list
    python scripts/17_run_experiment.py --set shock_odour
    python scripts/17_run_experiment.py --set shock_odour --flies 12 --seconds 45
    python scripts/17_run_experiment.py --dir out/experiments/20260917-183512-shock_odour
    python scripts/17_run_experiment.py --report out/experiments/20260917-183512-shock_odour

Every set in `flyplay.experiment.EXPERIMENT_SETS` compares two conditions, A
and B, that differ in one thing. Each condition runs `--flies` flies; fly k of
A and fly k of B share their wiring, random numbers and start headings. Trials
are written to the experiment directory as they finish, so the web viewer can
show progress and replay them while the run goes on (`09_web_viewer.py
--sandbox`, 실험 tab), and a report is written at the end.

`--dir` runs a directory whose protocol.json the viewer has already written; a
file named STOP in that directory ends the run, keeping what was recorded.

Why CPU cores and not the GPU, measured on this laptop (Core Ultra 7 356H,
RTX 5060 Laptop), totals over all flies in simulated seconds per wall second:

    CPU, whole sandbox flies    1 / 8 / 14 processes      0.72 / 4.5 / 6.5x
    GPU, walking physics only   256 / 1024 / 4096 worlds  8.5 / 11.5 / 12.8x
    GPU, + 500 Hz CPU round trip for the controller       8.5 / 11.1 / 12.6x

The GPU's ceiling is twice the CPU's, but only for the physics: the mushroom
body, behaviour rules and the eyes would all have to be rewritten in batched
form and added on top, and MuJoCo-Warp turns off the noslip iterations the CPU
model uses (it warns), so walking itself would have to be re-validated. The
first CPU measurement read 3.75x at 15 processes because another viewer's idle
thread pools were spinning on 15 cores; see the thread settings below.
"""

import os

# Idle thread pools that spin instead of sleeping took 14.7 cores in one
# sandbox viewer process, and slowed everything else on the machine:
#   numba runs FlyGym's retina as parallel loops, 10 times a second, on the
#   OpenMP layer, whose workers busy-wait after each loop for longer than the
#   gap to the next one;
#   numpy's OpenBLAS keeps a thread per core for the mushroom body's small
#   matrix products.
# Measured on a lone sandbox fly, 400 steps: spinning, 13.7 ms a step on 4.7
# cores; OpenMP waiting passively, 13.8 ms on 1.1 cores; one numba thread,
# 19.7 ms. Must be set before numpy and numba are first imported.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_BLOCKTIME", "0")
# A worker pinned to one core has nothing to gain from numba's parallel loops:
# its 16 OpenMP threads then share that one core (the fisheye correction took
# 1.90 ms pinned against 1.82 ms on one thread). The correction is a pixel
# copy, so the readout is the same whichever way it is split.
for _threads in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_threads, "1")

import argparse
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import _bootstrap  # noqa: F401

from flyplay.experiment import (
    EXPERIMENT_SETS,
    Protocol,
    jobs_for,
    new_experiment_dir,
    protocol_for,
    read_json,
    read_records,
    run_fly_job,
    write_json,
)


def worker_cores(workers: int) -> list[int]:
    """One core per worker, counting up from 1 so that core 0 is left to the
    viewer and the desktop. Empty when there are not enough cores to go round:
    pinning two workers onto one core is worse than letting Windows move them."""
    n = os.cpu_count() or 1
    return list(range(1, workers + 1)) if workers < n else []


def pin_worker(cores: "multiprocessing.Queue[int]") -> None:
    """Give this worker its own core.

    A thread that holds an OpenGL context on the RTX is kept on the four P
    cores by Windows however busy they are, and never moved to an E core --
    measured on one process, 10 s: a plain Python loop used 0.96 of a core,
    the same loop with an RTX context 0.37, with an Intel context 0.95, and
    pinned to an E core 0.94. Every worker renders the eyes, so 14 of them
    shared 4 cores. Pinned, the same 14 wall-following flies ran in 199-309 s
    of wall time against 409-426 s, with identical records.
    """
    try:
        import psutil

        psutil.Process().cpu_affinity([cores.get_nowait()])
    except Exception as error:  # noqa: BLE001 -- a slower run beats no run
        print(f"  worker not pinned: {error!r}", file=sys.stderr, flush=True)


def write_status(directory: Path, protocol: Protocol, state: str, started: float, **extra) -> None:
    """Progress for the viewer. Only advisory: a status file that cannot be
    written is reported and skipped, never the end of the run."""
    done = len(read_records(directory))
    try:
        write_json(directory / "status.json", {
            "state": state,
            "pid": os.getpid(),
            "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_s": round(time.time() - started, 1),
            "done_trials": done,
            "total_trials": protocol.total_trials(),
            **extra,
        })
    except OSError as error:
        print(f"  status not written: {error!r}", file=sys.stderr, flush=True)


def finish(directory: Path) -> None:
    """The CSV and the report, from whatever the directory holds."""
    from flyplay.report import write_outputs

    write_outputs(directory)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="List the experiment sets and exit.")
    parser.add_argument("--set", help="Experiment set key (see --list).")
    parser.add_argument("--dir", type=Path, help="Run the protocol.json already in this directory.")
    parser.add_argument("--report", type=Path, help="Only rewrite the CSV and report of this directory.")
    parser.add_argument("--flies", type=int, help="Flies per condition (default: the set's).")
    parser.add_argument("--seconds", type=float, help="Length of every trial, overriding the set's.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--replay-flies", type=int, default=3,
                        help="Flies per condition recorded for replay (-1 all, 0 none).")
    parser.add_argument("--jobs", type=int, help="Worker processes (default: cores - 2).")
    parser.add_argument("--no-pin", action="store_true",
                        help="Let Windows place the workers (see pin_worker: about half the speed).")
    args = parser.parse_args()

    if args.list:
        for key, s in EXPERIMENT_SETS.items():
            p = protocol_for(key)
            # Keys and numbers only: the console is cp949 and titles are Korean.
            print(f"{key:20s} flies {p.flies:>2}  trials/fly {p.trials_per_fly(0)}  "
                  f"simulated {p.simulated_seconds() / 60:5.1f} min")
        return
    if args.report:
        finish(args.report)
        print("report written")
        return

    if args.dir:
        directory = args.dir.resolve()
        protocol = Protocol.from_dict(read_json(directory / "protocol.json"))
    elif args.set:
        if args.set not in EXPERIMENT_SETS:
            raise SystemExit(f"unknown set {args.set!r}; see --list")
        protocol = protocol_for(args.set, flies=args.flies, seconds=args.seconds, seed=args.seed,
                                replay_flies=args.replay_flies)
        directory = new_experiment_dir(tag=args.set)
        write_json(directory / "protocol.json", protocol.to_dict())
    else:
        raise SystemExit("give --set, --dir, --report or --list")

    jobs = jobs_for(protocol, directory)
    # Two cores stay free for the viewer and the desktop.
    workers = args.jobs or max(1, min(len(jobs), (os.cpu_count() or 4) - 2))
    cores = [] if args.no_pin else worker_cores(workers)
    started = time.time()
    print(f"experiment {protocol.set_key} -> {directory}")
    print(f"  {len(protocol.conditions)} conditions x {protocol.flies} flies, "
          f"{protocol.total_trials()} trials, {protocol.simulated_seconds() / 60:.1f} simulated min, "
          f"{workers} workers{', cores ' + str(cores[0]) + '-' + str(cores[-1]) if cores else ', not pinned'}",
          flush=True)
    write_status(directory, protocol, "running", started, workers=workers, pinned=bool(cores))

    state = "done"
    pool_args = {}
    if cores:
        # The queue is passed as a Process argument, which is the one way a
        # multiprocessing queue survives spawn; each worker takes one core.
        queue: "multiprocessing.Queue[int]" = multiprocessing.Queue()
        for core in cores:
            queue.put(core)
        pool_args = {"initializer": pin_worker, "initargs": (queue,)}
    try:
        with ProcessPoolExecutor(max_workers=workers, **pool_args) as pool:
            futures = [pool.submit(run_fly_job, job) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                if result.get("stopped"):
                    state = "stopped"
                print(f"  condition {result['condition']} fly {result['fly']:>3}: "
                      f"{result['trials']} trials in {result['wall_s']:.0f} s", flush=True)
                write_status(directory, protocol, "running", started, workers=workers, pinned=bool(cores))
    except BaseException as error:  # noqa: BLE001 -- recorded, then re-raised
        state = "failed"
        write_status(directory, protocol, state, started, workers=workers, pinned=bool(cores), error=repr(error)[:300])
        raise
    finally:
        if state != "failed":
            if (directory / "STOP").exists():
                state = "stopped"
            write_status(directory, protocol, "reporting", started, workers=workers, pinned=bool(cores))
            try:
                finish(directory)
            except Exception as error:  # noqa: BLE001 -- the records are what matters
                print(f"  report failed: {error!r}", file=sys.stderr)
            write_status(directory, protocol, state, started, workers=workers, pinned=bool(cores))
    print(f"{state} in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()

"""Drive the fly from a browser. No MuJoCo window, no hijacked keys.

MuJoCo's native viewer binds most of the keyboard for itself, so a script that
adds its own shortcuts fights the viewer and usually loses. This serves the
simulation as an MJPEG stream with plain HTML controls instead, which means the
keys do what you asked and the UI can say what it is doing.

    python scripts/09_web_viewer.py                        # drive it yourself
    python scripts/09_web_viewer.py --odor --pillars 4     # ...with food and obstacles
    python scripts/09_web_viewer.py --policy nav_flat      # a finished policy drives
    python scripts/09_web_viewer.py --follow forage_odor   # watch a policy LEARN
    python scripts/09_web_viewer.py --conditioning         # watch a fly learn a SMELL
    python scripts/09_web_viewer.py --conditioning --modality colour   # ...or a COLOUR
    python scripts/09_web_viewer.py --sandbox              # a room you furnish live

`--follow` is the one to use while training is running: it plays the newest
checkpoint, swaps in a fresher one whenever the trainer writes it, and shows
how many of the last 20 episodes ended at the food. The fly visibly gets better.

`--conditioning` runs the olfactory protocol from `flyplay.conditioning` in
place of an RL policy and shows the mushroom body carrying it out: which Kenyon
cells the current smell lights up, the approach MBON sagging every time
dopamine arrives with the punished odour, and the moment the learned valence
drags the steering drive through zero and approach becomes avoidance. There is
no vision in that mode, so the compound-eye panel is not shown.

`--modality colour` runs the floor-colour assay of Vogt et al. (2014) instead:
the eyes are on, the compound-eye panel shows each ommatidium in the colour it
reads, and the dashboard adds the visual projection neurons, both eyes' visual
Kenyon cells, and the learned valence each eye sees -- the difference between
the two is the turn.

`--sandbox` puts one fly in the sealed room of `flyplay.sandbox` and hands the
room to the page: click the map to drop sugar, shock zones, odours, coloured
floor and obstacles, drag them around, and watch the fly react and learn.

Then open http://localhost:8000 . Controls: WASD or arrow keys, or the on-screen
buttons. The camera follows the fly; switch between side, top and close-up views
in the page.

FlyGym also ships an official browser build that needs no Python at all:
  game   https://neuromechfly.org/wasm/game/game.html
  viewer https://neuromechfly.org/wasm/viewer/viewer.html
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
for _threads in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_threads, "1")

import argparse
import base64
import csv
import io
import json
import queue
import subprocess
import sys
import threading
from dataclasses import asdict
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import _bootstrap  # noqa: F401
import mujoco
import numpy as np
import psutil
from PIL import Image

from flyplay.arena import ring_positions
from flyplay.build import TERRAINS, build, display_colours
from flyplay.conditioning import (
    INNATE_VALENCE,
    ODOUR_NAMES,
    ConditioningConfig,
    ConditioningExperiment,
    TrialResult,
    colour_config,
    experiment_from_memory,
    read_memory,
    session_plan,
)
from flyplay.control import DEFAULT_DECIMATION, RealtimePacer, Walker
from flyplay.env_multimodal import OBS_SLICES
from flyplay.odor import OdorField, OdorSource
from flyplay.sandbox import (
    ODOUR_LABELS,
    ODOURS,
    PRESET_GROUPS,
    PRESET_NOTES,
    PRESETS,
    SUGARS,
    Sandbox,
    load_preset,
)
from flyplay.experiment import (
    EXPERIMENT_SETS,
    EXPERIMENTS_DIR,
    MODES,
    PHASE_LABELS,
    ROLE_LABELS,
    SET_GROUPS,
    VERDICT_TEXT,
    Protocol,
    describe_steps,
    estimate_wall_seconds,
    evaluate,
    furnish,
    metric_value,
    new_experiment_dir,
    prediction_choices,
    prediction_text,
    protocol_for,
    read_json,
    room_items,
    room_label,
    write_json,
)
from flyplay.report import build_report, controlled_variables, svg_room
from flyplay.sandbox import METRICS_BY_KEY, duration_ko
from flyplay.vision import RetinaFeatures
from flyplay.visual_pathway import eye_layout, photoreceptors

SIGNAL_LOW, SIGNAL_HIGH = 0.4, 1.6
MAX_TURN = 0.6
#: Samples kept in each rolling trace. At the 25 Hz sampling rate this is a
#: little under 10 s, so a whole 8 s episode fits in the window.
TRACE_LEN = 240
#: Observations held aside as the fixed probe set for the attribution history.
PROBE_BATCH = 64
#: Samples kept in each conditioning trace, and the rate they are taken at.
#: Measured in *simulated* seconds, not wall clock: a trial is 5 s long and
#: yields 20 samples, so 800 spans 40 trials -- the whole default session, with
#: room left over. The policy traces above cannot be reused for this; they hold
#: 10 s, which is one and a half trials and tells you nothing about a memory
#: built over dozens of them.
COND_TRACE_LEN = 800
COND_SAMPLE_HZ = 4.0
#: Protocol block -> page label. Korean never reaches a print(): the Windows
#: console is cp949 and one non-ASCII character there takes the server down.
BLOCK_LABELS = {
    "naive": "순진",
    "train": "훈련",
    "acquisition": "학습 후",
    "reversal_train": "반전 훈련",
    "reversal": "반전 후",
    "memory_test": "기억",
}
#: Sensor names as the dashboard labels them, in `flyplay.odor.SENSOR_NAMES` order.
SENSOR_LABELS = ("L palp", "R palp", "L antenna", "R antenna")
#: (label, observation block, perturbation) for the sensory attribution panel.
ATTRIBUTION_CHANNELS = (
    ("냄새 좌우차", "odor_asymmetry", 0.5),
    ("냄새 변화", "odor_change", 0.5),
    ("냄새 세기", "odor_level", 0.2),
    ("시각", "vision", 0.5),
)

# The camera is a tracking MjvCamera rather than one of the model's fixed
# cameras, so distance and angle are free to change while it still follows the
# fly. Distances are in mm; the animal is about 3 mm long.
VIEWS = {
    "side": (130.0, -12.0, 13.0),
    "top": (90.0, -89.0, 20.0),
    "close": (140.0, -8.0, 5.5),
    "wide": (130.0, -25.0, 34.0),
}
#: The whole sealed room from above: a free camera, not a tracking one.
ROOM_VIEW = (90.0, -89.0, 175.0)
ZOOM_STEP = 1.3
#: Dashboard samples kept in sandbox mode, taken every 10 action steps (10 Hz of
#: simulated time): 40 s of dopamine and hunger.
SANDBOX_TRACE_LEN = 400
#: Item kinds as the page names them. Only ever used for the page's event log.
KIND_LABELS = {"sugar": "설탕", "shock": "전기 구역", "odour": "냄새",
               "patch": "색 바닥", "obstacle": "장애물"}
#: Room edits the page can take back.
UNDO_DEPTH = 200
#: One fly's speed in a background experiment, simulated seconds per wall
#: second: 14 sandbox processes at once ran at 0.44-0.50x each (measured, with
#: idle threads sleeping). The page's time estimate divides by it.
POOL_SPEED_PER_FLY = 0.46
#: Worker processes a background experiment uses: 17_run_experiment.py's default.
POOL_WORKERS = max(1, (os.cpu_count() or 4) - 2)
#: Page commands that do not touch the live fly. Anything else ends a replay first.
REPLAY_OPS = ("replay", "replay_ctl", "follow", "exp_start", "exp_stop")


def pi_sides(protocol: Protocol) -> tuple[str, str] | None:
    """The sides a preference index counts +1 and -1, in words, in the room the
    verdict is judged in; None for other measures. The metric's own sentence
    points at the report's "냄새 선호 기준" column, which the lab does not show."""
    if protocol.metric == "colour_pi":
        return ("파랑", "초록")
    if protocol.metric != "odour_pi":
        return None
    condition = protocol.conditions[0]
    present = {it["params"].get("odour") for it in room_items(condition.steps[-1].room, condition, protocol.layout)
               if it["kind"] == "odour"}
    # Sandbox.odour_pair: octanol against MCH whenever both are there, otherwise in ODOURS order.
    pair = ("octanol", "mch") if {"octanol", "mch"} <= present else tuple(o for o in ODOURS if o in present)
    return (ODOUR_LABELS[pair[0]], ODOUR_LABELS[pair[1]]) if len(pair) == 2 else None


def measure_words(protocol: Protocol) -> str:
    """The deciding measure explained for the lab."""
    metric = METRICS_BY_KEY[protocol.metric]
    sides = pi_sides(protocol)
    if sides is None and metric.latency:
        # The metric's own sentence says a latency that never ended is left
        # blank, as the CSV has it; the comparison counts it as the whole trial.
        kept = [part for part in metric.meaning.split(". ") if "빈칸" not in part]
        return ". ".join(kept).rstrip(".") + ". 시행 시간 안에 일어나지 않으면 시행 시간으로 셉니다."
    if sides is None:
        return metric.meaning
    plus, minus = sides
    return (f"방을 {plus}에 더 가까운 쪽과 {minus}에 더 가까운 쪽으로 나누어, 어느 쪽에 오래 머물렀는지 잽니다. "
            f"+1이면 시행 내내 {plus} 쪽, -1이면 내내 {minus} 쪽, 0이면 반반입니다.")


def _load_meta(run: str) -> dict:
    meta_path = _bootstrap.OUT / "rl" / run / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def _newest_checkpoint(run: str):
    """Newest checkpoint by modification time, with its step count.

    Sorting by the step number in the filename would be wrong while a fresh
    run is overwriting a previous one's checkpoints -- an old ``ppo_200000``
    outranks a new ``ppo_10000`` by number but not by time.
    """
    ckpt_dir = _bootstrap.OUT / "rl" / run / "checkpoints"
    files = list(ckpt_dir.glob("ppo_*_steps.zip")) if ckpt_dir.exists() else []
    if not files:
        return None, 0
    newest = max(files, key=lambda p: p.stat().st_mtime)
    try:
        steps = int(newest.stem.split("_")[1])
    except (IndexError, ValueError):
        steps = 0
    return newest, steps


def _viewer_plan(
    n_train: int,
    n_test: int,
    cs_plus: str,
    reversal: bool,
    differential: bool = False,
    names: tuple[str, ...] = ODOUR_NAMES,
):
    """`conditioning.session_plan`, plus which odour the valence panel follows.

    The schedule itself comes from `flyplay.conditioning`, so this viewer and
    `run_session` cannot drift apart. The extra element is display-only: the
    panel tracks the CS+ until the reversal and the new CS+ after it, instead
    of watching a memory that is no longer being written.

    The third element is the odour a "train" or "expose" trial presents. Only a
    "train" trial punishes it -- callers must not read it as the CS+ otherwise.
    """
    other = names[1] if cs_plus == names[0] else names[0]
    return [
        (block, kind, odour, other if block.startswith("reversal") else cs_plus)
        for block, kind, odour in session_plan(
            n_train, n_test, cs_plus, reversal, differential, names
        )
    ]


class FlyServer:
    """Runs the simulation in one thread and hands out JPEG frames.

    Everything that touches OpenGL is built inside `run`, on the simulation
    thread. This is not tidiness: an OpenGL context belongs to whichever thread
    made it current, and creating the viewer's renderer on a *different* thread
    from the fly's eye cameras silently breaks the eyes rather than raising.
    Measured, with the env built on the main thread and the viewer's renderer
    on the worker: mean ommatidia intensity fell from 0.602 to 0.012 -- the fly
    goes blind, and because near-black readouts still produce numbers, nothing
    complains. Same thread for both, and the readouts are correct.
    """

    def __init__(self, args):
        self.args = args
        self.ready = threading.Event()
        self.failure: BaseException | None = None
        self._frame: bytes | None = None
        self._frame_lock = threading.Lock()
        self._new_frame = threading.Condition(self._frame_lock)
        self._eye_frame: bytes | None = None
        self._eye_lock = threading.Condition(threading.Lock())
        self._stop = threading.Event()
        self.status: dict = {}
        self.neuro_snapshot: dict = {}

    def build(self) -> None:
        """Construct the simulation. Must run on the simulation thread."""
        args = self.args
        self.policy = None
        self.env = None
        self._obs = None
        #: The colour task is on. Set for real in the conditioning branch.
        self.colour = False
        #: Sandbox mode: the sealed room, furnished live from the page.
        self.sandbox_mode = bool(args.sandbox)
        self.sandbox: Sandbox | None = None
        #: Page edits for the sandbox. HTTP threads only enqueue; the simulation
        #: thread applies them between steps, so nothing touches the model
        #: mid-step.
        self.commands: queue.SimpleQueue = queue.SimpleQueue()
        self.preset_name = args.preset

        self.conditioning = bool(args.conditioning)
        self.follow = bool(args.follow)
        run = args.follow or args.policy
        if self.sandbox_mode:
            self.meta = {}
            args.terrain = "flat"
            self.sandbox = Sandbox(seed=args.seed)
            load_preset(self.sandbox, args.preset)
            self.fs = self.sandbox.fs
            # The sandbox fly is built plain so its eyes never see coloured legs;
            # only this render gets NeuroMechFly's colours.
            self._display_colours = display_colours(self.fs.fly, self.fs.sim)
            #: Room edits that can be undone, newest last, and those undone.
            self._undo: list[dict] = []
            self._redo: list[dict] = []
            #: The running trial, results so far, and the layout trials repeat.
            self.sb_trial: dict | None = None
            self.trial_rows: list[dict] = []
            self._trial_count = 0
            self._trial_layout: list | None = None
            self._room_edited = False
            self._yaw_rng = np.random.default_rng(args.seed + 1)
            #: (wall s, simulated s) pairs for the page's real speed readout.
            self._speed_samples: deque = deque(maxlen=60)
            #: Background experiments started from this page, by directory name.
            self._batches: dict[str, subprocess.Popen] = {}
            #: The trial being replayed (`_replay_start`), or None.
            self.replay: dict | None = None
            #: An experiment whose new trials are replayed as they are recorded.
            self.follow_id: str | None = None
            self._followed: set = set()
            self._follow_checked = 0.0
            #: Parsed trial records and line counts per records file, reused
            #: while the file's size and modification time stay the same.
            self._records_cache: dict = {}
            self._counts_cache: dict = {}
            self._last_exp_start: tuple = ("", -1e9)
            #: The lab's set descriptions, rooms drawn: built once, they never change.
            self._sets_payload: dict | None = None
            #: Verdicts for the experiment list, by (directory, finished trials).
            self._verdicts: dict = {}
        elif self.conditioning:
            self.meta = {}
            args.terrain = "flat"  # the protocol lays out its own flat arena
            if args.vision:
                # The colour task renders the eyes itself, inside the trial loop;
                # the odour task has no use for them.
                print("conditioning chooses its own vision; ignoring --vision")
                args.vision = False
            #: The memory file this fly was restored from, or None for a fresh fly.
            self.memory_path = None
            self.memory_info: dict = {}
            if args.memory:
                self.memory_path = _bootstrap.MEMORIES / f"{args.memory}.npz"
                if not self.memory_path.exists():
                    filed = sorted(p.stem for p in _bootstrap.MEMORIES.glob("*.npz"))
                    raise SystemExit(
                        f"no memory named {args.memory!r}; filed: {filed}. "
                        f"See scripts/16_memory.py."
                    )
                # The file supplies the wiring and the fly's configuration;
                # --seed only varies where the sources land trial to trial.
                self.exp = experiment_from_memory(self.memory_path, seed=args.seed)
                self.memory_info = read_memory(self.memory_path)
                # A remembered fly is here to show what it does with what it
                # knows, so the session is tests only. The valence panel follows
                # whichever odour this memory fears more.
                summary = self.memory_info.get("summary") or {}
                names = self.exp.stimulus_names
                feared = min(names, key=lambda o: summary.get(o, {}).get("valence", 0.0))
                colour = self.exp.config.modality == "colour"
                # A colour test is 90 s against 5 s for an odour test.
                trials = args.memory_trials or (2 if colour else 12)
                self.plan = [("memory_test", "test", None, feared)] * trials
            else:
                colour = args.modality == "colour"
                config = colour_config() if colour else ConditioningConfig()
                self.exp = ConditioningExperiment(config, seed=args.seed)
                names = self.exp.stimulus_names
                cs_plus = args.cs_plus or names[0]
                if cs_plus not in names:
                    raise SystemExit(f"--cs-plus must be one of {names} for this task")
                # Colour follows Vogt et al. (2014): four CS+ and four CS-
                # periods and one test, no reversal. The odour defaults are the
                # viewer's shortened protocol.
                n_train = args.train_trials or (8 if colour else 12)
                n_test = args.test_trials or (1 if colour else 4)
                self.plan = _viewer_plan(
                    n_train, n_test, cs_plus, (not args.no_reversal) and not colour,
                    self.exp.config.differential, names,
                )
            #: Whether this session is the colour task -- decided by the
            #: experiment, since a restored memory brings its own config.
            self.colour = self.exp.config.modality == "colour"
            self.fs = self.exp.fs
            #: Each odour's Kenyon-cell code, read once without learning, drawn
            #: as the raster's backdrop. Measured on seed 0: A and B share 3 of
            #: their 100 cells, and A at a tenth the concentration keeps 99 of
            #: them -- which is the one thing the panel exists to show, so the
            #: live pattern needs both constellations to sit against.
            if self.colour:
                # The colour task's backdrop: each colour's visual code as seen
                # standing on it. Measured: blue and green share no cells.
                visual = self.exp.mb.visual
                self.kc_ref = {
                    name: [int(i) for i in np.flatnonzero(
                        visual(self.exp._unit_readouts[name]).mean_kc)]
                    for name in self.exp.stimulus_names
                }
            else:
                self.kc_ref = {
                    name: [int(i) for i in self.exp.read(name).olfactory.active]
                    for name in ODOUR_NAMES
                }
        elif run:
            # The policy needs its env's goal sampling and observation
            # assembly, so build through the env rather than standalone.
            args.policy = run
            self.meta = _load_meta(run)
            terrain = self.meta.get("terrain", args.terrain)
            if terrain != args.terrain:
                print(f"using the policy's terrain '{terrain}'")
                args.terrain = terrain

            if self.meta.get("task") == "forage":
                from flyplay.env_multimodal import ForageConfig, ForageEnv

                self.env = ForageEnv(
                    ForageConfig(
                        terrain=terrain,
                        n_pillars=self.meta.get("n_pillars", 2),
                        episode_seconds=10.0,
                    ),
                    seed=args.seed,
                )
            else:
                from flyplay.env import FlyNavEnv, NavConfig

                self.env = FlyNavEnv(
                    NavConfig(terrain=terrain, episode_seconds=8.0), seed=args.seed
                )
            self.fs = self.env.fs
        else:
            self.meta = {}
            # Props only exist if asked for: an empty flat world is the fast
            # default, and vision costs 25 ms every time it is sampled.
            field = None
            if args.odor:
                field = OdorField(
                    sources=[OdorSource(pos=(22.0, 0.0, 1.5), peak=(1.0,))]
                )
            self.fs = build(
                args.terrain,
                vision=args.vision,
                odor_field=field,
                pillars=tuple(
                    ring_positions(args.pillars, radius=11.0, start_deg=-50.0)
                ),
                seed=args.seed,
            )
        # The renderer is created in run(), on the simulation thread. An
        # OpenGL context belongs to the thread that made it current, so
        # building it here (main thread) and rendering there fails with
        # "WGL: Failed to make context current: the resource is in use".
        self.renderer: mujoco.Renderer | None = None

        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = self.fs.thorax_body_id
        self.view = "side"
        self.cam.azimuth, self.cam.elevation, self.cam.distance = VIEWS[self.view]
        if self.sandbox_mode:
            self.set_view("room")
        if self.env is not None:
            self.walker = self.env.walker
            self._load_policy(args.policy)
        elif self.sandbox_mode:
            self.walker = self.sandbox.walker
        elif self.conditioning:
            # The experiment owns the walker; every trial resets it itself.
            self.walker = self.exp.walker
        else:
            self.walker = Walker(self.fs, decimation=args.decimation, seed=args.seed)
            self.walker.reset()

        self.retina = RetinaFeatures() if args.vision else None
        self.episodes = 0
        self.reloads = 0
        self.outcomes: deque[str] = deque(maxlen=20)

        # Rolling traces for the neural dashboard. TRACE_LEN samples at the
        # sampling rate below is roughly the last few seconds.
        self.neuro: dict = {}
        self.traces = {
            k: deque([0.0] * TRACE_LEN, maxlen=TRACE_LEN)
            for k in (
                "value", "rpe", "reward", "left", "right",
                "odor", "odor_l", "odor_r",
            )
        }
        if self.sandbox_mode:
            self.traces.update({
                k: deque([0.0] * SANDBOX_TRACE_LEN, maxlen=SANDBOX_TRACE_LEN)
                for k in ("punish", "sweet", "nutrient", "hunger")
            })
            self._sandbox_counter = 0
        #: 1 at samples where an episode ended, so the plots can mark them.
        #: Without this the value trace's sawtooth (V collapses when a new goal
        #: is placed far away) reads as noise rather than structure.
        self.episode_marks: deque[int] = deque([0] * TRACE_LEN, maxlen=TRACE_LEN)
        #: Live per-sample attribution, for the moment-to-moment line graph.
        self.attr_traces = {
            label: deque([0.0] * TRACE_LEN, maxlen=TRACE_LEN)
            for label, _, _ in ATTRIBUTION_CHANNELS
        }
        #: Attribution measured on every checkpoint on disk, so the history
        #: covers the whole training run instead of only what this viewer
        #: happened to be up for. checkpoint path -> (steps, {channel: mean})
        self.attr_history: dict[str, tuple[int, dict[str, float]]] = {}
        #: Fixed batch of observations the history is measured on. It has to be
        #: the same states for every checkpoint or the comparison is
        #: meaningless -- different states, not different policies, would
        #: explain the differences.
        self._probe_states: np.ndarray | None = None
        self._probe_pool: list[np.ndarray] = []
        self._neuro_counter = 0
        self._prev_value = None
        self.drive = 1.0
        self.turn = 0.0
        self.speed = args.speed
        self.paused = False
        self.reset_requested = False
        #: Set when a trial reset the physics clock and the pacer's anchor with
        #: it. Checked in `run` because only that loop owns the pacer.
        self._reset_pacer = False

        # --- conditioning session state ---
        self.plan_index = 0
        self.trial_no = 0
        self.trial_kind: str | None = None
        self.trial_cs: str | None = None
        self.block_key: str | None = None
        self.focus: str | None = None
        self.session_done = False
        #: One entry per finished test block: its mean preference index.
        self.blocks: list[dict] = []
        self._block_prefs: list[float] = []
        self._trial_iter = None
        self._cond_counter = 0
        self._mark_phase = False
        self.kc_active: list[int] = []
        self.cond_odour: str | None = None
        self.intensity = 0.0
        self.smelling = False
        self.dopamine = False
        # --- colour task ---
        #: Active visual Kenyon cells, one list per eye.
        self.vkc_active: list[list[int]] = [[], []]
        #: Last VPN outputs, (2, n_vpn), and the last visual pass for the eyes.
        self.vpn = None
        self.last_visual = None
        #: The floor colour under the fly, and seconds on each colour so far in
        #: the running test.
        self.under: str | None = None
        self.time_on: dict[str, float] = {}
        if self.conditioning:
            self.traces.update(
                {
                    k: deque([0.0] * COND_TRACE_LEN, maxlen=COND_TRACE_LEN)
                    for k in ("mbon_approach", "mbon_avoid", "valence", "drive")
                }
            )
            if self.colour:
                self.traces["valence_other"] = deque([0.0] * COND_TRACE_LEN,
                                                     maxlen=COND_TRACE_LEN)
                # Steering happens over a second, not over a session, so these
                # use the policy traces' 10 s window at the dashboard rate.
                self.traces.update({
                    k: deque([0.0] * TRACE_LEN, maxlen=TRACE_LEN)
                    for k in ("eye_left", "eye_right", "turn")
                })
            #: 1 at samples where dopamine was being delivered, and at block
            #: boundaries, so the plots can mark both.
            self.dan_marks = deque([0] * COND_TRACE_LEN, maxlen=COND_TRACE_LEN)
            self.phase_marks = deque([0] * COND_TRACE_LEN, maxlen=COND_TRACE_LEN)
            self._cond_every = max(
                1, int(round(self.exp.config.action_hz / COND_SAMPLE_HZ))
            )

    def _load_policy(self, run: str):
        from stable_baselines3 import PPO

        self._PPO = PPO
        run_dir = _bootstrap.OUT / "rl" / run
        if self.follow:
            # A stage that just started has neither a checkpoint nor a final
            # model yet. Waiting is the useful behaviour here -- the whole
            # point of --follow is to attach to a run already in progress.
            deadline = time.monotonic() + self.args.wait_for_model
            path, steps = _newest_checkpoint(run)
            while path is None and not (run_dir / "model.zip").exists():
                if time.monotonic() > deadline:
                    break
                print(f"waiting for '{run}' to write its first checkpoint...")
                time.sleep(5.0)
                path, steps = _newest_checkpoint(run)
            if path is None:
                path, steps = run_dir / "model.zip", self.meta.get("steps", 0)
        else:
            path, steps = run_dir / "model.zip", self.meta.get("steps", 0)

        if not path.exists():
            available = sorted(
                p.name
                for p in (_bootstrap.OUT / "rl").glob("*")
                if (p / "model.zip").exists() or (p / "checkpoints").exists()
            )
            raise SystemExit(f"Nothing to load at {path}. Runs found: {available}")

        self.policy = PPO.load(path, device="cpu")
        want = int(np.prod(self.policy.observation_space.shape))
        have = int(np.prod(self.env.observation_space.shape))
        if want != have:
            raise SystemExit(
                f"'{run}' was trained with a {want}-dim observation but the "
                f"environment built here gives {have}. The run's meta.json is "
                f"missing or says the wrong task, so the wrong environment was "
                f"constructed. Check {run_dir / 'meta.json'}."
            )
        self._policy_path = path
        self._policy_mtime = path.stat().st_mtime
        self._policy_steps = steps
        self._last_reload_check = 0.0
        self._obs, _ = self.env.reset(seed=self.args.seed)
        print(f"policy loaded: {path.name} ({steps:,} steps)")

    def _maybe_reload_policy(self) -> None:
        """Swap in a newer checkpoint while training is still running.

        Checked on a timer rather than every frame: a stat() per rendered frame
        is wasteful, and a policy that changes mid-episode is confusing to
        watch. The swap happens at an episode boundary.
        """
        now = time.monotonic()
        if now - self._last_reload_check < self.args.reload_every:
            return
        self._last_reload_check = now
        path, steps = _newest_checkpoint(self.args.policy)
        if path is None:
            return
        mtime = path.stat().st_mtime
        if mtime <= self._policy_mtime:
            return
        try:
            self.policy = self._PPO.load(path, device="cpu")
        except Exception as e:  # a half-written checkpoint; try again next tick
            print(f"  (checkpoint {path.name} not readable yet: {type(e).__name__})")
            return
        self._policy_path, self._policy_mtime, self._policy_steps = path, mtime, steps
        self.reloads += 1
        print(f"  -> reloaded {path.name} ({steps:,} steps)")

    def _read_brain(self, obs: np.ndarray) -> tuple[np.ndarray, float]:
        """Actor's last hidden layer and the critic's value, for this state."""
        import torch

        tensor, _ = self.policy.policy.obs_to_tensor(obs[None, :])
        with torch.no_grad():
            features = self.policy.policy.extract_features(tensor)
            feat_pi, feat_vf = (
                features if isinstance(features, tuple) else (features, features)
            )
            latent_pi = self.policy.policy.mlp_extractor.forward_actor(feat_pi)
            latent_vf = self.policy.policy.mlp_extractor.forward_critic(feat_vf)
            value = self.policy.policy.value_net(latent_vf)
        return latent_pi.numpy()[0], float(value)

    def _snapshot_brain(self, obs_before, reward, value_before, terminated, done) -> None:
        """Record one sample of the network's internal state.

        The reward-prediction error is the textbook mapping between temporal
        difference learning and dopamine (Schultz, Dayan & Montague 1997):

            delta = r + gamma * V(s') - V(s)

        Positive means the situation turned out better than the critic
        expected. In the fly this is the job of the dopaminergic neurons that
        write onto the mushroom body, so it is the closest thing this agent has
        to a dopamine trace.
        """
        hidden, value_after = self._read_brain(self._obs)
        gamma = self.policy.gamma
        rpe = reward + gamma * value_after * (0.0 if terminated else 1.0) - value_before

        self.traces["value"].append(value_before)
        self.traces["rpe"].append(rpe)
        self.traces["reward"].append(float(reward))
        self.episode_marks.append(1 if done else 0)
        left, right = self.walker.descending_signal
        self.traces["left"].append(float(left))
        self.traces["right"].append(float(right))
        self.traces["odor"].append(float(obs_before[OBS_SLICES["odor_level"]][0]))
        self.neuro["hidden"] = [round(float(v), 3) for v in hidden]
        self._attribution(obs_before)

    def _attribution(self, obs: np.ndarray) -> None:
        """How much each sense is moving the turn command right now.

        Nudge one sensory channel, see how far the turn command moves, put it
        back. No gradients needed, and the answer is in the units you care
        about: "the fly is currently turning because of smell, not sight".
        Kept as a trace rather than a snapshot, because the interesting thing
        is when a channel's influence *rises* -- that is the policy learning to
        use a sense it was ignoring.
        """
        for label, value in self._measure(self.policy, obs[None, :]).items():
            self.attr_traces[label].append(value)

        # Keep a fixed probe batch for the checkpoint-by-checkpoint history.
        if self._probe_states is None:
            self._probe_pool.append(obs.copy())
            if len(self._probe_pool) >= PROBE_BATCH:
                self._probe_states = np.asarray(self._probe_pool, dtype=np.float32)

    @staticmethod
    def _measure(policy, batch: np.ndarray) -> dict[str, float]:
        """Mean |change in turn command| when each sense is nudged.

        Works on a batch so one call covers many states; with a single row it
        is the instantaneous reading.
        """

        def turn(x):
            action, _ = policy.predict(x, deterministic=True)
            action = np.atleast_2d(action)
            return action[:, 1] - action[:, 0]

        base = turn(batch)
        out = {}
        for label, key, delta in ATTRIBUTION_CHANNELS:
            sl = OBS_SLICES[key]
            swing = np.zeros(len(batch))
            for sign in (+1.0, -1.0):
                probe = batch.copy()
                probe[:, sl] = probe[:, sl] + sign * delta
                swing += np.abs(turn(probe) - base)
            out[label] = float(np.mean(swing) / 2.0)
        return out

    def scan_checkpoints(self) -> None:
        """Measure sensory attribution on every checkpoint saved so far.

        Runs on its own thread. The alternative -- accumulating buckets as the
        viewer plays -- only ever covers the stretch of training the viewer
        happened to be running for, which is why the history used to show a
        single bar. The checkpoints are all on disk, so the whole run can be
        measured, including the part that finished before the viewer started.
        """
        ckpt_dir = _bootstrap.OUT / "rl" / self.args.policy / "checkpoints"
        while not self._stop.is_set():
            if self._probe_states is None or not ckpt_dir.exists():
                time.sleep(3.0)
                continue
            for path in sorted(ckpt_dir.glob("ppo_*_steps.zip")):
                if self._stop.is_set() or str(path) in self.attr_history:
                    continue
                try:
                    steps = int(path.stem.split("_")[1])
                    model = self._PPO.load(path, device="cpu")
                    means = self._measure(model, self._probe_states)
                except Exception:
                    continue  # still being written; it will be picked up later
                self.attr_history[str(path)] = (steps, means)
            time.sleep(self.args.reload_every)

    def _distance_to_goal(self) -> float:
        """Straight-line distance from the fly to whatever it is heading for."""
        if hasattr(self.env, "_distance"):
            return float(self.env._distance())
        goal = np.asarray(getattr(self.env, "goal", (0.0, 0.0)))
        return float(np.linalg.norm(goal - self.fs.thorax_pos()[:2]))

    def _turn_command(self, obs: np.ndarray) -> float:
        action, _ = self.policy.predict(obs, deterministic=True)
        action = np.asarray(action).ravel()
        return float(action[1] - action[0])

    def _policy_step(self) -> None:
        """One env step under the trained policy, restarting on episode end."""
        obs_before = self._obs
        sample = self._neuro_counter % self.args.neuro_decim == 0
        self._neuro_counter += 1
        value_before = 0.0
        if sample:
            _, value_before = self._read_brain(obs_before)

        action, _ = self.policy.predict(self._obs, deterministic=True)
        self._obs, reward, terminated, truncated, info = self.env.step(action)

        if sample:
            self._snapshot_brain(
                obs_before, reward, value_before, terminated, terminated or truncated
            )

        if terminated or truncated:
            self.outcomes.append(info["outcome"])
            self.episodes += 1
            print(
                f"  [{self._policy_steps:>7,} steps] episode {self.episodes}: "
                f"{info['outcome']} after {self.env.elapsed:.2f}s"
            )
            if self.follow:
                self._maybe_reload_policy()
            self._obs, _ = self.env.reset()

    # --- sandbox --------------------------------------------------------

    def set_view(self, name: str) -> None:
        """Switch camera preset. "room" frames the whole room from above with a
        free camera; the others track the fly."""
        if name == "room":
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.cam.lookat[:] = (0.0, 0.0, 0.0)
            self.cam.azimuth, self.cam.elevation, self.cam.distance = ROOM_VIEW
        elif name in VIEWS:
            self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.cam.trackbodyid = self.fs.thorax_body_id
            self.cam.azimuth, self.cam.elevation, self.cam.distance = VIEWS[name]
        else:
            return
        self.view = name

    def pan(self, dx: float, dy: float) -> None:
        """Slide the camera across the scene by a mouse drag of (dx, dy) pixels.

        A tracking camera is let go of the fly first, keeping the point it was
        looking at, so the view does not jump; a view button re-attaches it.
        """
        cam = self.cam
        if cam.type != mujoco.mjtCamera.mjCAMERA_FREE:
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.view = "free"
        azimuth = np.deg2rad(cam.azimuth)
        right = np.array([np.sin(azimuth), -np.cos(azimuth), 0.0])
        forward = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
        # Scaled by distance so a drag moves the scene about as far as the
        # pointer moved, zoomed in or out.
        scale = cam.distance * 0.0012
        cam.lookat[:] = np.asarray(cam.lookat) + (-right * dx + forward * dy) * scale

    def _drain_commands(self) -> bool:
        """Apply queued page edits. True if any was applied."""
        applied = False
        while True:
            try:
                cmd = self.commands.get_nowait()
            except queue.Empty:
                break
            try:
                self._apply_sandbox(cmd)
                applied = True
            except (KeyError, ValueError, TypeError, OSError) as error:
                # Shown in the page's event log. Never printed: the message can
                # carry non-ASCII, and the console is cp949.
                self.sandbox._event(f"적용하지 못했습니다: {error}")
        if applied:
            self.sandbox.refresh_telemetry()
        return applied

    def _apply_sandbox(self, cmd: dict) -> None:
        sb = self.sandbox
        op = cmd.get("op")
        if self.replay is not None and op not in REPLAY_OPS:
            # Editing the room or the fly is about the live fly: back to it first.
            self._replay_stop()
        if op in ("place", "move", "update", "remove", "clear", "preset"):
            self._edit(cmd)
        elif op == "undo":
            self._step_history(self._undo, self._redo, "undo", "되돌림")
        elif op == "redo":
            self._step_history(self._redo, self._undo, "redo", "다시 함")
        elif op == "history_clear":
            # For scripts that rebuild a room through the page's commands: the
            # rebuild itself should not be something a Ctrl+Z takes apart.
            self._undo.clear()
            self._redo.clear()
        elif op == "hunger":
            sb.hunger = float(np.clip(cmd["value"], 0.0, 1.0))
        elif op == "sugar_depletes":
            sb.sugar_depletes = bool(cmd["value"])
            sb._event("설탕이 먹는 만큼 줄어듭니다" if sb.sugar_depletes
                      else "설탕이 먹어도 줄지 않습니다")
        elif op == "new_fly":
            sb.reset_fly()
            self._clear_sandbox_traces()
            self._rewound()
        elif op == "fly_home":
            sb.return_home()
        elif op == "fly_move":
            self._fly_move(cmd)
        elif op == "trial_start":
            self._sb_trial_start(cmd)
        elif op == "trial_stop":
            self._sb_trial_finish(stopped=True)
        elif op == "trial_clear":
            self.trial_rows.clear()
            sb._event("실험 결과표를 비웠습니다")
        elif op == "exp_start":
            self._exp_start(cmd)
        elif op == "exp_stop":
            self._exp_stop(cmd)
        elif op == "exp_preview":
            self._exp_preview(cmd)
        elif op == "replay":
            self._replay_start(cmd)
        elif op == "replay_ctl":
            self._replay_ctl(cmd)
        elif op == "follow":
            self._follow(str(cmd["id"]) if cmd.get("on") and cmd.get("id") else None)
        else:
            raise ValueError(f"unknown op {op!r}")

    # --- room edits, with undo --------------------------------------------

    def _edit(self, cmd: dict) -> None:
        """Apply one room edit and record how to take it back.

        Each record holds the edit's own inverse rather than a copy of the
        room: undoing a move puts that one item back and leaves every drop as
        eaten as it is now. Removed items return under their old id, so older
        records that name them still find them.
        """
        sb = self.sandbox
        op = cmd["op"]
        # One drag or one slider pull carries one gesture id from the page.
        gesture = cmd.get("gesture")
        params = {k: v for k, v in cmd.items() if k not in ("op", "kind", "x", "y", "id", "gesture")}
        key = None
        if op == "place":
            placed = sb.place(cmd["kind"], float(cmd["x"]), float(cmd["y"]), **params)
            kind = KIND_LABELS.get(placed["kind"], placed["kind"])
            entry = {"label": f"{kind} 놓기", "undo": [{"op": "remove", "id": placed["id"]}],
                     "redo": [{"op": "place", **sb.item_spec(placed["id"])}]}
            sb._event(f"{kind} 놓음 ({placed['x']:.0f}, {placed['y']:.0f})")
        elif op == "move":
            item_id = int(cmd["id"])
            item = sb.room.items[item_id]
            before = {"op": "move", "id": item_id, "x": item.x, "y": item.y}
            sb.move(item_id, float(cmd["x"]), float(cmd["y"]))
            after = {"op": "move", "id": item_id, "x": item.x, "y": item.y}
            # A drag sends a move every 60 ms; one record per drag.
            key = ("move", item_id)
            entry = {"label": f"{KIND_LABELS.get(item.kind, item.kind)} 옮기기", "undo": [before], "redo": [after]}
        elif op == "update":
            item_id = int(cmd["id"])
            item = sb.room.items[item_id]
            keys = set(params) | ({"left"} if item.kind == "sugar" and "volume" in params else set())
            before = {"op": "update", "id": item_id, **{k: item.params[k] for k in keys if k in item.params}}
            sb.update(item_id, **params)
            after = {"op": "update", "id": item_id, **{k: item.params[k] for k in keys if k in item.params}}
            # A slider sends an update per step of the drag; one record per drag.
            key = ("update", item_id, tuple(sorted(params)))
            entry = {"label": f"{KIND_LABELS.get(item.kind, item.kind)} 값 바꾸기", "undo": [before], "redo": [after]}
        elif op == "remove":
            item_id = int(cmd["id"])
            spec = sb.item_spec(item_id)
            kind = KIND_LABELS.get(spec["kind"], spec["kind"])
            sb.remove(item_id)
            entry = {"label": f"{kind} 지우기", "undo": [{"op": "place", **spec}],
                     "redo": [{"op": "remove", "id": item_id}]}
            sb._event(f"{kind} 지움 ({spec['x']:.0f}, {spec['y']:.0f})")
        else:  # clear, preset: whole-room changes, recorded as whole layouts
            before = [sb.item_spec(i) for i in sb.room.items]
            if op == "clear":
                sb.clear()
                label = "전부 지우기"
                sb._event("방을 비웠습니다")
            else:
                load_preset(sb, cmd["name"])
                label = f"프리셋 {PRESETS[cmd['name']][0]}"
                note = PRESET_NOTES.get(cmd["name"])
                sb._event(f"프리셋: {PRESETS[cmd['name']][0]}" + (f" — {note}" if note else ""))
            after = [sb.item_spec(i) for i in sb.room.items]
            entry = {"label": label, "undo": [{"op": "layout", "items": before}],
                     "redo": [{"op": "layout", "items": after}]}
        # The room is no longer the preset, nor the layout a trial repeats.
        self.preset_name = cmd["name"] if op == "preset" else None
        self._room_edited = True
        # Same gesture, same record. Without a gesture id, commands of the same
        # kind closer than 1.5 s merge -- a fallback only: measured from Python
        # on this machine each POST to localhost took 2.0 s (IPv6 tried first),
        # which split a five-step drag into five records.
        now = time.monotonic()
        top = self._undo[-1] if self._undo else None
        same = key is not None and top is not None and top.get("key") == key and (
            top.get("gesture") == gesture if gesture is not None else now - top["t"] < 1.5)
        if same:
            top["redo"], top["t"] = entry["redo"], now
        else:
            self._undo.append({**entry, "key": key, "t": now, "gesture": gesture})
            del self._undo[:-UNDO_DEPTH]
        self._redo.clear()

    def _fly_move(self, cmd: dict) -> None:
        """Put the fly down where the map says, as one undoable step."""
        sb = self.sandbox
        pos = self.fs.thorax_pos()
        before = {"op": "fly_move", "x": float(pos[0]), "y": float(pos[1]), "yaw": float(self.fs.yaw())}
        yaw = cmd.get("yaw")
        sb.move_fly(float(cmd["x"]), float(cmd["y"]), None if yaw is None else float(yaw))
        pos = self.fs.thorax_pos()
        after = {"op": "fly_move", "x": float(pos[0]), "y": float(pos[1]), "yaw": float(self.fs.yaw())}
        self._undo.append({"label": "초파리 옮기기", "undo": [before], "redo": [after], "key": None,
                           "t": time.monotonic(), "gesture": None})
        del self._undo[:-UNDO_DEPTH]
        self._redo.clear()

    def _apply_actions(self, actions: list[dict]) -> int:
        """Replay recorded actions; returns how many found their item. One the
        fly has since eaten, or the user removed another way, is skipped."""
        sb, done = self.sandbox, 0
        for action in actions:
            op = action["op"]
            if op == "layout":
                sb.clear()
                for spec in action["items"]:
                    sb.place(spec["kind"], spec["x"], spec["y"], item_id=spec["item_id"], **spec["params"])
                done += 1
            elif op == "fly_move":
                sb.move_fly(action["x"], action["y"], action["yaw"])
                done += 1
            elif op == "place":
                if action["item_id"] not in sb.room.items:
                    sb.place(action["kind"], action["x"], action["y"], item_id=action["item_id"], **action["params"])
                    done += 1
            elif action["id"] in sb.room.items:
                if op == "remove":
                    sb.remove(action["id"])
                elif op == "move":
                    sb.move(action["id"], action["x"], action["y"])
                elif op == "update":
                    sb.update(action["id"], **{k: v for k, v in action.items() if k not in ("op", "id")})
                done += 1
        return done

    def _step_history(self, source: list, target: list, direction: str, verb: str) -> None:
        sb = self.sandbox
        if not source:
            sb._event("되돌릴 편집이 없습니다" if direction == "undo" else "다시 할 편집이 없습니다")
            return
        entry = source.pop()
        done = self._apply_actions(entry[direction])
        entry["key"] = None  # never coalesce with a later edit
        target.append(entry)
        self.preset_name = None
        self._room_edited = True
        sb._event(f"{verb}: {entry['label']}" + ("" if done else " (그 물건은 이미 없습니다)"))

    # --- background A/B experiments, and replaying their trials ---------------

    @staticmethod
    def _experiment_dir(exp_id: str) -> Path:
        """An experiment's directory by name, refusing anything that is not one."""
        name = Path(str(exp_id)).name
        path = EXPERIMENTS_DIR / name
        if not name or name != str(exp_id) or not (path / "protocol.json").exists():
            raise ValueError(f"실험 기록 {exp_id!r}을(를) 찾을 수 없습니다")
        return path

    def _records(self, directory: Path) -> list[dict]:
        """`read_records`, parsing only files that changed since the last call:
        the page polls every 3 s, and a finished experiment's records run to
        megabytes of paths."""
        cache = self._records_cache.setdefault(directory.name, {})
        records = []
        for path in sorted((directory / "records").glob("*.jsonl")):
            try:
                stat = path.stat()
                key = (stat.st_size, stat.st_mtime_ns)
                if cache.get(path.name, (None,))[0] != key:
                    rows = []
                    for line in path.read_text(encoding="utf-8").splitlines():
                        try:
                            rows.append(json.loads(line))
                        except ValueError:
                            pass  # a line still being written
                    cache[path.name] = (key, rows)
            except OSError:
                continue
            records.extend(cache[path.name][1])
        return sorted(records, key=lambda r: (r["condition"], r["fly"], r["seq"]))

    def _count_records(self, directory: Path) -> int:
        """Finished trials in an experiment, counting only changed files' lines."""
        total = 0
        for path in (directory / "records").glob("*.jsonl"):
            try:
                stat = path.stat()
                key = (str(path), stat.st_size, stat.st_mtime_ns)
                if key not in self._counts_cache:
                    with path.open(encoding="utf-8") as f:
                        self._counts_cache[key] = sum(1 for line in f if line.strip())
                total += self._counts_cache[key]
            except OSError:
                continue
        return total

    def _exp_start(self, cmd: dict) -> None:
        """Write the protocol and hand it to 17_run_experiment.py in its own
        process: the flies run on the other cores while this one keeps drawing."""
        key = str(cmd.get("set"))
        if key not in EXPERIMENT_SETS:
            raise ValueError(f"unknown experiment set {key!r}")
        s = EXPERIMENT_SETS[key]
        # A double click once started the same set twice a second apart, and
        # the two runs shared the cores for the rest of the experiment.
        if key == self._last_exp_start[0] and time.monotonic() - self._last_exp_start[1] < 10.0:
            self.sandbox._event("같은 실험을 방금 시작했습니다. 한 번만 돌립니다")
            return
        self._last_exp_start = (key, time.monotonic())
        flies = int(np.clip(int(cmd.get("flies") or s.flies), 1, 50))
        prediction = cmd.get("prediction") if cmd.get("prediction") in ("A<B", "A>B", "same") else s.expect
        # A new seed every run, so running a set again tests new flies.
        protocol = protocol_for(key, flies=flies, prediction=prediction, seed=int(time.time()) % 1_000_000)
        directory = new_experiment_dir(tag=key)
        write_json(directory / "protocol.json", protocol.to_dict())
        write_json(directory / "status.json", {"state": "starting", "done_trials": 0,
                                               "total_trials": protocol.total_trials(),
                                               "started": time.strftime("%Y-%m-%d %H:%M:%S")})
        busy = any(proc.poll() is None for proc in self._batches.values())
        with (directory / "log.txt").open("w", encoding="utf-8") as log:
            self._batches[directory.name] = subprocess.Popen(
                [sys.executable, str(Path(__file__).with_name("17_run_experiment.py")), "--dir", str(directory)],
                cwd=str(_bootstrap.ROOT), stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        minutes = estimate_wall_seconds(protocol, POOL_WORKERS, 1.0 / POOL_SPEED_PER_FLY) / 60.0
        self.sandbox._event(f"배경 실험 시작: {s.title} · 조건마다 파리 {flies}마리 · 약 {max(1, round(minutes))}분"
                            + (" (다른 배경 실험과 CPU를 나눠 쓰므로 더 걸립니다)" if busy else ""))

    def _exp_stop(self, cmd: dict) -> None:
        directory = self._experiment_dir(cmd["id"])
        (directory / "STOP").touch()
        self.sandbox._event("배경 실험을 멈춥니다. 돌고 있던 시행은 중간에 멈춘 채로 기록됩니다")

    def _exp_preview(self, cmd: dict) -> None:
        """Furnish the live room as condition A or B's first trial, as one undoable edit."""
        key = str(cmd.get("set"))
        condition = EXPERIMENT_SETS[key].conditions[int(cmd.get("condition", 0))]
        step, sb = condition.steps[0], self.sandbox
        before = [sb.item_spec(i) for i in sb.room.items]
        furnish(sb, step.room, protocol_for(key), condition)
        after = [sb.item_spec(i) for i in sb.room.items]
        self._undo.append({"label": f"{condition.label} 방 보기", "undo": [{"op": "layout", "items": before}],
                           "redo": [{"op": "layout", "items": after}], "key": None,
                           "t": time.monotonic(), "gesture": None})
        del self._undo[:-UNDO_DEPTH]
        self._redo.clear()
        changes = condition.changes()
        self.preset_name = None if changes else step.room
        self._room_edited = True
        sb._event(f"{condition.label}의 첫 방: {room_label(step.room)}" + (f" ({', '.join(changes)})" if changes else ""))

    def _replay_start(self, cmd: dict) -> None:
        """Show a recorded trial: its room, and the body pose by pose, on this
        viewer's own model. The live fly's full state and room are kept aside
        and put back by `_replay_stop`; nothing live steps meanwhile."""
        directory = self._experiment_dir(cmd["id"])
        c, fly, seq = int(cmd["condition"]), int(cmd["fly"]), int(cmd["seq"])
        record = next((r for r in self._records(directory)
                       if r["condition"] == c and r["fly"] == fly and r["seq"] == seq), None)
        if record is None or not record.get("replay"):
            raise ValueError("이 시행은 다시 보기 기록이 없습니다")
        data = np.load(directory / record["replay"])
        model, d = self.fs.sim.mj_model, self.fs.sim.mj_data
        if int(data["nq"]) != model.nq:
            raise ValueError("다른 모델로 기록된 시행이라 다시 볼 수 없습니다")
        sb = self.sandbox
        if self.replay is None:
            spec = mujoco.mjtState.mjSTATE_INTEGRATION
            state = np.empty(mujoco.mj_stateSize(model, spec))
            mujoco.mj_getState(model, d, state, spec)
            saved = {"state": state, "items": [sb.item_spec(i) for i in sb.room.items],
                     "paused": self.paused, "feeding_on": sb._feeding_on}
        else:
            saved = self.replay["saved"]
        sb.clear()
        for item in record["items"]:
            sb.place(item["kind"], item["x"], item["y"], item_id=item["item_id"], **item["params"])
        protocol = Protocol.from_dict(read_json(directory / "protocol.json"))
        s = EXPERIMENT_SETS.get(protocol.set_key)
        role = ROLE_LABELS.get(record["role"], record["role"])
        self.replay = {
            "id": directory.name, "condition": c, "fly": fly, "seq": seq, "record": record,
            "qpos": data["qpos"], "hz": float(data["hz"]), "tel": data["telemetry"],
            "sugar_ids": [int(i) for i in data["sugar_ids"]], "radii": data["sugar_radii"],
            "duration": len(data["qpos"]) / float(data["hz"]),
            "t": 0.0, "playing": True, "speed": 1.0, "wall": time.perf_counter(), "saved": saved,
            "label": f"{s.title if s else protocol.title} · {record['condition_label']} · 파리 {fly} · "
                     f"{seq}번째 시행({role}) · {record['room_label']}",
        }
        self.paused = True
        self._replay_apply()

    def _replay_apply(self) -> None:
        r, sb = self.replay, self.sandbox
        model, d = self.fs.sim.mj_model, self.fs.sim.mj_data
        d.qpos[:] = r["qpos"][min(len(r["qpos"]) - 1, int(r["t"] * r["hz"]))]
        d.qvel[:] = 0.0
        if r["sugar_ids"] and len(r["radii"]):
            # Drops as big as they were then; one eaten up is gone, and back if
            # the replay is wound back before that.
            row = r["radii"][min(len(r["radii"]) - 1, int(r["t"] * 10))]
            specs = {it["item_id"]: it for it in r["record"]["items"]}
            for item_id, radius in zip(r["sugar_ids"], row):
                if radius < 0:
                    if item_id in sb.room.items:
                        sb.room.remove(item_id)
                    continue
                if item_id not in sb.room.items and item_id in specs:
                    spec = specs[item_id]
                    sb.place(spec["kind"], spec["x"], spec["y"], item_id=item_id, **spec["params"])
                if item_id in sb.room.items:
                    sb.room.update(item_id, radius=float(radius))
        mujoco.mj_forward(model, d)

    def _replay_tick(self) -> None:
        r = self.replay
        now = time.perf_counter()
        if r["playing"]:
            r["t"] = min(r["duration"], r["t"] + (now - r["wall"]) * r["speed"])
            if r["t"] >= r["duration"]:
                r["playing"] = False
        r["wall"] = now
        self._replay_apply()

    def _replay_ctl(self, cmd: dict) -> None:
        r = self.replay
        if r is None:
            return
        action = cmd.get("action")
        if action == "stop":
            self._replay_stop()
            return
        if action == "toggle":
            action = "pause" if r["playing"] else "play"
        if action == "play":
            if r["t"] >= r["duration"]:
                r["t"] = 0.0
            r["playing"] = True
        elif action == "pause":
            r["playing"] = False
        elif action == "speed":
            r["speed"] = float(np.clip(float(cmd["value"]), 0.1, 8.0))
        elif action == "seek":
            r["t"] = float(np.clip(float(cmd["value"]), 0.0, r["duration"]))
        r["wall"] = time.perf_counter()

    def _replay_stop(self) -> None:
        r, sb = self.replay, self.sandbox
        if r is None:
            return
        self.replay = None
        model, d = self.fs.sim.mj_model, self.fs.sim.mj_data
        mujoco.mj_setState(model, d, r["saved"]["state"], mujoco.mjtState.mjSTATE_INTEGRATION)
        sb.clear()
        for item in r["saved"]["items"]:
            sb.place(item["kind"], item["x"], item["y"], item_id=item["item_id"], **item["params"])
        sb._feeding_on = r["saved"]["feeding_on"]
        mujoco.mj_forward(model, d)
        self.paused = r["saved"]["paused"]
        self._reset_pacer = True
        sb._event("다시 보기를 마치고 실시간 파리로 돌아왔습니다")

    def _follow(self, exp_id: str | None) -> None:
        """Replay each new trial of `exp_id` as it is recorded, starting with the
        newest already there."""
        self.follow_id = exp_id
        self._followed.clear()
        if exp_id is None:
            self.sandbox._event("따라 보기를 껐습니다")
            return
        records = [r for r in self._records(self._experiment_dir(exp_id)) if r.get("replay")]
        self._followed.update((r["condition"], r["fly"], r["seq"]) for r in records)
        self.sandbox._event("새 시행이 기록될 때마다 자동으로 다시 봅니다")
        if records:
            newest = max(records, key=lambda r: (r.get("finished", 0.0), r["seq"]))
            self._replay_start({"id": exp_id, "condition": newest["condition"], "fly": newest["fly"],
                                "seq": newest["seq"]})

    def _follow_tick(self) -> None:
        now = time.monotonic()
        if self.follow_id is None or now - self._follow_checked < 2.0:
            return
        self._follow_checked = now
        if self.replay is not None and self.replay["playing"]:
            return
        fresh = [r for r in self._records(self._experiment_dir(self.follow_id))
                 if r.get("replay") and (r["condition"], r["fly"], r["seq"]) not in self._followed]
        if not fresh:
            return
        newest = max(fresh, key=lambda r: (r.get("finished", 0.0), r["seq"]))
        self._followed.add((newest["condition"], newest["fly"], newest["seq"]))
        self._replay_start({"id": self.follow_id, "condition": newest["condition"], "fly": newest["fly"],
                            "seq": newest["seq"]})

    def _replay_status(self) -> dict | None:
        r = self.replay
        if r is None:
            return None
        t = r["t"]
        telemetry = None
        tel = r["tel"]
        if len(tel):
            row = tel[int(np.clip(np.searchsorted(tel[:, 0], t), 0, len(tel) - 1))]
            telemetry = {
                "time": round(t, 2), "mode": MODES[int(row[1])] if int(row[1]) < len(MODES) else "explore",
                "hunger": round(float(row[2]), 3),
                "dopamine": {"punish": round(float(row[3]), 3), "sweet": round(float(row[4]), 3),
                             "nutrient": round(float(row[5]), 3)},
                "odour_valence": {o: round(float(v), 3) for o, v in zip(ODOURS, row[6:9])},
                "colour_valence": {"blue": round(float(row[9]), 3), "green": round(float(row[10]), 3)},
                "legs_on_sugar": 0, "legs_on_shock": 0, "sugar_under": None, "counts": {}, "reflex": [],
            }
        return {
            "id": r["id"], "condition": r["condition"], "fly": r["fly"], "seq": r["seq"],
            "label": r["label"], "t": round(t, 2), "duration": round(r["duration"], 2),
            "playing": r["playing"], "speed": r["speed"],
            "path": [p for p in r["record"]["path"] if p[0] <= t + 1e-6],
            "telemetry": telemetry,
            "events": [{"t": e[0], "text": e[1]} for e in r["record"]["events"] if e[0] <= t + 1e-6],
        }

    def sets_payload(self) -> dict:
        """The experiment sets for the lab: what A and B are, their rooms drawn
        from above, the hypotheses to choose from, what is held constant, and
        what a time estimate needs. A room drawn once is shared by every step
        and set that furnishes it the same way."""
        if self._sets_payload is not None:
            return self._sets_payload
        rooms: dict[str, str] = {}
        ids: dict[str, str] = {}

        def thumb(room: str, condition) -> str:
            items = room_items(room, condition)
            key = json.dumps(items, sort_keys=True)
            if key not in ids:
                ids[key] = f"r{len(ids)}"
                rooms[ids[key]] = svg_room(items, size=120)
            return ids[key]

        groups = []
        for group, keys in SET_GROUPS:
            items = []
            for key in keys:
                s = EXPERIMENT_SETS[key]
                protocol = protocol_for(key)
                metric = METRICS_BY_KEY[s.metric]
                conditions = [{
                    "label": c.label, "changes": c.changes(), "same_fly": c.same_fly, "hunger": c.hunger,
                    "steps": [{"role": st.role, "role_ko": ROLE_LABELS.get(st.role, st.role),
                               "room_label": room_label(st.room), "seconds_label": duration_ko(st.seconds),
                               "repeats": st.repeats, "rest_label": duration_ko(st.rest_after) if st.rest_after else "",
                               "gap_label": duration_ko(st.gap) if st.gap else "", "thumb": thumb(st.room, c)}
                              for st in c.steps],
                } for c in protocol.conditions]
                items.append({
                    "key": key, "group": group, "title": s.title, "question": s.question, "a": s.a, "b": s.b,
                    "changed": s.changed, "metric": metric.column, "metric_key": s.metric,
                    "meaning": measure_words(protocol),
                    "latency": metric.latency, "phase": PHASE_LABELS.get(s.phase, s.phase), "expect": s.expect,
                    "basis": s.basis, "flies": s.flies, "choices": prediction_choices(s),
                    "held": controlled_variables(protocol, flies=False), "conditions": conditions,
                    "seconds_per_fly": max(sum(st.seconds * st.repeats for st in c.steps)
                                           for c in protocol.conditions),
                })
            groups.append({"group": group, "sets": items})
        self._sets_payload = {"groups": groups, "rooms": rooms, "workers": POOL_WORKERS,
                              "speed": POOL_SPEED_PER_FLY}
        return self._sets_payload

    def _exp_status(self, directory: Path, protocol: Protocol) -> dict:
        """State, progress and time left of one experiment directory."""
        status = read_json(directory / "status.json", {}) or {}
        done, total = self._count_records(directory), protocol.total_trials()
        state = status.get("state", "")
        proc = self._batches.get(directory.name)
        alive = (proc is not None and proc.poll() is None) or (
            status.get("pid") is not None and psutil.pid_exists(int(status["pid"])))
        if state in ("starting", "running", "reporting") and not alive:
            state = "done" if done >= total else "stopped"
        eta = None
        if state in ("starting", "running", "reporting"):
            # The whole run's estimate less the time already spent. A rate from
            # trials done so far misleads: flies finish in waves, one per
            # worker pool's worth, so at the end of the first wave most of the
            # trials are done and most of the time is still ahead.
            try:
                started = time.mktime(time.strptime(status.get("started", ""), "%Y-%m-%d %H:%M:%S"))
                elapsed = max(0.0, time.time() - started)
            except (TypeError, ValueError):
                elapsed = 0.0
            eta = max(0.0, estimate_wall_seconds(protocol, POOL_WORKERS, 1.0 / POOL_SPEED_PER_FLY) - elapsed)
        return {"state": state, "done": done, "total": total, "eta_s": None if eta is None else round(eta),
                "stop_requested": (directory / "STOP").exists()}

    def _verdict(self, directory: Path, protocol: Protocol, done: int) -> dict | None:
        """The verdict so far, for the list; worked out again only when a trial finishes."""
        if not done:
            return None
        key = (directory.name, done)
        if key not in self._verdicts:
            result = evaluate(protocol, self._records(directory), protocol.prediction or protocol.expect)
            self._verdicts[key] = {"code": result["verdict"], "text": VERDICT_TEXT[result["verdict"]]}
        return self._verdicts[key]

    def experiments_list(self) -> list[dict]:
        """Experiment directories, newest first, with progress and the verdict so far."""
        out = []
        if not EXPERIMENTS_DIR.exists():
            return out
        for directory in sorted((p for p in EXPERIMENTS_DIR.iterdir() if p.is_dir()), reverse=True)[:60]:
            raw = read_json(directory / "protocol.json")
            if not raw:
                continue
            try:
                protocol = Protocol.from_dict(raw)
            except (TypeError, KeyError):
                continue
            status = self._exp_status(directory, protocol)
            verdict = self._verdict(directory, protocol, status["done"])
            s = EXPERIMENT_SETS.get(protocol.set_key)
            out.append({"id": directory.name, "title": s.title if s else protocol.title, "set": protocol.set_key,
                        "flies": protocol.flies, "created": protocol.created, "prediction": protocol.prediction,
                        "verdict": verdict["text"] if verdict else "", "verdict_code": verdict["code"] if verdict else "",
                        **status})
        return out

    def experiment_detail(self, exp_id: str) -> dict:
        """One experiment for the lab's result view: its trial grid on the deciding
        measure, every fly's score, and the verdict so far."""
        directory = self._experiment_dir(exp_id)
        protocol = Protocol.from_dict(read_json(directory / "protocol.json"))
        records = self._records(directory)
        s = EXPERIMENT_SETS.get(protocol.set_key)
        key = protocol.metric
        metric = METRICS_BY_KEY[key]
        grid = []
        for c, condition in enumerate(protocol.conditions):
            roles = [st.role for st in condition.steps for _ in range(max(1, int(st.repeats)))]
            flies = [[None] * len(roles) for _ in range(protocol.flies)]
            for r in records:
                if r["condition"] == c and 1 <= r["fly"] <= protocol.flies and 1 <= r["seq"] <= len(roles):
                    value = metric_value(r, key)
                    flies[r["fly"] - 1][r["seq"] - 1] = {
                        "v": None if value is None else round(value, 3),
                        # A latency that never ended, counted as the trial's length.
                        "censored": value is not None and r["metrics"].get(key) is None,
                        "replay": bool(r.get("replay")), "stopped": bool(r["stopped"])}
            grid.append({"label": condition.label, "roles": roles, "flies": flies})
        summary = None
        prediction = protocol.prediction or protocol.expect
        if records:
            result = evaluate(protocol, records, prediction)
            summary = {"verdict": VERDICT_TEXT[result["verdict"]], "code": result["verdict"],
                       **{k: result.get(k) for k in ("a", "b", "pairs", "a_higher", "a_lower", "ties", "p_sign",
                                                     "direction", "difference", "consistent", "tolerance")},
                       "scores": {side: {str(fly): round(v, 4) for fly, v in result["scores"][side].items()}
                                  for side in ("a", "b")}}
        return {"id": directory.name, "title": s.title if s else protocol.title, "set": protocol.set_key,
                "question": s.question if s else protocol.title, "a": s.a if s else "A", "b": s.b if s else "B",
                "why": s.why if s else "", "basis": s.basis if s else "",
                "metric": metric.column, "metric_key": key, "meaning": measure_words(protocol),
                "sides": pi_sides(protocol), "latency": metric.latency,
                "phase": PHASE_LABELS.get(protocol.phase, protocol.phase),
                "prediction": prediction, "prediction_text": prediction_text(s, prediction) if s else prediction,
                "expect": protocol.expect, "flies": protocol.flies, "created": protocol.created,
                "replay_flies": protocol.replay_flies, "grid": grid, "summary": summary,
                **self._exp_status(directory, protocol), "done": len(records)}

    # --- trials: fixed-length runs whose results collect in a table -------

    def _room_snapshot(self) -> list[tuple]:
        keep = {"sugar": ("sugar", "molar", "volume"), "shock": ("volts", "half"),
                "patch": ("colour", "half"), "odour": ("odour", "strength"), "obstacle": ("hx", "hy")}
        return [(it.kind, it.x, it.y, {k: it.params[k] for k in keep[it.kind] if k in it.params})
                for it in self.sandbox.room.items.values()]

    def _sb_trial_start(self, cmd: dict) -> None:
        sb = self.sandbox
        preset = cmd.get("preset") or ""
        if preset:
            if preset not in PRESETS:
                raise ValueError(f"unknown preset {preset!r}")
            load_preset(sb, preset)
            self.preset_name = preset
            self._trial_layout = self._room_snapshot()
        elif self._trial_layout is None or self._room_edited:
            self._trial_layout = self._room_snapshot()
        # Every trial starts from the full layout: drops the last fly ate, or
        # finished and removed, come back. That re-creates every item, so edit
        # history from before no longer names anything; it is dropped.
        dropped_history = bool(self._undo or self._redo)
        self._undo.clear()
        self._redo.clear()
        sb.clear()
        for kind, x, y, item_params in self._trial_layout:
            sb.place(kind, x, y, **item_params)
        self._room_edited = False
        heading = ""
        if cmd.get("new_fly", True):
            yaw = float(self._yaw_rng.uniform(0.0, 2 * np.pi)) if cmd.get("random_heading", True) else None
            sb.reset_fly(yaw=yaw)
            heading = "" if yaw is None else round(float(np.degrees(yaw)) % 360)
            self._clear_sandbox_traces()
            self._rewound()
        else:
            sb.reset_counts()
        sb.hunger = float(np.clip(cmd.get("hunger", sb.config.hunger_start), 0.0, 1.0))
        self._trial_count += 1
        self.sb_trial = {
            "number": self._trial_count,
            "label": str(cmd.get("label") or "")[:40],
            "layout": PRESETS[self.preset_name][0] if self.preset_name else "직접 꾸민 방",
            "hunger": sb.hunger,
            "seconds": float(np.clip(cmd.get("seconds", 60.0), 5.0, 1800.0)),
            "t0": sb.time,
            "new_fly": bool(cmd.get("new_fly", True)),
            "heading": heading,
        }
        self.paused = False
        sb._event(f"실험 {self._trial_count} 시작: {self.sb_trial['label'] or '이름 없음'} · "
                  f"{self.sb_trial['seconds']:.0f}초")
        if dropped_history:
            # After the new fly, whose reset starts a fresh event log.
            sb._event("실험을 시작해 되돌리기 기록을 비웠습니다")

    def _sb_trial_finish(self, stopped: bool = False) -> None:
        if self.sb_trial is None:
            return
        sb, trial = self.sandbox, self.sb_trial
        elapsed = sb.time - trial["t0"]
        row = {
            "번호": trial["number"],
            "조건": trial["label"],
            "방": trial["layout"],
            "새 파리": "예" if trial["new_fly"] else "아니오",
            "시작 방향(°)": trial["heading"],
            "시작 배고픔": round(trial["hunger"], 2),
            "시간(초)": round(elapsed, 2),
            "끝남": "중간에 멈춤" if stopped else "정해진 시간",
            **sb.trial_summary(),
        }
        self.trial_rows.append(row)
        self._append_trial_csv(row)
        self.sb_trial = None
        self.paused = True
        sb.refresh_telemetry()
        sb._event(f"실험 {row['번호']} 끝: {elapsed:.1f}초 · 결과표에 기록했습니다")

    def _append_trial_csv(self, row: dict) -> None:
        """Every row also goes to out/sandbox/trials.csv, so results survive a
        closed tab or a restarted viewer. UTF-8 with BOM: Excel reads Korean."""
        path = _bootstrap.OUT / "sandbox" / "trials.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["기록 시각", *row.keys()]
        new = not path.exists()
        try:
            if not new:
                with path.open(encoding="utf-8-sig") as f:
                    header = next(csv.reader(f), [])
                if header != fields:
                    # The columns changed (a new version added measures):
                    # appending would put values under the wrong headings.
                    path.rename(path.with_name(f"trials_until_{time.strftime('%Y%m%d_%H%M%S')}.csv"))
                    new = True
            with path.open("a", newline="", encoding="utf-8-sig" if new else "utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                if new:
                    writer.writeheader()
                writer.writerow({"기록 시각": time.strftime("%Y-%m-%d %H:%M:%S"), **row})
        except OSError as error:
            self.sandbox._event(f"CSV 파일에 쓰지 못했습니다: {error}")

    def _clear_sandbox_traces(self) -> None:
        for key in ("punish", "sweet", "nutrient", "hunger"):
            trace = self.traces.get(key)
            if trace is not None:
                trace.extend([0.0] * trace.maxlen)

    def _rewound(self) -> None:
        """A new fly restarts sim.time. The pacer's anchor would then sit in the
        future and never wait again, and the speed readout would go negative."""
        self._reset_pacer = True
        self._speed_samples.clear()

    def trail_since(self, since: int) -> dict:
        """Trail points from index `since` on, as base64 float32 x/y pairs:
        8 bytes a point instead of about 30 as JSON."""
        if self.sandbox is None:
            return {"epoch": -1, "start": 0, "xy": ""}
        epoch, start, xy = self.sandbox.trail.since(since)
        return {"epoch": epoch, "start": start, "xy": base64.b64encode(xy.tobytes()).decode("ascii")}

    def _speed_factor(self) -> float | None:
        """Simulated seconds per wall second over the last few seconds of play."""
        if self.paused or len(self._speed_samples) < 2:
            return None
        (w0, s0), (w1, s1) = self._speed_samples[0], self._speed_samples[-1]
        return round((s1 - s0) / (w1 - w0), 2) if w1 - w0 > 0.5 else None

    def _nearest_sugar(self) -> float | None:
        xy = self.fs.thorax_pos()[:2]
        sugars = self.sandbox.room.of_kind("sugar")
        if not sugars:
            return None
        return round(min(max(0.0, float(np.hypot(xy[0] - s.x, xy[1] - s.y)) - s.half) for s in sugars), 1)

    def _draw_shock_outlines(self, scene) -> None:
        """Dashed red outlines of the shock zones, added to this render's scene
        only. The plates themselves are fully transparent, and the fly's eyes
        render a different scene, so this can never become a cue it sees."""
        dash, gap, radius = 2.5, 1.5, 0.22
        rgba = np.array([1.0, 0.27, 0.12, 1.0], dtype=np.float32)
        z = 0.08
        for item in self.sandbox.room.of_kind("shock"):
            hx, hy = item.extent
            corners = [(item.x - hx, item.y - hy), (item.x + hx, item.y - hy),
                       (item.x + hx, item.y + hy), (item.x - hx, item.y + hy)]
            for (x0, y0), (x1, y1) in zip(corners, corners[1:] + corners[:1]):
                length = float(np.hypot(x1 - x0, y1 - y0))
                for k in range(max(1, int(length // (dash + gap)) + 1)):
                    a = k * (dash + gap)
                    if a >= length:
                        break
                    b = min(length, a + dash)
                    if scene.ngeom >= scene.maxgeom:
                        return
                    geom = scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                                        np.zeros(3), np.eye(3).ravel(), rgba)
                    ux, uy = (x1 - x0) / length, (y1 - y0) / length
                    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                                         np.array([x0 + ux * a, y0 + uy * a, z]),
                                         np.array([x0 + ux * b, y0 + uy * b, z]))
                    geom.emission = 0.5
                    scene.ngeom += 1

    def _sandbox_status(self) -> dict:
        sb = self.sandbox
        pos = self.fs.thorax_pos()
        trial = None
        if self.sb_trial is not None:
            trial = {k: self.sb_trial[k] for k in ("number", "label", "layout", "seconds")}
            trial["elapsed"] = round(sb.time - self.sb_trial["t0"], 1)
        out = {
            "sandbox": True,
            "room_half": sb.room.half,
            "items": sb.room.as_list(),
            "fly": {"x": round(float(pos[0]), 2), "y": round(float(pos[1]), 2),
                    "yaw": round(self.fs.yaw(), 3)},
            # The newest 5 s of the trail; the page fetches older points
            # once from /trail (a day of trail is 36 MB of JSON per poll).
            **dict(zip(("trail_epoch", "trail_total", "trail"), sb.trail.recent(50))),
            "telemetry": asdict(sb.telemetry),
            "events": list(sb.events),
            "palette": {
                "sugars": list(SUGARS),
                "odours": list(ODOURS),
                "presets": {name: label for name, (label, _) in PRESETS.items()},
                "preset_groups": [[group, names] for group, names in PRESET_GROUPS],
            },
            "preset": self.preset_name,
            "sugar_depletes": sb.sugar_depletes,
            "sb_trial": trial,
            "trial_rows": self.trial_rows,
            "speed_factor": self._speed_factor(),
            "undo": self._undo[-1]["label"] if self._undo else None,
            "redo": self._redo[-1]["label"] if self._redo else None,
            "follow": self.follow_id,
        }
        replay = self._replay_status()
        out["replay"] = replay
        if replay is not None:
            # The dashboard shows the replayed fly: its hunger, learned values
            # and events at this moment of its trial.
            if replay["telemetry"] is not None:
                out["telemetry"] = replay["telemetry"]
            out["events"] = replay["events"]
        return out

    # --- conditioning ---------------------------------------------------

    def _conditioning_step(self) -> None:
        """One 100 Hz action step of the running trial, starting trials as needed."""
        if self._trial_iter is None and not self._start_trial():
            return
        try:
            state, concentration = next(self._trial_iter)
        except StopIteration as finished:
            self._finish_trial(finished.value)
            return

        self.intensity = float(concentration.sum())
        self.smelling = bool(state.smelling)
        # The reinforcing compartment: punishment depresses approach, sugar
        # depresses avoid. Omission dopamine is not the US and is not flagged.
        target = "avoid" if self.exp.config.reinforcer == "reward" else "approach"
        self.dopamine = state.dan.get(target, 0.0) > 0.0
        if self.colour:
            self.under = self.fs.floor.colour_at(self.fs.thorax_pos()[:2])
            self.cond_odour = self.under
            self.last_visual = state.visual
            if self.trial_kind == "test" and self.under in self.time_on:
                self.time_on[self.under] += self.exp.dt
        else:
            # Which odour the fly is mostly in. A training trial parks the other
            # source 500 mm away, so this is exact there; on a test trial it is
            # whichever source the fly is nearer, which is what the raster's
            # colour should follow.
            self.cond_odour = (
                ODOUR_NAMES[int(np.argmax(concentration))] if self.smelling else None
            )

        count = self._cond_counter
        self._cond_counter += 1
        if count % self.args.neuro_decim == 0:
            # 100 indices per action step is 100 Hz of JSON carrying no more
            # information than the eye can take at 25.
            if self.colour and state.visual is not None:
                self.vkc_active = [[int(i) for i in state.visual.active(e)] for e in (0, 1)]
                self.vpn = state.visual.vpn
                scale = self.exp.mb.visual_scale
                eyes = self.exp.mb.eye_valences(state.visual) / scale
                left, right = self.walker.descending_signal
                self.traces["eye_left"].append(float(eyes[0]))
                self.traces["eye_right"].append(float(eyes[1]))
                # +1 = the hardest left turn the signal band allows.
                self.traces["turn"].append(float(right - left) / (SIGNAL_HIGH - SIGNAL_LOW))
            else:
                self.kc_active = [int(i) for i in state.olfactory.active]
        if count % self._cond_every == 0:
            self._sample_conditioning(state)

    def _sample_conditioning(self, state) -> None:
        """One sample of the mushroom body, at COND_SAMPLE_HZ of simulated time.

        The MBON lines are the live readout, so they fall to zero whenever the
        fly is out of odour. That is the circuit being honest and not a gap in
        the plot -- no Kenyon cells, no MBON drive -- and it is what makes the
        result readable: the height of each burst is the approach MBON's answer
        to this smell today, and that envelope is the learning curve.

        Valence is the read-only probe instead. Live valence is zero between
        encounters for the same reason, so a live drive trace would snap back to
        `INNATE_VALENCE` after every burst and cross zero dozens of times
        without meaning it. The probe crosses once, when the memory is deep
        enough, which is the moment the panel is for.
        """
        valence = self.exp.valence_of(self.focus) if self.focus else 0.0
        if self.colour:
            # Everything in units of the visual code's scale, so 1.0 is a
            # resting MBON and -1 a colour whose synapses are all silent.
            scale = self.exp.mb.visual_scale
            other = next(n for n in self.exp.stimulus_names if n != self.focus)
            self.traces["mbon_approach"].append(float(state.mbon["approach"]) / scale)
            self.traces["mbon_avoid"].append(float(state.mbon["avoid"]) / scale)
            self.traces["valence"].append(valence / scale)
            self.traces["valence_other"].append(self.exp.valence_of(other) / scale)
            self.traces["drive"].append(0.0)
        else:
            self.traces["mbon_approach"].append(float(state.mbon["approach"]))
            self.traces["mbon_avoid"].append(float(state.mbon["avoid"]))
            self.traces["valence"].append(valence)
            self.traces["drive"].append(INNATE_VALENCE + valence)
        self.dan_marks.append(1 if self.dopamine else 0)
        self.phase_marks.append(1 if self._mark_phase else 0)
        self._mark_phase = False

    def _start_trial(self) -> bool:
        """Begin the next trial in the plan. False once the session is over."""
        if self.plan_index >= len(self.plan):
            if not self.session_done:
                self._close_block()
                self.session_done = True
                summary = " | ".join(
                    f"{b['key']} {b['preference']:+.3f}" for b in self.blocks
                )
                print(f"session complete: {summary}")
            return False
        key, kind, odour, focus = self.plan[self.plan_index]
        if key != self.block_key:
            self._close_block()
            self.block_key = key
            self._block_prefs = []
            self._mark_phase = True
        # An "expose" trial presents an odour without punishing it; showing it
        # as the punished odour in the header would be exactly backwards.
        self.trial_kind, self.focus = kind, focus
        self.trial_cs = odour if kind == "train" else None
        self.trial_no = self.plan_index + 1
        if self.colour:
            self.time_on = {name: 0.0 for name in self.exp.stimulus_names}
        self._trial_iter = self.exp.trial_steps(kind, odour, block=key)
        self._reset_pacer = True  # the trial is about to rewind sim.time
        return True

    def _finish_trial(self, result: TrialResult) -> None:
        self._trial_iter = None
        self.plan_index += 1
        if result.preference is not None:
            self._block_prefs.append(result.preference)
        preference = "    -" if result.preference is None else f"{result.preference:+.3f}"
        print(
            f"  trial {result.index:>3} {result.kind:<5} "
            f"cs+={result.cs_plus or '-'} pref={preference} valence "
            + " ".join(f"{name}={value:+.4f}" for name, value in result.valence.items())
            + f" dopamine={result.dopamine_seconds:.2f}s"
        )

    def _close_block(self) -> None:
        """Record a finished test block's preference index. Training blocks score
        nothing: they are reinforced, so any preference in them is the dopamine
        talking, not the memory."""
        if self.block_key and self._block_prefs:
            self.blocks.append(
                {
                    "key": self.block_key,
                    "label": BLOCK_LABELS[self.block_key],
                    "preference": round(float(np.mean(self._block_prefs)), 3),
                    "n": len(self._block_prefs),
                }
            )

    def _restart_session(self) -> None:
        """What reset means in conditioning mode: a new fly, the same protocol.

        With a memory loaded, "new fly" means the same remembered fly again --
        the memory is restored from its file, since test trials under
        extinction change it and resetting to naive would lose what was loaded.
        """
        if self.memory_path is not None:
            self.exp.load_memory(self.memory_path)
        else:
            self.exp.reset_memory()
        self.plan_index = 0
        self.trial_no = 0
        self.block_key = None
        self.blocks = []
        self._block_prefs = []
        self._trial_iter = None
        self._cond_counter = 0
        self.session_done = False
        for key in ("mbon_approach", "mbon_avoid", "valence", "drive", "valence_other"):
            if key in self.traces:
                self.traces[key].extend([0.0] * COND_TRACE_LEN)
        self.dan_marks.extend([0] * COND_TRACE_LEN)
        self.phase_marks.extend([0] * COND_TRACE_LEN)
        print("session restarted with a fresh fly")

    def _protocol_blocks(self) -> list[dict]:
        """The session's blocks with trial ranges and, once finished, results.

        Drives the page's protocol timeline. The ranges come from the same
        `session_plan` the trials are run from, so the timeline cannot describe
        a protocol other than the one executing.
        """
        blocks: list[dict] = []
        for number, (key, kind, odour, _focus) in enumerate(self.plan, start=1):
            if blocks and blocks[-1]["key"] == key:
                blocks[-1]["last"] = number
                continue
            # A training block opens with its CS+ (session_plan puts the
            # punished trial first even when differential), so the block's
            # first odour is the one it punishes.
            blocks.append(
                {"key": key, "label": BLOCK_LABELS[key], "kind": kind,
                 "odour": odour, "first": number, "last": number}
            )
        finished = {b["key"]: b["preference"] for b in self.blocks}
        for block in blocks:
            if block["key"] in finished:
                block["preference"] = finished[block["key"]]
        return blocks

    def _conditioning_status(self) -> dict:
        """Header numbers for the session. Read-only probes, so calling it at the
        frame rate cannot disturb the experiment."""
        running = (
            round(float(np.mean(self._block_prefs)), 3) if self._block_prefs else None
        )
        # Colour valences are shown in units of full depression of a colour's
        # visual code; odour valences are already in MBON units.
        scale = self.exp.mb.visual_scale if self.colour else 1.0
        memory = None
        if self.memory_path is not None:
            config = self.memory_info.get("config") or {}
            memory = {
                "name": self.args.memory,
                "note": self.memory_info.get("note", ""),
                "trials": self.memory_info.get("trials", 0),
                "extinction": config.get("extinction"),
                # What the memory held when it was filed. The live values are
                # "valence" below, and under extinction they drift during tests.
                "filed": {
                    odour: round(s["valence"] / scale, 3)
                    for odour, s in (self.memory_info.get("summary") or {}).items()
                },
            }
        return {
            "memory": memory,
            "protocol": self._protocol_blocks(),
            "conditioning": True,
            "trial": self.trial_no,
            "trials_total": len(self.plan),
            "trial_kind": self.trial_kind,
            "block": self.block_key,
            "block_label": BLOCK_LABELS.get(self.block_key, ""),
            "cs_plus": self.trial_cs,
            "focus": self.focus,
            "intensity": round(self.intensity, 5),
            "smelling": self.smelling,
            "dopamine": self.dopamine,
            "valence": {
                n: round(self.exp.valence_of(n) / scale, 3) for n in self.exp.stimulus_names
            },
            "preference": running,
            "blocks": list(self.blocks),
            "done": self.session_done,
            "modality": "colour" if self.colour else "odour",
            "stimuli": list(self.exp.stimulus_names),
            "reinforcer": self.exp.config.reinforcer,
            "under": self.under,
            "time_on": {k: round(v, 1) for k, v in self.time_on.items()},
        }

    # --- simulation thread ---------------------------------------------

    def run(self):
        try:
            self.build()
            # Render with a plain MuJoCo renderer so the camera can be switched
            # per frame; flygym's Renderer is built for writing video files.
            self.renderer = mujoco.Renderer(
                self.fs.sim.mj_model, height=self.args.height, width=self.args.width
            )
        except BaseException as e:  # surface it instead of hanging the server
            self.failure = e
            self.ready.set()
            raise
        self.ready.set()

        pacer = RealtimePacer(self.fs.sim.timestep, self.speed)
        steps_per_frame = max(1, int(self.speed / self.args.fps / self.fs.sim.timestep))
        last_speed = self.speed

        while not self._stop.is_set():
            if self.sandbox_mode:
                # Edits apply even while paused: furnishing a frozen room is
                # the easiest way to set up a scene. Paused, nothing steps, so
                # geom positions would stay where they were drawn last and a
                # new preset never appeared (persona test); recompute them.
                if self._drain_commands() and self.paused and self.replay is None:
                    mujoco.mj_forward(self.fs.sim.mj_model, self.fs.sim.mj_data)
                try:
                    self._follow_tick()
                except (KeyError, ValueError, OSError) as error:
                    self.follow_id = None
                    self.sandbox._event(f"따라 보기를 멈췄습니다: {error}")
            if self.reset_requested:
                self.reset_requested = False
                if self.sandbox_mode:
                    self._replay_stop()
                    self.sandbox.reset_fly()
                    self._clear_sandbox_traces()
                    self._speed_samples.clear()
                elif self.conditioning:
                    self._restart_session()
                elif self.policy is not None:
                    self._obs, _ = self.env.reset()
                else:
                    self.walker.reset()
                pacer.reset()
            if self.speed != last_speed:
                last_speed = self.speed
                pacer.speed = self.speed
                pacer.reset()
                steps_per_frame = max(
                    1, int(self.speed / self.args.fps / self.fs.sim.timestep)
                )

            if self.sandbox_mode and self.replay is not None:
                # A replay sets poses; nothing steps. Frames at the page's rate.
                frame_start = time.perf_counter()
                self._replay_tick()
                self._render()
                time.sleep(max(0.0, 1.0 / self.args.fps - (time.perf_counter() - frame_start)))
                pacer.reset()
                continue
            if self.paused:
                time.sleep(0.05)
                pacer.reset()
                if self.sandbox_mode:
                    self._speed_samples.clear()
            elif self.sandbox_mode:
                for _ in range(max(1, steps_per_frame // self.sandbox.steps_per_action)):
                    self.sandbox.step()
                    self._sandbox_counter += 1
                    if self._sandbox_counter % 10 == 0:
                        dopamine = self.sandbox.telemetry.dopamine
                        for key in ("punish", "sweet", "nutrient"):
                            self.traces[key].append(float(dopamine.get(key, 0.0)))
                        self.traces["hunger"].append(float(self.sandbox.hunger))
                    # Checked every step, so a 60 s trial stops at 60.00 s.
                    if self.sb_trial is not None and (self.sandbox.time - self.sb_trial["t0"]
                                                      >= self.sb_trial["seconds"] - 1e-9):
                        self._sb_trial_finish()
                        break
                self._speed_samples.append((time.perf_counter(), self.fs.sim.time))
            elif self.conditioning:
                if self.session_done:
                    time.sleep(0.05)  # hold the last frame; reset starts a new fly
                    pacer.reset()
                else:
                    # The mushroom body reads at config.action_hz; take as many
                    # of its steps as fit in one displayed frame.
                    steps = max(1, steps_per_frame // self.exp.steps_per_action)
                    for _ in range(steps):
                        self._conditioning_step()
            elif self.policy is not None:
                # The policy decides at env.action_hz; take as many of its
                # steps as fit in one displayed frame.
                for _ in range(max(1, steps_per_frame // self.env.steps_per_action)):
                    self._policy_step()
            else:
                self.walker.descending_signal = np.clip(
                    [self.drive - self.turn, self.drive + self.turn],
                    SIGNAL_LOW,
                    SIGNAL_HIGH,
                )
                for _ in range(steps_per_frame):
                    self.walker.physics_step()

            if self._reset_pacer:
                # A trial rewound sim.time, so the pacer's anchor is in the
                # future and every wait() would return instantly from here on.
                self._reset_pacer = False
                pacer.reset()

            self._render()
            if not self.paused:
                pacer.wait(self.fs.sim.time)

        self.renderer.close()
        self.fs.close()

    def _render(self):
        self.renderer.update_scene(self.fs.sim.mj_data, self.cam)
        colours = getattr(self, "_display_colours", None)
        if colours:
            scene = self.renderer.scene
            for i in range(scene.ngeom):
                geom = scene.geoms[i]
                if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid in colours:
                    geom.rgba[:] = colours[geom.objid]
        if self.sandbox_mode:
            self._draw_shock_outlines(self.renderer.scene)
        rgb = self.renderer.render()
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=80)

        pos = self.fs.thorax_pos()
        left, right = self.walker.descending_signal
        self.status = {
            "time": round(self.fs.sim.time, 2),
            "x": round(float(pos[0]), 1),
            "y": round(float(pos[1]), 1),
            "speed_mms": round(float(np.linalg.norm(self.fs.body_velocity()[:2])), 1),
            "left": round(float(left), 2),
            "right": round(float(right), 2),
            "drive": round(self.drive, 2),
            "turn": round(self.turn, 2),
            "playback": self.speed,
            "upright": round(self.fs.upright(), 2),
            "legs_down": int(self.fs.leg_contacts().sum()),
            "view": self.view,
            "distance": round(float(self.cam.distance), 1),
            "azimuth": round(float(self.cam.azimuth)),
            "paused": self.paused,
            "terrain": self.args.terrain,
            "driver": (
                "sandbox"
                if self.sandbox_mode
                else "conditioning"
                if self.conditioning
                else self.args.policy
                if self.policy is not None
                else "keyboard"
            ),
        }
        if self.conditioning:
            self.status.update(self._conditioning_status())
        if self.sandbox_mode:
            self.status.update(self._sandbox_status())
        if self.policy is not None:
            recent = list(self.outcomes)
            self.status.update(
                training_steps=self._policy_steps,
                episodes=self.episodes,
                reloads=self.reloads,
                recent_n=len(recent),
                recent_found=sum(1 for o in recent if o in ("found", "goal")),
                following=self.follow,
            )
        if self.sandbox_mode:
            # All odours summed: the first dimension alone is vinegar, and the
            # first source's position was a parked slot 900 mm away.
            intensities = self.fs.odor()
            self.status["odor"] = round(float(intensities.sum(axis=1).mean()), 5)
            self.status["odor_asym"] = round(float(OdorField.asymmetry(
                intensities.sum(axis=1, keepdims=True))[0]), 4)
            nearest = self._nearest_sugar()
            if nearest is not None:
                self.status["goal_dist"] = nearest
            self.status["touching"] = bool(self.sandbox.room.touching_solid())
        elif self.policy is None and not self.conditioning and self.fs.odor_field is not None:
            intensities = self.fs.odor()
            self.status["odor"] = round(float(intensities[:, 0].mean()), 5)
            self.status["odor_asym"] = round(
                float(OdorField.asymmetry(intensities)[0]), 4
            )
            self.status["goal"] = [
                round(float(v), 1) for v in self.fs.odor_field.sources[0].pos[:2]
            ]
            self.status["touching"] = bool(self.fs.touching_pillar())
        if self.policy is None and self.retina is not None:
            features = self.retina.extract(
                self.fs.sim.get_ommatidia_readouts(self.fs.name)
            )
            self.status["vis_asym"] = round(float(features[4]), 4)
            self.status["vis_total"] = round(float(features[5]), 4)

        if self.policy is not None:
            if self.meta.get("task") == "forage":
                source = self.env.odor_field.sources[0].pos
                self.status["goal"] = [round(float(v), 1) for v in source[:2]]
                self.status["goal_dist"] = round(self._distance_to_goal(), 1)
                self.status["odor"] = round(
                    float(self.env.odor_field.read(self.fs.sim)[:, 0].mean()), 4
                )
                self.status["touching"] = bool(self.fs.touching_pillar())
            else:
                self.status["goal"] = [round(float(v), 1) for v in self.env.goal]
                self.status["bearing"] = round(
                    float(np.rad2deg(self.env.goal_bearing()))
                )
        with self._new_frame:
            self._frame = buf.getvalue()
            self._new_frame.notify_all()

        self._sample_body()
        self._render_eyes()
        self.neuro_snapshot = self._build_neuro_payload()

    def _sample_body(self) -> None:
        """Sample the signals that do not need the policy, at the frame rate."""
        if self.fs.odor_field is not None:
            intensities = self.fs.odor()
            if self.conditioning:
                # Two odour dimensions here, and a training trial parks the one
                # it is not using 500 mm away, so plotting dimension 0 alone
                # would flatline through every B trial. Summed, the panel reads
                # as "how much smell is on each side", which is the quantity the
                # steering law is actually given.
                intensities = intensities.sum(axis=1, keepdims=True)
            left, right = OdorField.left_right(intensities)[:, 0]
            self.traces["odor_l"].append(float(left))
            self.traces["odor_r"].append(float(right))

    def _render_eyes(self) -> None:
        """Turn the last ommatidia readouts into a left|right hex mosaic.

        Reuses whatever the environment already rendered for its own
        observation -- an eye render costs 25 ms, so paying for a second one
        just to draw it would halve the frame rate.
        """
        retina = self.fs.sim.retina
        if self.sandbox_mode:
            readouts = self.sandbox._readouts
            if readouts is None:
                return
            own = photoreceptors(readouts)
            pale = eye_layout()[0]
            panels = []
            for eye in own:
                rgb = np.zeros((eye.size, 3))
                rgb[~pale, 0], rgb[~pale, 1] = 0.25 * eye[~pale], eye[~pale]
                rgb[pale, 1], rgb[pale, 2] = 0.35 * eye[pale], eye[pale]
                panels.append(retina.hex_pxls_to_human_readable(np.clip(rgb, 0, 1),
                                                                color_8bit=True))
            image = Image.fromarray(np.concatenate(panels, axis=1).astype(np.uint8))
        elif self.colour:
            # The colour task already rendered the eyes for the mushroom body;
            # draw what it read, each ommatidium in the colour its type reads --
            # green for yellow-type, blue for pale-type. Red is read by none.
            visual = self.last_visual
            if visual is None:
                return
            pale = self.exp.mb.visual.pale
            panels = []
            for eye in visual.own:
                rgb = np.zeros((eye.size, 3))
                rgb[~pale, 0], rgb[~pale, 1] = 0.25 * eye[~pale], eye[~pale]
                rgb[pale, 1], rgb[pale, 2] = 0.35 * eye[pale], eye[pale]
                panels.append(retina.hex_pxls_to_human_readable(np.clip(rgb, 0, 1),
                                                                color_8bit=True))
            image = Image.fromarray(np.concatenate(panels, axis=1).astype(np.uint8))
        else:
            readouts = getattr(self.env, "last_readouts", None) if self.env else None
            if readouts is None and self.retina is not None:
                readouts = self.fs.sim.get_ommatidia_readouts(self.fs.name)
            if readouts is None:
                return
            panels = [
                retina.hex_pxls_to_human_readable(eye.max(axis=1), color_8bit=True)
                for eye in readouts
            ]
            mosaic = np.concatenate(panels, axis=1)
            image = Image.fromarray(mosaic.astype(np.uint8)).convert("RGB")
        image = image.resize((self.args.eye_width, self.args.eye_width * image.height
                              // image.width), Image.NEAREST)
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=70)
        with self._eye_lock:
            self._eye_frame = buf.getvalue()
            self._eye_lock.notify_all()

    def next_frame(self, timeout: float = 2.0) -> bytes | None:
        with self._new_frame:
            self._new_frame.wait(timeout)
            return self._frame

    def next_eye_frame(self, timeout: float = 2.0) -> bytes | None:
        with self._eye_lock:
            self._eye_lock.wait(timeout)
            return self._eye_frame

    def _build_neuro_payload(self) -> dict:
        """Everything the neural dashboard draws, as plain JSON.

        Assembled on the simulation thread and stored; HTTP threads only ever
        read the finished dict, so they never touch `mj_data` mid-step.
        """
        payload = {
            "traces": {k: [round(v, 4) for v in q] for k, q in self.traces.items()},
            "cpg_phase": [round(float(p) % (2 * np.pi), 3) for p in self.walker.cpg_phases],
            "cpg_magnitude": [round(float(m), 3) for m in self.walker.cpg_magnitudes],
            "descending": [round(float(v), 3) for v in self.walker.descending_signal],
            "sensor_labels": list(SENSOR_LABELS),
            "has_brain": self.policy is not None,
            "sandbox": self.sandbox_mode,
        }
        if self.policy is not None:
            payload["episode_marks"] = list(self.episode_marks)
            payload["channels"] = [label for label, _, _ in ATTRIBUTION_CHANNELS]
            payload["attribution"] = {
                label: [round(v, 4) for v in q]
                for label, q in self.attr_traces.items()
            }
            payload["attr_history"] = [
                {"steps": steps, "means": {k: round(v, 4) for k, v in means.items()}}
                for steps, means in sorted(self.attr_history.values())
            ]
            payload["probe_ready"] = self._probe_states is not None
        if self.conditioning:
            payload["conditioning"] = True
            payload["kc_n"] = int(self.exp.mb.front_end.n_kc)
            payload["kc_active"] = self.kc_active
            payload["kc_ref"] = self.kc_ref
            payload["kc_odour"] = self.cond_odour
            payload["dan_marks"] = list(self.dan_marks)
            payload["phase_marks"] = list(self.phase_marks)
            payload["innate"] = INNATE_VALENCE
            payload["focus"] = self.focus
            payload["modality"] = "colour" if self.colour else "odour"
            if self.colour:
                visual = self.exp.mb.visual
                payload["vkc_n"] = int(visual.n_kc)
                payload["vkc_active"] = self.vkc_active
                payload["stimuli"] = list(self.exp.stimulus_names)
                payload["vpn_labels"] = visual.vpn_labels
                payload["vpn"] = (
                    None if self.vpn is None
                    else [[round(float(v), 3) for v in eye] for eye in self.vpn]
                )
                payload["under"] = self.under
                payload["gain"] = self.exp.config.visual_gain
        if self.fs.odor_field is not None:
            payload["odor_sensors"] = [
                round(float(v), 6) for v in self.fs.odor().ravel()[:4]
            ]
        readouts = getattr(self.env, "last_readouts", None) if self.env else None
        if readouts is not None:
            payload["vision"] = [
                round(float(v), 4) for v in self.env.retina.extract(readouts)
            ]
            payload["eye_brightness"] = round(float(readouts.max(axis=2).mean()), 3)
        return payload

    def stop(self):
        self._stop.set()


PAGE = """<!doctype html>
<meta charset="utf-8"><title>NeuroMechFly — live</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 :root { color-scheme: dark; --bd:#2a2a2a; --panel:#161616; --dim:#8a8a8a; }
 * { box-sizing:border-box; }
 html, body { height:100%; }
 body { margin:0; background:#0e0e0e; color:#eee; overflow:hidden;
        font:13px/1.45 ui-sans-serif,system-ui,'Segoe UI',sans-serif;
        display:grid; gap:8px; padding:8px;
        grid-template-rows:auto minmax(0,1fr) auto;
        grid-template-columns:minmax(0,1fr); }

 /* ---- header: title, learning status, live stats, all on one line ---- */
 header { display:flex; align-items:center; gap:10px; flex-wrap:wrap;
          padding:0 2px; }
 h1 { font-size:15px; font-weight:600; margin:0; white-space:nowrap; }
 h1 small { color:var(--dim); font-weight:400; }
 .learn { background:#1b2430; border:1px solid #2e3d4f; border-radius:8px;
          padding:4px 12px; color:#c9d4e0; white-space:nowrap; }
 .learn b { color:#7fb2f0; font-variant-numeric:tabular-nums; }
 .learn .sep { color:#44525f; margin:0 7px; }
 .stats { display:flex; gap:12px; flex-wrap:wrap; margin-left:auto;
          font-variant-numeric:tabular-nums; color:#cfcfcf; }
 .stats i { color:var(--dim); font-style:normal; margin-right:4px; }

 /* ---- main: 3D view on the left, instrument rail on the right ---- */
 main { display:grid; gap:8px; min-height:0;
        grid-template-columns:minmax(0,1fr) 380px; }
 .stage { position:relative; background:#000; border-radius:10px;
          overflow:hidden; border:1px solid var(--bd); min-height:0; }
 /* cover, not contain: the render is 16:9 and the stage is whatever the window
    leaves, so contain letterboxed a small picture inside a large black frame.
    The camera tracks the fly, so cropping trims the edges of the arena and
    never the animal. */
 #view { position:absolute; inset:0; width:100%; height:100%; object-fit:cover;
         display:block; cursor:grab; touch-action:none;
         -webkit-user-drag:none; user-select:none; -webkit-user-select:none; }
 #view.dragging { cursor:grabbing; }
 /* Over the video, bottom-left, so the colours are explained where they are
    seen. pointer-events:none keeps drag-to-orbit working through it. */
 .legend { position:absolute; left:10px; bottom:10px; max-width:min(460px, 70%);
           background:rgba(10,10,10,0.72); border:1px solid #333; border-radius:8px;
           padding:6px 10px; font-size:12px; color:#ddd; pointer-events:none;
           display:flex; flex-wrap:wrap; gap:4px 14px; }
 .legend i { display:inline-block; width:11px; height:11px; border-radius:50%;
             margin-right:5px; vertical-align:-1px; }
 .legend small { flex-basis:100%; color:#9a9a9a; font-size:11px; line-height:1.4; }
 /* The explanation sits under the video rather than in the rail. It is prose,
    and prose gets shorter as it gets wider: in the 380 px rail it took 372 px
    of height and left the Kenyon-cell raster 65 px to draw 2000 cells in. */
 .stagecol { display:flex; flex-direction:column; gap:8px; min-height:0; min-width:0; }
 .stagecol > .stage { flex:1 1 auto; }
 .stagecol > #explaincard { flex:0 0 auto; }
 .rail { display:flex; flex-direction:column; gap:8px; min-height:0; }
 .rail > .card { flex:1 1 0; }
 .rail > #kccard { flex-grow:1.4; }
 .rail > #vkccard { flex-grow:1.1; }

 .card { background:var(--panel); border:1px solid var(--bd); border-radius:10px;
         padding:6px 9px; display:flex; flex-direction:column; min-height:0; }
 /* A canvas's width/height attributes are its intrinsic size, and fitCanvases()
    writes the displayed size back into them. Left in normal flow that is a
    feedback loop: any moment a card is taller, the canvas keeps that height as
    its new intrinsic size and the card can never shrink again -- measured, the
    MBON card grew to 465 px and slid up under the video while the Kenyon-cell
    raster was squeezed to 13 px. Positioned absolutely inside .plot, the
    canvas takes whatever space the layout gives and feeds nothing back. */
 .plot { position:relative; flex:1 1 auto; min-height:60px; }
 .plot > canvas, .plot > img { position:absolute; inset:0; width:100%; height:100%;
                               display:block; object-fit:contain; }
 /* Captions wrap in full. They carry how to read each graph, and an ellipsis
    cut exactly that part off. */
 .cap { color:var(--dim); font-size:11px; line-height:1.4; margin-bottom:4px; }
 .cap small { color:#7a7a7a; font-size:11px; }
 #eyeview { image-rendering:pixelated; background:#000; }

 /* ---- experiment explanation (conditioning mode) ---- */
 .intro { color:#bdbdbd; font-size:12px; margin:0 0 6px; }
 .intro > b:first-child { color:#eee; }
 /* The protocol as a row of blocks, left to right in the order they run. */
 .protocol { list-style:none; margin:0 0 6px; padding:0; font-size:12px;
             display:flex; flex-wrap:wrap; gap:6px; font-variant-numeric:tabular-nums; }
 .protocol li { flex:1 1 140px; padding:4px 8px; border-radius:6px;
                border:1px solid #2c2c2c; border-top:3px solid #2c2c2c; color:#8f8f8f; }
 .protocol li.now { border-color:#6b5230; border-top-color:#e0a24d;
                    background:#1e1b16; color:#eee; }
 .protocol li.past { color:#c4c4c4; border-top-color:#4a4a4a; }
 .protocol .name { display:block; font-weight:600; }
 .protocol .meta { display:flex; justify-content:space-between; gap:6px;
                   font-size:11px; color:#7a7a7a; }
 .protocol .pi { color:#7fb2f0; }
 .now-text { color:#d8d8d8; font-size:12px; line-height:1.55; margin:0; }
 .now-text b { color:#e0a24d; font-weight:600; }

 /* ---- footer: wide traces + one compact control row ---- */
 footer { display:grid; gap:8px; grid-template-columns:minmax(0,1.35fr) minmax(0,1fr); }
 footer > .card { height:clamp(170px, 26vh, 260px); }
 /* The colour task has three footer graphs: memory, valence, and steering. */
 footer.three { grid-template-columns:repeat(3, minmax(0,1fr)); }
 .foot-controls { grid-column:1 / -1; display:flex; gap:6px; align-items:center;
                  flex-wrap:wrap; }
 button { background:#242424; color:#eee; border:1px solid #3a3a3a; border-radius:7px;
          padding:5px 11px; font:inherit; cursor:pointer; }
 button:hover { background:#333; border-color:#555; }
 button.on { background:#3d6ea5; border-color:#5b8fd0; }
 .pad { display:flex; gap:5px; }
 .lbl { color:var(--dim); font-size:11px; }
 .gap { width:8px; }
 input[type=range] { width:110px; accent-color:#3d6ea5; }
 .help { color:#6e6e6e; font-size:11px; margin-left:auto; cursor:help; }
 /* ---- sandbox ---- */
 /* The map is the control surface here, so it takes the rail: the eye mosaic
    is not shown in this mode, and the learned-value bars keep a fixed strip. */
 main.sandbox { grid-template-columns:minmax(0,1fr) 540px; }
 .rail > #mapcard { flex:1 1 auto; }
 .rail > #learncard { flex:0 0 190px; }
 .palette { display:flex; flex-direction:column; gap:5px; margin-bottom:6px; font-size:12px; }
 .palette .tools { display:flex; flex-wrap:wrap; gap:4px; }
 .palette .tools button { padding:3px 9px; }
 .palette .opts { display:none; align-items:center; gap:6px; flex-wrap:wrap; color:#bdbdbd; }
 .palette .opts.on { display:flex; }
 .palette select, .palette input[type=range] { background:#1d1d1d; color:#eee;
   border:1px solid #3a3a3a; border-radius:5px; }
 .palette input[type=range] { width:96px; }
 .palette .row { display:flex; align-items:center; gap:6px; flex-wrap:wrap; }
 .palette .hint { color:#8a8a8a; font-size:11px; }
 #c_map { cursor:crosshair; touch-action:none; }
 .log { list-style:none; margin:0; padding:0; overflow:auto; font-size:12px;
        font-variant-numeric:tabular-nums; flex:1 1 auto; min-height:0; }
 .log li { padding:2px 0; border-bottom:1px solid #202020; color:#d0d0d0; }
 .log li b { color:#8a8a8a; font-weight:400; margin-right:6px; }
 .kv { display:grid; grid-template-columns:auto 1fr; gap:3px 10px; font-size:12px;
       font-variant-numeric:tabular-nums; }
 .kv i { color:#8a8a8a; font-style:normal; }
 .hungerrow { display:flex; align-items:center; gap:8px; margin:4px 0 8px; font-size:12px; }
 .hungerrow input { flex:1 1 auto; accent-color:#e0a24d; }
 .tabs { display:flex; gap:4px; margin-bottom:6px; }
 .tabs button { padding:2px 12px; font-size:12px; }
 .trialrow { display:flex; flex-wrap:wrap; align-items:center; gap:6px; font-size:12px; color:#bdbdbd; margin-bottom:6px; }
 .trialrow input[type=text], .trialrow select { background:#1d1d1d; color:#eee; border:1px solid #3a3a3a;
   border-radius:5px; padding:2px 5px; font:inherit; }
 .trialrow input[type=text] { width:96px; }
 .trialrow input[type=range] { width:80px; }
 .trialrow label { display:flex; align-items:center; gap:3px; }
 .trialrow button:disabled { opacity:0.4; cursor:default; }
 #t_status { color:#e0a24d; }
 .trialtable { max-height:150px; overflow:auto; border:1px solid #262626; border-radius:6px; }
 .trialtable table { border-collapse:collapse; font-size:11px; font-variant-numeric:tabular-nums; white-space:nowrap; }
 .trialtable th { position:sticky; top:0; background:#1a1a1a; color:#9a9a9a; font-weight:400;
   text-align:right; padding:3px 6px; border-bottom:1px solid #333; }
 .trialtable td { color:#ddd; text-align:right; padding:2px 6px; border-bottom:1px solid #202020; }
 .trialtable th:nth-child(-n+3), .trialtable td:nth-child(-n+3) { text-align:left; }
 .helpinline { color:#8fb4dc; cursor:help; margin-left:6px; }
 .mapctl { position:absolute; left:6px; top:6px; z-index:2; display:flex; align-items:center; gap:5px;
   font-size:11px; color:#bbb; background:rgba(13,13,13,0.82); border:1px solid #2c2c2c; border-radius:6px; padding:2px 6px; }
 .mapctl select { background:#1d1d1d; color:#eee; border:1px solid #3a3a3a; border-radius:4px; font-size:11px; }
 .mapctl input[type=range] { width:170px; }
 .mapctl b { color:#ddd; font-weight:400; min-width:40px; }
 .mapctl button { padding:1px 8px; font-size:11px; border-radius:5px; }
 /* The sandbox's learned values join the footer, giving the map the whole rail:
    in the rail at 1366x768 the map was left its 60 px minimum. */
 footer.four { grid-template-columns:repeat(4, minmax(0,1fr)); }
 /* Fixed heights, so selecting something never moves the map under the
    pointer: a second click on the same spot has to land on the same spot. */
 .palette .tools .undo { margin-left:auto; display:flex; gap:4px; }
 .palette .tools .undo button:disabled { opacity:0.35; cursor:default; }
 .palette .optsbox { height:54px; overflow:hidden; display:flex; flex-direction:column; justify-content:center; }
 /* The preset row never wraps either: measured, a selection's name squeezed
    the clear button onto two lines and pushed the map down 17 px. */
 .palette .row { flex-wrap:nowrap; overflow:hidden; }
 .palette .row > * { flex-shrink:0; white-space:nowrap; }
 #p_preset { max-width:160px; }
 .selbar { display:inline-flex; visibility:hidden; align-items:center; gap:6px; font-size:12px; color:#e8e8e8;
   background:#232a33; border:1px solid #34506e; border-radius:6px; padding:1px 4px 1px 8px; }
 .selbar.on { visibility:visible; }
 #lr_grid td.rp { cursor:pointer; }
 #lr_grid td.rp:hover { outline:1px solid #8fb4dc; }
 #lr_grid td.nr { color:#666; }
 #lr_grid th.cond, #lr_grid td.cond { text-align:left; }
 .trialrow input[type=number] { background:#1d1d1d; color:#eee; border:1px solid #3a3a3a; border-radius:5px; padding:2px 4px; width:52px; font:inherit; }
 .verdictline { color:#e0a24d; }
 .replaybar { position:absolute; left:8px; right:8px; top:8px; z-index:4; display:flex; flex-wrap:wrap; gap:4px 8px;
   align-items:center; background:rgba(10,12,18,0.88); border:1px solid #3d6ea5; border-radius:8px; padding:4px 10px;
   font-size:12px; color:#ddd; }
 .replaybar b { color:#8fb4dc; }
 .replaybar button { padding:1px 9px; font-size:12px; }
 .replaybar select { background:#1d1d1d; color:#eee; border:1px solid #3a3a3a; border-radius:4px; font-size:12px; }
 #r_seek { flex:1 1 140px; min-width:100px; }
 /* The video keeps at least 240 px whatever tab is open below it: the A/B pane
    once took 393 of the column's 440 px at 1366x768 and left the video 39. The
    card shrinks instead, and its grid scrolls. */
 main.sandbox .stagecol > .stage { min-height:240px; }
 main.sandbox #explaincard { flex:0 1 auto; min-height:0; overflow:auto; }
 /* ---- A/B lab ----
    A dialog over the whole window, one step at a time. The first A/B pane put a
    set picker, a record picker, room previews and a results grid into the
    192 px under the video, with the prediction offered as "A가 더 작다", and
    the user found it too hard to use. */
 .lab { position:fixed; inset:0; z-index:50; background:rgba(0,0,0,0.62); display:flex; padding:12px; }
 .labpanel { margin:0 auto; width:100%; max-width:1480px; background:#121212; border:1px solid #2e2e2e;
   border-radius:12px; display:flex; flex-direction:column; min-height:0; box-shadow:0 12px 48px rgba(0,0,0,0.6); }
 .labtop { display:flex; flex-wrap:wrap; align-items:center; gap:8px 14px; padding:10px 16px; border-bottom:1px solid #262626; }
 .labtitle { font-size:17px; color:#f2f2f2; }
 .labsteps { list-style:none; display:flex; flex-wrap:wrap; gap:6px; margin:0; padding:0; }
 .labsteps li { font-size:13px; padding:4px 12px; border-radius:999px; border:1px solid #333; color:#6f6f6f; }
 .labsteps li.done { color:#c4c4c4; border-color:#4a4a4a; cursor:pointer; }
 .labsteps li.done:hover { border-color:#5b8fd0; }
 .labsteps li.now { color:#fff; background:#3d6ea5; border-color:#5b8fd0; }
 .labspacer { flex:1 1 auto; }
 .labbody { flex:1 1 auto; min-height:0; overflow:auto; padding:14px 18px 28px; font-size:14px; line-height:1.55; color:#d6d6d6; }
 .labbody h2 { font-size:22px; line-height:1.3; margin:4px 0; color:#f2f2f2; }
 .labbody h3 { font-size:15px; margin:18px 0 8px; color:#e6e6e6; }
 .labbody h3 small { font-weight:400; margin-left:6px; }
 .labbody .lbl { font-size:12px; }
 .labintro { margin:0 0 4px; color:#bdbdbd; max-width:980px; }
 .labcard { background:#181818; border:1px solid #2a2a2a; border-radius:10px; padding:12px 14px; min-width:0; }
 .labcardtitle { font-weight:600; color:#eee; margin-bottom:8px; }
 .labcardtitle small { font-weight:400; font-size:12px; color:#8a8a8a; margin-left:6px; }
 .labstep { margin-top:10px; }
 .linkbtn { background:none; border:none; color:#8fb4dc; padding:2px 0; border-radius:0; }
 .linkbtn:hover { background:none; text-decoration:underline; }
 button.primary { background:#2f7d4f; border-color:#3f9a64; color:#fff; font-weight:600; }
 button.primary:hover { background:#389159; border-color:#4fb277; }
 button.primary:disabled { background:#223a2c; border-color:#2e4a39; color:#7a9a86; cursor:default; }
 button.big { font-size:16px; padding:9px 24px; border-radius:9px; }
 .ab-a { color:#e0874f; }
 .ab-b { color:#6f9fe8; }
 i.tagA, i.tagB { font-style:normal; font-weight:700; font-size:12px; display:inline-block; min-width:18px; text-align:center;
   border-radius:5px; padding:0 5px; margin-right:5px; color:#111; }
 i.tagA { background:#e0874f; }
 i.tagB { background:#6f9fe8; }
 .progress { display:inline-block; vertical-align:middle; width:120px; height:7px; background:#2a2a2a; border-radius:4px; overflow:hidden; }
 .progress > i { display:block; height:100%; background:#5b8fd0; }
 .badge { display:inline-block; font-size:12px; line-height:18px; padding:0 8px; border-radius:999px; border:1px solid #444;
   color:#ccc; white-space:nowrap; }
 .badge.starting, .badge.running, .badge.reporting { border-color:#5b8fd0; color:#9cc3f0; }
 .badge.done { border-color:#3f9a64; color:#8fd3a8; }
 .badge.stopped, .badge.failed { border-color:#8a5a3a; color:#e0a27a; }
 .labexps { display:grid; grid-template-columns:repeat(auto-fill, minmax(250px, 1fr)); gap:8px; }
 .expcard { display:flex; flex-direction:column; gap:4px; text-align:left; padding:10px 12px; border-radius:10px; background:#1b1b1b; }
 .expcard:hover, .qcard:hover { background:#1f2530; border-color:#5b8fd0; }
 .expcard b { color:#eee; font-weight:600; }
 .expcard small { color:#9a9a9a; font-size:12px; }
 .expcard .progress { width:100%; }
 .expcardtop { display:flex; justify-content:space-between; align-items:center; gap:6px; }
 .vmini.good { color:#8fe0ae; }
 .vmini.lean { color:#f0c070; }
 .vmini.bad { color:#f09a8a; }
 .labgroup { margin-bottom:12px; }
 .labgrouptitle { color:#e0a24d; font-weight:600; margin:4px 0 6px; }
 .qgrid { display:grid; grid-template-columns:repeat(auto-fill, minmax(320px, 1fr)); gap:8px; }
 .qcard { display:flex; gap:10px; align-items:flex-start; text-align:left; padding:8px; border-radius:10px; background:#1a1a1a; }
 .qthumb { flex:0 0 76px; width:76px; border-radius:6px; overflow:hidden; }
 .qthumb svg, .thumb svg { display:block; width:100%; height:auto; }
 .qtext { display:flex; flex-direction:column; gap:2px; min-width:0; font-size:13px; color:#c8c8c8; }
 .qtext b { color:#f2f2f2; font-size:14px; }
 .qtext small { color:#8a8a8a; font-size:12px; }
 .qtext small.tried { color:#8fd3a8; }
 .sethead p { margin:2px 0 0; color:#bdbdbd; }
 .setcols { display:grid; grid-template-columns:repeat(2, minmax(0,1fr)); gap:10px; margin:12px 0 10px; }
 .cond.a { border-top:3px solid #e0874f; }
 .cond.b { border-top:3px solid #6f9fe8; }
 .seq { display:flex; flex-wrap:wrap; align-items:flex-start; gap:6px; margin:4px 0 6px; }
 .seqstep { width:132px; font-size:12px; color:#c4c4c4; text-align:center; line-height:1.35; }
 .seqstep .thumb { border-radius:6px; overflow:hidden; margin-bottom:4px; }
 .seqarrow { align-self:center; color:#6a6a6a; font-size:18px; }
 .seqrest { align-self:center; font-size:12px; color:#e0a24d; }
 .condnote { font-size:12px; color:#9a9a9a; margin:2px 0 4px; }
 .condnote b { color:#e0a24d; font-weight:600; }
 .vars { display:grid; grid-template-columns:auto minmax(0,1fr); gap:8px 16px; margin:0; }
 .vars dt { color:#e8e8e8; font-weight:600; white-space:nowrap; }
 .vars dt .lbl { display:block; font-weight:400; }
 .vars dd { margin:0; }
 .choice { display:flex; align-items:center; gap:10px; padding:9px 12px; margin:6px 0; border:1px solid #333;
   border-radius:9px; background:#1c1c1c; cursor:pointer; color:#e6e6e6; }
 .choice:hover { border-color:#5b8fd0; }
 .choice.on { border-color:#e0a24d; background:#2a2418; }
 .choice input { accent-color:#e0a24d; width:16px; height:16px; margin:0; }
 .runrow { display:flex; flex-wrap:wrap; align-items:center; gap:8px 12px; }
 .runrow input[type=range] { width:240px; }
 .warn { color:#e0a24d; font-size:13px; margin:4px 0; }
 .labmore { margin-top:10px; }
 .labmore summary { cursor:pointer; color:#9ab8d8; }
 .labmore p { max-width:980px; }
 .reshead { display:flex; flex-wrap:wrap; justify-content:space-between; align-items:flex-end; gap:8px 20px; margin:6px 0 10px; }
 .resheadside { display:flex; flex-direction:column; align-items:flex-end; gap:6px; }
 .resheadbtns { display:flex; flex-wrap:wrap; gap:6px; }
 .resgrid { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1.2fr); gap:10px; align-items:start; }
 .myhyp { color:#eaeaea; }
 .verdictbox { font-size:19px; font-weight:700; line-height:1.35; margin:10px 0 6px; padding:9px 12px; border-radius:8px;
   background:#232323; color:#d0d0d0; }
 .verdictbox.good { background:#173524; color:#8fe0ae; }
 .verdictbox.lean { background:#2f2715; color:#f0c070; }
 .verdictbox.bad { background:#3a1c1a; color:#f09a8a; }
 .labchart svg { display:block; width:100%; max-width:560px; height:auto; }
 .labvideo { position:relative; aspect-ratio:16 / 9; background:#000; border-radius:8px; overflow:hidden; }
 #labview { position:absolute; inset:0; width:100%; height:100%; object-fit:contain; display:block; cursor:grab;
   touch-action:none; user-select:none; -webkit-user-select:none; -webkit-user-drag:none; }
 #labview.dragging { cursor:grabbing; }
 .labvideohint { position:absolute; left:10px; right:10px; bottom:10px; text-align:center; font-size:12px; color:#ccc;
   background:rgba(10,10,10,0.7); border-radius:6px; padding:3px 8px; pointer-events:none; }
 .labviews { display:flex; flex-wrap:wrap; align-items:center; gap:4px 6px; margin:8px 0 4px; font-size:12px; color:#9a9a9a; }
 .labviews button { padding:2px 9px; font-size:12px; }
 .labfollow { display:flex; align-items:center; gap:4px; margin-left:auto; color:#c8c8c8; }
 .trialbtns { display:flex; flex-direction:column; gap:6px; }
 .trialcond { display:flex; flex-direction:column; gap:3px; }
 .trialbtns .row { display:flex; flex-wrap:wrap; align-items:center; gap:4px; font-size:12px; }
 .trialbtns .row > span { width:50px; color:#9a9a9a; }
 .trialbtns button { padding:2px 8px; font-size:12px; font-variant-numeric:tabular-nums; }
 .trialbtns button.playing { background:#3d6ea5; border-color:#5b8fd0; }
 .labgrid { max-height:340px; }
 .launch { display:flex; flex-wrap:wrap; align-items:center; gap:6px 10px; margin-bottom:6px; }
 .labmini { display:flex; flex-direction:column; gap:3px; font-size:12px; color:#bdbdbd; }
 .labmini .row { display:flex; flex-wrap:wrap; align-items:center; gap:4px 8px; }
 .sablab { cursor:pointer; }
 .sablab.on b { color:#9cc3f0; }
 @media (max-width:1100px) {
   .setcols, .resgrid { grid-template-columns:minmax(0,1fr); }
   .lab { padding:0; }
   .labpanel { border-radius:0; }
 }
 .selbar button { padding:1px 8px; font-size:11px; }
 @media (max-width:1100px) {
   /* One column, and the page scrolls: every panel gets a real height instead
      of a share of a viewport too short to hold them all. */
   html, body { height:auto; }
   main.sandbox { grid-template-columns:minmax(0,1fr); }
   .rail > #mapcard { flex:none; height:620px; }
   .rail > #learncard { height:230px; }
   body { overflow:auto; grid-template-rows:auto auto auto; }
   main { grid-template-columns:minmax(0,1fr); }
   .stagecol > .stage { flex:none; aspect-ratio:16 / 9; }
   #view { object-fit:contain; }
   .rail > .card { flex:none; height:230px; }
   .rail > #kccard { height:300px; }
   footer, footer.three, footer.four { grid-template-columns:minmax(0,1fr); }
   footer > .card { height:240px; }
   .rail > #vkccard { height:260px; }
 }
</style>

<header>
  <h1>NeuroMechFly <small id="terrain"></small></h1>
  <div id="learn" class="learn" style="display:none">
    <b id="lsteps"></b> 스텝 학습<span class="sep">·</span>
    최근 <b id="lrecent"></b>회 중 <b id="lfound"></b>회 도달<span class="sep">·</span>
    에피소드 <b id="leps"></b><span class="sep">·</span>
    체크포인트 <b id="lreload"></b>회 갱신
  </div>
  <div id="condbar" class="learn" style="display:none">
    시행 <b id="ctrial"></b>/<b id="ctotal"></b><span class="sep">·</span>
    <b id="ckind"></b><span class="sep">·</span>
    <span id="ccslabel">벌받는 냄새</span> <b id="ccs"></b><span class="sep">·</span>
    <span id="cintlabel">세기</span> <b id="cint"></b><span class="sep">·</span>
    선호도 <b id="cpref"></b>
  </div>
  <div id="sandbar" class="learn" style="display:none">
    행동 <b id="smode"></b><span class="sep">·</span>
    배고픔 <b id="shunger"></b><span class="sep">·</span>
    전기 <b id="sshocks"></b><span class="sep">·</span>
    먹은 시간 <b id="sfeed"></b><span class="sep">·</span>
    한 번 시행 <b id="strial"></b><span class="sep">·</span>
    <span id="sablab" class="sablab" title="눌러서 A/B 실험실 열기">A/B 실험 <b id="sab">-</b></span><span class="sep">·</span>
    <span title="시뮬레이션 1초가 실제 몇 초에 흐르는지입니다. ×0.40이면 실제 시계보다 2.5배 느립니다.">실제 속도</span> <b id="sspeed"></b>
  </div>
  <div class="stats">
    <span><i>시간</i><b id="time"></b></span>
    <span><i>위치</i><b id="pos"></b></span>
    <span><i>속도</i><b id="spd"></b></span>
    <span><i id="lbl_sig">하행 L/R</i><b id="sig"></b></span>
    <span><i id="lbl_up">기울기</i><b id="up"></b></span>
    <span><i id="lbl_legs">접지</i><b id="legs"></b></span>
    <span id="forage" style="display:none">
      <i>냄새</i><b id="odor"></b> <i id="lbl_goal">먹이까지</i><b id="goaldist"></b>
      <i id="lbl_touch">충돌</i><b id="touch"></b>
    </span>
    <span id="senses" style="display:none">
      <i>냄새 좌우차</i><b id="oasym"></b> <span id="vissense"><i>시각 좌우차</i><b id="vasym"></b>
      <i>시야</i><b id="vtot"></b></span>
    </span>
  </div>
</header>

<main>
  <div class="stagecol">
    <div class="stage">
      <img id="view" src="/stream" draggable="false" alt="" title="왼쪽 드래그: 시점 회전 · 오른쪽 드래그: 화면 이동 · 휠: 줌">
      <div id="replaybar" class="replaybar" style="display:none">
        <b>다시 보기</b> <span id="r_label"></span>
        <button id="r_play">⏸</button>
        <select id="r_speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select>
        <input id="r_seek" type="range" min="0" max="1000" value="0">
        <span id="r_time"></span>
        <button id="r_stop" title="기록 보기를 끝내고 원래 파리로 돌아갑니다">■ 실시간으로</button>
      </div>
      <div class="legend" id="stagelegend" style="display:none"><span id="legendodour" style="display:contents">
        <span><i style="background:#f2731a"></i>냄새 A 발생원</span>
        <span><i style="background:#408cf2"></i>냄새 B 발생원</span>
        <small>초파리 눈에는 보이지 않는 표시입니다. 냄새로만 찾아갑니다. 훈련 때는 벌받는 냄새
          하나만, 시험 때는 두 냄새가 좌우에 놓입니다. 드래그로 시점을 돌리고 휠로 확대합니다.</small>
      </span><span id="legendcolour" style="display:none">
        <span><i style="background:#2f5fd6"></i>파랑 타일</span>
        <span><i style="background:#3aa843"></i>초록 타일</span>
        <small>바닥 타일이 자극이고, 초파리 눈에 보입니다(냄새는 없습니다). 훈련 때는 바닥 전체가 한 색으로
          60초, 시험 때는 두 색 체커보드로 90초입니다. 타일은 초파리를 따라 한 주기씩 옮겨져 무늬는 끝없이
          이어집니다. 드래그로 시점을 돌리고 휠로 확대합니다.</small>
      </span></div>
    </div>
    <div class="card" id="explaincard" style="display:none">
      <div class="tabs" id="sbtabs" style="display:none">
        <button data-tab="explain" class="on">설명</button><button data-tab="exp">A/B 실험실</button><button data-tab="trial">한 번 시행 (결과표)</button>
      </div>
      <p class="intro" id="intro"><b>실험 설명</b> — 냄새 A와 B 중 하나에만 벌(도파민)을 짝지으면 초파리가
        그 냄새를 피하게 되는지 봅니다. 학습되는 것은 <b>케니언 세포 → MBON 시냅스</b>뿐이고
        강화학습(PPO)은 쓰지 않습니다. 선호도는 +1이 B 쪽, −1이 A 쪽입니다.</p>
      <ol class="protocol" id="protocol"></ol>
      <p class="now-text" id="explainnow"></p>
      <div id="trialpane" style="display:none">
        <div class="trialrow">
          조건 이름 <input id="t_label" type="text" value="조건 A" maxlength="40">
          방 <select id="t_preset"></select>
          시작 배고픔 <input id="t_hunger" type="range" min="0" max="1" step="0.05" value="0.8"><b id="t_hunger_v">0.80</b>
          시간 <select id="t_seconds"><option value="30">30초</option><option value="60" selected>60초</option>
            <option value="120">2분</option><option value="300">5분</option><option value="600">10분</option></select>
          <label title="끄면 앞 실험의 파리(기억·배고픔·위치)로 이어서 합니다"><input id="t_newfly" type="checkbox" checked>새 파리로</label>
          <label title="방향이 매번 같으면 늘 같은 쪽 물건부터 만나게 됩니다"><input id="t_random" type="checkbox" checked>시작 방향 무작위</label>
        </div>
        <div class="trialrow">
          <button id="t_start">▶ 실험 시작</button>
          <button id="t_stop">■ 지금 멈추고 기록</button>
          <button id="t_csv">결과표 CSV 저장</button>
          <button id="t_clear">표 지우기</button>
          <span id="t_status"></span>
        </div>
        <div class="trialtable"><table id="t_table"></table></div>
      </div>
      <div id="exppane" style="display:none">
        <div class="launch">
          <button id="lab_open" class="primary">A/B 실험실 열기</button>
          <span class="lbl">질문 고르기 → 가설 세우기 → 배경에서 실험 → 결과와 3D 다시 보기</span>
        </div>
        <div id="labmini" class="labmini"></div>
      </div>
    </div>
  </div>
  <div class="rail">
    <div class="card" id="mapcard" style="display:none">
      <div class="cap">방 편집 — 위에서 본 100 × 100 mm 방
        <span class="helpinline" title="놓기 도구(설탕·전기·냄새·색 바닥·장애물)로 클릭하면 겹쳐 있어도 새로 놓입니다.&#10;선택 도구로 클릭하면 고르고, 끌면 옮깁니다. 고른 물건은 아래 값을 바꾸면 바로 바뀝니다. 겹친 곳은 같은 자리를 다시 클릭하면 다음 물건이 골라집니다. Esc로 해제.&#10;오른쪽 클릭: 지우기 · 오른쪽 버튼으로 끌기: 지도 이동 · 휠: 확대 · 빈 곳 더블클릭: 원래대로.&#10;마우스를 물건에 올리면 자세한 값이 뜹니다.&#10;설탕은 먹는 만큼 줄어 다 먹으면 사라집니다(초당 약 5 nl). 주황 삼각형이 초파리, 흐린 선이 지나온 길입니다. 초파리는 초파리 도구로 클릭하거나 선택 도구로 끌어서 옮깁니다(기억·배고픔은 그대로).&#10;초파리 눈에 보이는 것은 색 바닥과 벽·장애물뿐이고, 설탕·전기·냄새 표시는 사람에게만 보입니다.">사용법 ⓘ</span>
      </div>
      <div class="palette">
        <div class="tools">
          <button class="tool" data-tool="select">선택</button>
          <button class="tool on" data-tool="sugar">설탕</button>
          <button class="tool" data-tool="shock">전기</button>
          <button class="tool" data-tool="odour">냄새</button>
          <button class="tool" data-tool="patch">색 바닥</button>
          <button class="tool" data-tool="obstacle">장애물</button>
          <button class="tool" data-tool="fly" title="지도를 클릭한 곳으로 초파리를 옮깁니다">초파리</button>
          <button class="tool" data-tool="erase">지우개</button>
          <span class="undo"><button id="b_undo" disabled>↶ 되돌리기</button><button id="b_redo" disabled>↷</button></span>
        </div>
        <div class="optsbox">
        <div class="opts" data-for="select"><span class="hint">클릭: 고르기 · 끌기: 옮기기 · 같은 자리 다시 클릭: 겹친 다음 물건 · Esc: 해제</span></div>
        <div class="opts on" data-for="sugar">
          <select id="p_sugar"></select>
          농도 <input id="p_molar" type="range" min="0.05" max="2" step="0.05" value="1">
          <b id="p_molar_v">1.00 M</b>
          양 <input id="p_volume" type="range" min="20" max="500" step="10" value="100" title="바꾸면 그 양으로 다시 채워집니다">
          <b id="p_volume_v">100 nl</b>
          <button id="b_deplete" class="on" title="켜짐: 먹는 만큼 줄고 다 먹으면 사라짐 · 꺼짐: 그대로 남음">먹으면 줄어듦: 켜짐</button>
        </div>
        <div class="opts" data-for="shock">
          전압 <input id="p_volts" type="range" min="0" max="120" step="5" value="60">
          <b id="p_volts_v">60 V</b>
          크기 <input id="p_half_shock" type="range" min="3" max="30" step="1" value="10"><b id="p_half_shock_v">20 mm</b>
        </div>
        <div class="opts" data-for="odour">
          <select id="p_odour"></select>
          세기 <input id="p_strength" type="range" min="0.05" max="1" step="0.05" value="1">
          <b id="p_strength_v">1.00</b>
        </div>
        <div class="opts" data-for="patch">
          <select id="p_colour"><option value="blue">파랑</option><option value="green">초록</option></select>
          크기 <input id="p_half_patch" type="range" min="3" max="30" step="1" value="12"><b id="p_half_patch_v">24 mm</b>
        </div>
        <div class="opts" data-for="obstacle">
          <select id="p_shape"><option value="block">블록</option><option value="wall_h">벽 (가로)</option><option value="wall_v">벽 (세로)</option></select>
          길이 <input id="p_length" type="range" min="4" max="100" step="2" value="8"><b id="p_length_v">8 mm</b>
          <span class="hint">벽 두께 2 mm. 초파리는 닿기 전에 돌아섭니다.</span>
        </div>
        <div class="opts" data-for="erase"><span class="hint">지울 물건을 클릭하세요.</span></div>
        <div class="opts" data-for="fly">
          방향 <select id="p_fly_yaw"><option value="keep">지금 그대로</option><option value="0">→ 오른쪽</option><option value="90">↑ 위</option><option value="180">← 왼쪽</option><option value="270">↓ 아래</option></select>
          <span class="hint">클릭한 곳으로 초파리를 옮깁니다. 기억·배고픔은 그대로. 선택 도구로 초파리를 끌어도 됩니다.</span>
        </div>
        </div>
        <div class="row">
          프리셋 <select id="p_preset"></select>
          <button id="b_clear">전부 지우기</button>
          <span class="selbar" id="selbar"><span id="p_selected"></span><button id="b_deselect">해제</button></span>
        </div>
      </div>
      <div class="plot">
        <div class="mapctl">
          이름표 <select id="p_labels"><option value="simple" selected>간단히</option><option value="full">자세히</option><option value="none">없음</option></select>
          지나온 길 <input id="p_trail" type="range" min="0" max="13" step="1" value="3" title="경로가 얼마 동안 남았다가 사라질지. 맨 오른쪽은 무한: 파리를 놓은 뒤의 길이 모두 남습니다(최대 하루)."><b id="p_trail_v">30초</b>
          <button id="p_heat" title="초파리가 오래 머문 곳일수록 밝게 칠합니다. 기간은 '지나온 길'과 같고, 무한이나 '안 보임'이면 파리를 놓은 뒤 전체입니다.">히트맵</button>
        </div>
        <canvas id="c_map"></canvas>
      </div>
    </div>
    <div class="card" id="eyecard">
      <div class="cap" id="eyecap">겹눈이 보는 화면
        <small>왼쪽 눈 | 오른쪽 눈. 먹이는 <b style="color:#c9863f">안 보이고</b>(냄새로만 찾음),
          기둥은 보입니다.</small>
      </div>
      <div class="plot"><img id="eyeview" alt=""></div>
    </div>
    <div class="card" id="kccard" style="display:none">
      <div class="cap">케니언 세포 2000개
        <small>밝은 점이 지금 켜진 100개(5%)입니다. 바탕의 흐린
          <b style="color:#b07434">A</b>·<b style="color:#4a7cb0">B</b> 점은 각 냄새의 기준 패턴입니다.
          냄새에 가까워져도 같은 점에 머물고(농도 불변), 냄새가 바뀌면 통째로 다른 자리로 옮겨 갑니다(냄새 분리).
          두 냄새가 공유하는 세포는 8% 정도입니다.</small>
      </div>
      <div class="plot"><canvas id="c_kc"></canvas></div>
    </div>
    <div class="card" id="vpncard" style="display:none">
      <div class="cap">시각 투사 뉴런 (VPN)
        <small>위 줄이 왼쪽 눈, 아래 줄이 오른쪽 눈입니다. <b style="color:#5b8fe8">파랑</b>·<b style="color:#4fbf5a">초록</b>
          막대 24개는 색 VPN으로, 바닥의 작은 조각(낱눈 20개)이 파랑 쪽 또는 초록 쪽으로 얼마나 기울었는지를
          냅니다. 밝기와 무관하고, 회색이면 0입니다. 오른쪽 <b style="color:#b0b0b0">회색</b> 막대 4개는 밝기 VPN으로,
          왼쪽부터 어두움 → 밝음 대역에 맞춰져 있습니다. 모두 낱눈 줄 분위 0.6–0.8의 바닥 띠만 봅니다(하늘과 자기 다리가
          섞이는 줄은 뺐습니다).</small>
      </div>
      <div class="plot"><canvas id="c_vpn"></canvas></div>
    </div>
    <div class="card" id="vkccard" style="display:none">
      <div class="cap">시각 케니언 세포 (γd형) — 왼쪽 눈 | 오른쪽 눈
        <small>눈마다 100개이고, 밝은 칸이 지금 켜진 5개입니다. 흐린 <b style="color:#3a5a9a">파랑</b>·<b style="color:#3a7a40">초록</b>
          칸은 각각 파랑 바닥·초록 바닥 위에서 켜지는 기준 배치입니다. 두 색은 겹치는 세포가 없어서, 한쪽 눈이
          경계 너머 다른 색을 보기 시작하면 그 눈의 배치만 통째로 바뀝니다. 이 세포들은 냄새 세포와 <b>같은</b> MBON·도파민을
          씁니다.</small>
      </div>
      <div class="plot"><canvas id="c_vkc"></canvas></div>
    </div>
    <div class="card" id="learncard" style="display:none">
      <div class="cap">버섯체가 배운 것
        <small>각 냄새와 색에 매긴 가치입니다. <b style="color:#6ad07a">+</b>는 끌림(설탕과 함께 겪음), <b style="color:#e06c6c">−</b>는
          피함(전기와 함께 겪음)이고, 한 번도 짝지어진 적 없으면 0입니다. 설탕 기억은 배고픔만큼만 드러나고
          전기 기억은 늘 드러납니다. 안쪽에는 구획이 넷 있습니다 — 전기: 빨리 배우고 몇 시간에 잊는 곳 + 천천히 배우고 며칠 가는 곳,
          설탕: 단맛(몇 시간) + 영양(며칠). 식초는 여기 값과 별개로 타고난 끌림이 있습니다.</small>
      </div>
      <div class="plot"><canvas id="c_learn"></canvas></div>
    </div>
    <div class="card" id="odorcard">
      <div class="cap">냄새
        <small>왼쪽·오른쪽 센서가 느끼는 냄새 세기의 시간 변화입니다. 두 선의 차이가 조향 신호이고,
          초파리는 그 차이를 따라 냄새 쪽(또는 학습 후 반대쪽)으로 돕니다.</small>
      </div>
      <div class="plot"><canvas id="c_odor"></canvas></div>
    </div>
  </div>
</main>

<footer>
  <div class="card" id="attrcard">
    <div class="cap">감각 기여도 — 지금
      <small>각 감각 입력을 조금씩 흔들었을 때 조향 명령이 얼마나 바뀌는지입니다. 높을수록 정책이
        그 감각에 의존하고 있다는 뜻입니다. 세로 점선은 에피소드 경계입니다.</small>
    </div>
    <div class="plot"><canvas id="c_attr"></canvas></div>
  </div>
  <div class="card" id="attrhistcard">
    <div class="cap">감각 기여도 — 학습 경과
      <small>디스크의 체크포인트마다 같은 상태 묶음 64개로 측정했습니다. 같은 입력으로 쟀기 때문에
        차이는 정책이 바뀐 결과입니다. 막대가 자라면 학습이 그 감각을 쓰기 시작한 것입니다.</small>
    </div>
    <div class="plot"><canvas id="c_attrhist"></canvas></div>
  </div>
  <div class="card" id="mboncard" style="display:none">
    <div class="cap" id="mboncap">MBON 출력
      <small>버섯체 출력 뉴런 두 개의 반응입니다. 케니언 세포는 냄새를 맡는 동안에만 켜지므로 값도
        그때만 올라옵니다. 주황 세로줄은 도파민(벌)이 들어온 구간이고, 그 순간 켜져 있던 세포의
        <b style="color:#6ad07a">접근</b> 쪽 시냅스만 약해집니다. 초록 봉우리가 시행마다 낮아지는 것이
        학습이고, <b style="color:#d66a6a">회피</b> 선은 벌로는 움직이지 않습니다.</small>
    </div>
    <div class="plot"><canvas id="c_mbon"></canvas></div>
  </div>
  <div class="card" id="valcard" style="display:none">
    <div class="cap" id="valcapdiv">학습된 가치와 접근 구동
      <small><b style="color:#c98fd6">가치</b> = 접근 MBON − 회피 MBON으로, 버섯체가 배운 것만 나타냅니다
        (처음 보는 냄새는 0). <b style="color:#7fb2f0">접근 구동</b> = 타고난 끌림(외측각, 0.35) + 가치이고,
        이것이 조향의 부호를 정합니다. 파란 선이 0 위면 냄새 쪽으로, 0 아래면 반대로 돕니다.
        세로 점선은 블록 경계입니다. <span id="valcap"></span></small>
    </div>
    <div class="plot"><canvas id="c_val"></canvas></div>
  </div>
  <div class="card" id="eyevalcard" style="display:none">
    <div class="cap">두 눈이 보는 가치와 조향 (최근 10초)
      <small>각 눈이 보는 바닥의 학습된 가치입니다(−1 = 그 색의 시냅스가 모두 억압됨).
        <b style="color:#4d8fd6">왼쪽 눈</b>과 <b style="color:#d68f4d">오른쪽 눈</b>의 차이에 이득 <span id="gainval"></span>을
        곱한 것이 회전 명령이고, <b style="color:#e8e8e8">흰 선</b>이 실제 회전입니다(+ 왼쪽, −1..+1).
        가치가 더 높은 눈 쪽으로 돕니다. 경계를 옆에 끼고 걸을 때만 두 눈이 달라지고, 정면으로 넘을 때는
        둘이 같이 움직여 회전이 생기지 않습니다.</small>
    </div>
    <div class="plot"><canvas id="c_eyeval"></canvas></div>
  </div>
  <div class="card" id="dancard" style="display:none">
    <div class="cap">도파민과 배고픔 (최근 40초)
      <small><b style="color:#e06c6c">벌</b>은 전기에 닿을 때마다 최소 1.5초(표준 충격 펄스 길이), <b style="color:#e58fd0">단맛 보상</b>과
        <b style="color:#5fc6d8">영양 보상</b>은 설탕을 먹는 동안 나옵니다. 그 순간 맡고 있는 냄새·보고 있는 색이 기억됩니다.
        영양 보상은 배고플 때만 기억에 새겨집니다. <b style="color:#9a9a9a">회색</b> 선은 배고픔(1 = 굶주림)으로, 영양 있는
        설탕을 먹으면 내려갑니다(실제 수 시간을 몇 분으로 압축).</small>
    </div>
    <div class="plot"><canvas id="c_dan"></canvas></div>
  </div>
  <div class="card" id="bodycard" style="display:none">
    <div class="cap">초파리 상태 <small>배고픔을 직접 바꿀 수 있습니다. 배부르면 설탕 위에서 멈추지 않고, 설탕 기억도 행동으로 드러나지 않습니다.</small></div>
    <div class="hungerrow">배고픔 <input id="p_hunger" type="range" min="0" max="1" step="0.05" value="0.8"><b id="p_hunger_v"></b></div>
    <div class="kv" id="bodykv"></div>
  </div>
  <div class="card" id="logcard" style="display:none">
    <div class="cap">사건 기록 <small>최근 것이 위에 옵니다. 시간은 시뮬레이션 초입니다.</small></div>
    <ul class="log" id="eventlog"></ul>
  </div>
  <div class="foot-controls">
    <button id="pause">일시정지</button>
    <button data-act="reset">리셋</button>
    <button id="b_home" style="display:none" title="같은 파리를 방 가운데로 옮깁니다. 배운 기억과 배고픔은 그대로입니다.">초파리 가운데로</button>
    <span class="pad" id="pad" style="display:none">
      <button data-k="a">◀ 좌</button><button data-k="c">중립</button>
      <button data-k="d">우 ▶</button><button data-k="w">빠르게</button>
      <button data-k="s">느리게</button>
    </span>
    <span class="gap"></span>
    <span class="lbl">속도</span>
    <button class="spd" data-spd="1">1.0x</button>
    <button class="spd" data-spd="0.5">0.5x</button>
    <button class="spd" data-spd="0.25">0.25x</button>
    <button class="spd" data-spd="0.1">0.1x</button>
    <span class="gap"></span>
    <span class="lbl">시점</span>
    <button class="view" data-view="room">방 전체</button>
    <button class="view" data-view="wide">넓게</button>
    <button class="view" data-view="side">측면</button>
    <button class="view" data-view="close">근접</button>
    <button class="view" data-view="top">위에서</button>
    <span class="gap"></span>
    <span class="lbl">줌</span>
    <button data-zoom="out">−</button>
    <input id="dist" type="range" min="2" max="60" step="0.5">
    <button data-zoom="in">+</button>
    <span id="distlbl" class="lbl"></span>
    <span class="help" title="화면을 드래그하면 시점이 돌고 휠로 확대·축소합니다. 카메라는 항상 파리를 따라갑니다.&#10;&#10;도파민 그래프의 δ는 강화학습의 보상예측오차입니다. 이것이 실제 동물의 도파민 뉴런 활동에 대응한다는 것이 Schultz, Dayan &amp; Montague (1997)의 결과이고, 초파리에서는 버섯체로 투사하는 도파민 뉴런이 같은 일을 합니다. δ가 0보다 크면 예상보다 잘 풀렸다는 뜻입니다.&#10;&#10;--conditioning 모드에서는 강화학습 대신 후각 조건화 실험이 돌아갑니다. 벌받는 냄새를 맡는 동안에만 도파민이 들어오고, 그 순간 켜져 있던 케니언 세포의 시냅스만 약해집니다. 회피는 별도의 회로가 아니라 접근이 사라진 결과입니다.">도움말 ⓘ</span>
  </div>
</footer>

<div id="labroot" class="lab" style="display:none">
  <div class="labpanel" role="dialog" aria-modal="true" aria-label="A/B 실험실">
    <div class="labtop">
      <b class="labtitle">A/B 실험실</b>
      <ol class="labsteps" id="labsteps">
        <li data-step="home">① 질문 고르기</li>
        <li data-step="set">② 가설 세우고 시작</li>
        <li data-step="result">③ 결과와 3D 다시 보기</li>
      </ol>
      <span class="labspacer"></span>
      <span class="lbl">닫아도 실험은 배경에서 계속 돕니다</span>
      <button id="lab_close" title="실험실 닫기 (Esc)">✕ 닫기</button>
    </div>
    <div class="labbody" id="labbody">
      <section id="lab_home"></section>
      <section id="lab_set" style="display:none"></section>
      <section id="lab_result" style="display:none">
        <button class="linkbtn" data-go="home">← 실험실 처음 화면</button>
        <div id="lr_head" class="reshead"></div>
        <div class="resgrid">
          <div class="labcard" id="lr_verdict"></div>
          <div class="labcard">
            <div class="labcardtitle">3D로 다시 보기 <small>아래 시행 버튼을 누르면 이 화면에서 그 시행을 다시 봅니다</small></div>
            <div class="labvideo" id="labvideo">
              <img id="labview" draggable="false" alt="" title="왼쪽 드래그: 시점 회전 · 오른쪽 드래그: 화면 이동 · 휠: 줌">
              <div class="labvideohint" id="lr_hint">지금은 실시간 초파리입니다. 아래에서 시행을 고르세요.</div>
            </div>
            <div class="labviews">시점
              <button data-labview="top">위에서</button><button data-labview="room">방 전체</button><button data-labview="close">근접</button><button data-labview="side">측면</button>
              <label class="labfollow" title="파리 1~3번의 시행이 끝날 때마다 차례로 다시 보여 줍니다"><input type="checkbox" id="lr_follow">새 시행이 끝나면 자동으로 보기</label>
            </div>
            <div id="lr_trials" class="trialbtns"></div>
          </div>
        </div>
        <details class="labmore"><summary>모든 시행의 숫자 표</summary><div class="trialtable labgrid"><table id="lr_grid"></table></div></details>
        <details class="labmore"><summary>왜 이런 결과가 나올까? (이 모델의 작동 방식)</summary><p id="lr_why"></p></details>
        <details class="labmore"><summary>실제 연구에서는</summary><p id="lr_basis"></p></details>
      </section>
    </div>
  </div>
</div>

<script>
const post = (o) => fetch('/control', {method:'POST', body:JSON.stringify(o)});
const keymap = {w:'w', a:'a', s:'s', d:'d', c:'c',
                ArrowUp:'w', ArrowLeft:'a', ArrowDown:'s', ArrowRight:'d'};
const on = (sel, fn) => document.querySelectorAll(sel).forEach(
  b => b.onclick = () => fn(b));

on('.pad button', b => post({key: b.dataset.k}));
on('[data-act]',  b => post({act: b.dataset.act}));
b_home.onclick = () => fetch('/sandbox', {method:'POST', body:JSON.stringify({op:'fly_home'})});
on('.view',       b => post({view: b.dataset.view}));
on('[data-zoom]', b => post({zoom: b.dataset.zoom}));
on('[data-orbit]',b => post({orbit: parseFloat(b.dataset.orbit)}));
on('.spd',        b => post({speed: parseFloat(b.dataset.spd)}));
document.getElementById('pause').onclick = () => post({act:'pause'});

let dragging = false;
dist.oninput = () => { dragging = true; post({distance: parseFloat(dist.value)}); };
dist.onchange = () => { dragging = false; };

addEventListener('keydown', e => {
  if (e.target.closest && e.target.closest('input, select, textarea')) return;
  const k = keymap[e.key];
  if (k) { e.preventDefault(); post({key: k}); }
  else if (e.key === '+' || e.key === '=') post({zoom:'in'});
  else if (e.key === '-') post({zoom:'out'});
});
// Drag inside the render to orbit, exactly like the native MuJoCo viewer.
// Deltas are throttled to one request per animation frame so a fast drag does
// not queue up hundreds of POSTs.
let drag = null, pending = null, queued = false;
const flush = () => {
  queued = false;
  if (pending) { post(pending); pending = null; }
};
// The live view and the A/B lab's replay view steer the same camera.
function attachOrbit(img) {
  img.addEventListener('wheel', e => {
    e.preventDefault();
    post({zoom: e.deltaY > 0 ? 'out' : 'in'});
  }, {passive:false});
  img.addEventListener('contextmenu', e => e.preventDefault());
  img.addEventListener('pointerdown', e => {
    // The render is an <img>, and a browser's default for press-and-move on an
    // image is to pick it up and drag a ghost copy around. That default fires
    // pointercancel the moment it starts, which ended the orbit after a few
    // pixels and left the user dragging a picture instead.
    e.preventDefault();
    // Left button orbits; right button slides the view across the scene.
    drag = {x: e.clientX, y: e.clientY, pan: e.button === 2};
    img.classList.add('dragging');
    img.setPointerCapture(e.pointerId);
  });
  img.addEventListener('dragstart', e => e.preventDefault());
  img.addEventListener('pointermove', e => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag = {x: e.clientX, y: e.clientY, pan: drag.pan};
    if (drag.pan) {
      pending = pending || {};
      pending.pan = [(pending.pan ? pending.pan[0] : 0) + dx, (pending.pan ? pending.pan[1] : 0) + dy];
      if (!queued) { queued = true; requestAnimationFrame(flush); }
      return;
    }
    pending = pending || {};
    pending.orbit = (pending.orbit || 0) - dx * 0.4;
    pending.elevation_delta = pending.elevation_delta || 0;
    // Dragging down tips the camera up over the fly and dragging up lowers it
    // toward the ground, as if pulling the scene. The first version had it the
    // other way round; inverted at the user's request.
    pending.elevation_delta += -dy * 0.3;
    if (!queued) { queued = true; requestAnimationFrame(flush); }
  });
  const endDrag = () => { drag = null; img.classList.remove('dragging'); };
  img.addEventListener('pointerup', endDrag);
  img.addEventListener('pointercancel', endDrag);
}
attachOrbit(view);
attachOrbit(labview);

// ---------- neural dashboard ----------
// The canvases are laid out by CSS but drawn in pixels, so the backing store
// has to track the displayed size or every graph comes out stretched.
function fitCanvases() {
  document.querySelectorAll('canvas').forEach(cv => {
    const r = cv.getBoundingClientRect();
    const w = Math.max(80, Math.round(r.width)), h = Math.max(50, Math.round(r.height));
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  });
}
addEventListener('resize', fitCanvases);

const ctx = id => document.getElementById(id).getContext('2d');
const clear = (c, bg='#111') => {
  c.fillStyle = bg; c.fillRect(0, 0, c.canvas.width, c.canvas.height);
};
const text = (c, s, x, y, col='#8a8a8a', size=10) => {
  c.fillStyle = col; c.font = size + 'px ui-sans-serif,system-ui,sans-serif';
  c.fillText(s, x, y);
};

function traces(c, series, opts={}) {
  clear(c);
  const W = c.canvas.width, H = c.canvas.height, pad = 16;
  // Either bound may be pinned; whatever is left out is fitted to the data.
  const all = series.flatMap(s => s.data);
  const dataLo = Math.min(...all), dataHi = Math.max(...all);
  const span = Math.max(1e-6, dataHi - dataLo);
  let lo = opts.lo !== undefined ? opts.lo : dataLo - span * 0.1;
  let hi = opts.hi !== undefined ? opts.hi : dataHi + span * 0.1;
  if (hi - lo < 1e-6) hi = lo + 1e-6;
  const y = v => H - pad - (v - lo) / Math.max(1e-9, hi - lo) * (H - 2 * pad);
  // Episode boundaries first, so the traces draw over them.
  if (opts.marks) {
    c.strokeStyle = opts.mcolor || '#3d3d3d'; c.setLineDash([3, 3]); c.lineWidth = 1;
    opts.marks.forEach((m, i) => {
      if (!m) return;
      const x = i / Math.max(1, opts.marks.length - 1) * W;
      c.beginPath(); c.moveTo(x, 0); c.lineTo(x, H); c.stroke();
    });
    c.setLineDash([]);
  }
  if (lo < 0 && hi > 0) {
    c.strokeStyle = '#333'; c.beginPath();
    c.moveTo(0, y(0)); c.lineTo(W, y(0)); c.stroke();
  }
  if (opts.hline !== undefined && opts.hline !== null) {
    c.strokeStyle = opts.hcolor || '#4caf6a'; c.lineWidth = 1.2;
    c.beginPath(); c.moveTo(0, y(opts.hline)); c.lineTo(W, y(opts.hline)); c.stroke();
  }
  series.forEach(s => {
    c.strokeStyle = s.color; c.lineWidth = 1.6; c.beginPath();
    s.data.forEach((v, i) => {
      const x = i / Math.max(1, s.data.length - 1) * W;
      i ? c.lineTo(x, y(v)) : c.moveTo(x, y(v));
    });
    c.stroke();
  });
  text(c, hi.toFixed(2), 4, 11);
  text(c, lo.toFixed(2), 4, H - 4);
  // Legend laid out right-to-left from the measured label widths, so four
  // series fit as readably as two.
  c.font = '10px ui-sans-serif,system-ui,sans-serif';
  let x = W - 6;
  for (let i = series.length - 1; i >= 0; i--) {
    const s = series[i], w = c.measureText(s.label).width;
    x -= w;
    text(c, s.label, x, 11, s.color, 10);
    x -= 12;
  }
}

//: Colours for the sensory attribution bars, in ATTRIBUTION_CHANNELS order.
const ATTR_COLORS = ['#e0a24d', '#e0c24d', '#c98fd6', '#6ad07a'];

// Grouped bars: one cluster per 10k training steps, one bar per sense. Reading
// left to right is reading the training run.
function attrBars(c, history, channels, ready) {
  clear(c);
  const W = c.canvas.width, H = c.canvas.height;
  const padL = 30, padB = 26, padT = 16;
  const shown = history.slice(-14);
  if (!shown.length) {
    text(c, ready ? '체크포인트 측정 중...'
                  : '상태 표본을 모으는 중... (에피소드 한두 번 기다려 주세요)',
         10, 22);
    return;
  }
  let max = 0;
  shown.forEach(h => channels.forEach(k => max = Math.max(max, h.means[k] || 0)));
  max = Math.max(max, 1e-6);
  const y = v => H - padB - v / max * (H - padB - padT);
  c.strokeStyle = '#333';
  c.beginPath(); c.moveTo(padL, H - padB); c.lineTo(W, H - padB); c.stroke();
  text(c, max.toFixed(2), 2, padT + 4);
  text(c, '0', 2, H - padB + 3);

  const slot = (W - padL - 6) / shown.length;
  const bw = Math.max(2, (slot - 7) / channels.length);
  shown.forEach((h, i) => {
    const x0 = padL + i * slot + 3;
    channels.forEach((k, j) => {
      const v = h.means[k] || 0;
      c.fillStyle = ATTR_COLORS[j % ATTR_COLORS.length];
      c.fillRect(x0 + j * bw, y(v), bw - 1, H - padB - y(v));
    });
    text(c, (h.steps / 1000) + 'k', x0, H - padB + 12, '#7a7a7a', 9);
  });
  // Legend along the bottom, coloured to match the bars.
  let lx = padL;
  channels.forEach((k, j) => {
    c.fillStyle = ATTR_COLORS[j % ATTR_COLORS.length];
    c.fillRect(lx, H - 9, 8, 8);
    text(c, k, lx + 11, H - 2, '#a8a8a8', 10);
    lx += 14 + c.measureText(k).width;
  });
}

// The 2000 Kenyon cells as a dot field. The panel exists for one comparison:
// the live pattern lies on top of the current odour's reference constellation
// and moves wholesale to the other one when the odour changes. Measured on seed
// 0, A and B share 3 of their 100 cells, and A at a tenth the concentration
// keeps 99 of them -- so "still the same dots" really does mean "the same
// smell, only nearer".
function kcRaster(c, active, refs, odour, n) {
  clear(c);
  const W = c.canvas.width, H = c.canvas.height, foot = 12;
  // Columns from the aspect ratio, so the cells stay square in whichever panel
  // slot this lands in.
  const cols = Math.max(1, Math.round(Math.sqrt(n * W / Math.max(1, H - foot))));
  const rows = Math.ceil(n / cols);
  const cw = W / cols, ch = (H - foot) / rows;
  const dot = (i, col, frac) => {
    const s = Math.max(1, Math.min(cw, ch) * frac);
    c.fillStyle = col;
    c.fillRect((i % cols) * cw + (cw - s) / 2,
               ((i / cols) | 0) * ch + (ch - s) / 2, s, s);
  };
  for (let i = 0; i < n; i++) dot(i, '#1f1f1f', 0.45);
  (refs.A || []).forEach(i => dot(i, '#5c3a18', 0.62));
  (refs.B || []).forEach(i => dot(i, '#1d3a5c', 0.62));
  (active || []).forEach(i => dot(i, odour === 'B' ? '#6aa9f0' : '#f0a055', 1.0));
  const lit = (active || []).length;
  text(c, lit ? '켜진 세포 ' + lit + ' / ' + n + ' · 지금 냄새 ' + (odour || '-')
              : '냄새 없음 · 켜진 세포 0 / ' + n,
       2, H - 2, '#8a8a8a', 10);
}

//: Floor colours as the page names them.
const COLOUR_KO = {blue:'파랑', green:'초록', green_tenth:'1/10 밝기 초록', dim_blue:'어두운 파랑',
                   dim_green:'어두운 초록', grey:'회색'};
const ko = name => COLOUR_KO[name] || name || '-';
//: Bright and dim swatches per colour, for rasters and bars.
const SWATCH = {blue:['#5b8fe8', '#1f3560'], green:['#4fbf5a', '#1f4a24'],
                green_tenth:['#9fd6a4', '#2a3d2c'], dim_blue:['#8fb0f0', '#26324d'],
                dim_green:['#8fd096', '#26402a'], grey:['#bdbdbd', '#3a3a3a']};

// Both eyes' visual Kenyon cells, side by side, over each colour's reference
// constellation. Measured: blue and green light disjoint sets of cells, so a
// panel whose bright cells sit on the blue backdrop is an eye looking at blue.
function vkcRaster(c, active, refs, stimuli, n, under) {
  clear(c);
  const W = c.canvas.width, H = c.canvas.height, foot = 13, gap = 10;
  const pw = (W - gap) / 2, ph = H - foot;
  const cols = Math.max(1, Math.round(Math.sqrt(n * pw / Math.max(1, ph))));
  const rows = Math.ceil(n / cols);
  const cw = pw / cols, ch = ph / rows;
  ['왼쪽 눈', '오른쪽 눈'].forEach((label, eye) => {
    const x0 = eye * (pw + gap);
    const dot = (i, col, frac) => {
      const s = Math.max(1, Math.min(cw, ch) * frac);
      c.fillStyle = col;
      c.fillRect(x0 + (i % cols) * cw + (cw - s) / 2, ((i / cols) | 0) * ch + (ch - s) / 2, s, s);
    };
    for (let i = 0; i < n; i++) dot(i, '#1f1f1f', 0.5);
    stimuli.forEach(name => (refs[name] || []).forEach(i => dot(i, (SWATCH[name] || SWATCH.grey)[1], 0.8)));
    // A lit cell takes the colour whose reference it belongs to, white if neither.
    (active[eye] || []).forEach(i => {
      const owner = stimuli.find(name => (refs[name] || []).includes(i));
      dot(i, owner ? (SWATCH[owner] || SWATCH.grey)[0] : '#f2f2f2', 1.0);
    });
    text(c, label + ' · 켜진 ' + (active[eye] || []).length + '/' + n, x0 + 2, H - 2, '#8a8a8a', 10);
  });
  text(c, '발밑: ' + ko(under), W - 70, H - 2, '#8a8a8a', 10);
}

// VPN outputs, one row per eye: colour VPNs in the colour they prefer, then the
// brightness bands in grey from dark to bright.
function vpnBars(c, vpn, labels) {
  clear(c);
  const W = c.canvas.width, H = c.canvas.height, gap = 16, top = 12;
  if (!vpn) { text(c, '눈이 아직 렌더되지 않았습니다', 10, 22); return; }
  const n = labels.length, rowH = (H - gap - top) / 2;
  const bw = W / n;
  ['왼쪽 눈', '오른쪽 눈'].forEach((label, eye) => {
    const y0 = top + eye * (rowH + gap), base = y0 + rowH;
    c.strokeStyle = '#333'; c.beginPath(); c.moveTo(0, base); c.lineTo(W, base); c.stroke();
    vpn[eye].forEach((v, i) => {
      const name = labels[i];
      const col = name === 'blue' ? '#5b8fe8' : name === 'green' ? '#4fbf5a'
        : 'hsl(0,0%,' + (35 + 12 * (i - labels.indexOf(labels.find(l => l.startsWith('brightness'))))) + '%)';
      c.fillStyle = col;
      const h = Math.max(0, Math.min(1, v)) * (rowH - 2);
      c.fillRect(i * bw + 1, base - h, Math.max(1, bw - 2), h);
    });
    text(c, label, 2, y0 - 2, '#8a8a8a', 10);
  });
}

// ---------- sandbox ----------
const SUGAR_KO = {sucrose:'자당', fructose:'과당', arabinose:'아라비노스 (단맛만)', sorbitol:'소르비톨 (영양만)',
                  arabinose_sorbitol:'아라비노스+소르비톨'};
const ODOUR_KO = {vinegar:'식초', octanol:'옥탄올', mch:'MCH'};
const ODOUR_COL = {vinegar:'#f0c030', octanol:'#d85ad8', mch:'#33cccc'};
const SUGAR_COL = {sucrose:'#fafaeb', fructose:'#ffd98c', arabinose:'#ff99cc', sorbitol:'#b3e6ff',
                   arabinose_sorbitol:'#d9bfff'};
const MODE_KO = {explore:'탐색', feed:'먹는 중', escape:'도망', wall:'벽 회피', search:'주변 탐색',
                 retract:'주둥이 접는 중'};
const sb = {tool:'sugar', items:[], fly:null, half:50, drag:null, selected:null,
            // The trail as flat x/y pairs, kept in step with the server's (syncTrail).
            tx:[], tEpoch:-1, tTotal:0, tFetching:false, heat:false,
            lastMove:0, paletteReady:false, hungerDrag:false,
            // The map's own view: world point at its centre, and magnification.
            view:{cx:0, cy:0, zoom:1}, pan:null,
            // Pointer hover (canvas px), last select click for cycling overlaps,
            // label detail, trail window in seconds, trial table state.
            hover:null, cycle:null, labels:'simple', trailSeconds:30, rowsShown:-1, tab:'explain'};
const sbPost = o => fetch('/sandbox', {method:'POST', body:JSON.stringify(o)});
// Trail window choices on the slider, seconds. The last keeps all of it.
const TRAIL_STEPS = [0, 10, 20, 30, 60, 90, 120, 180, 300, 600, 1200, 1800, 3600, Infinity];
const trailLabel = v => v === 0 ? '안 보임' : v === Infinity ? '무한' : v < 60 ? v + '초' :
  v < 3600 ? (v / 60) + '분' : (v / 3600) + '시간';
// Occupancy heatmap: trail points counted in 2 mm cells, smoothed over the
// neighbouring cells, square-rooted so a place visited briefly still shows
// beside one sat on for a minute.
const HEAT_BINS = 50;
const HEAT_STOPS = [[0, [70, 20, 110]], [0.35, [200, 40, 60]], [0.7, [250, 150, 30]], [1, [255, 245, 150]]];
function heatColour(v) {
  for (let i = 1; i < HEAT_STOPS.length; i++) {
    if (v <= HEAT_STOPS[i][0]) {
      const [t0, c0] = HEAT_STOPS[i - 1], [t1, c1] = HEAT_STOPS[i], u = (v - t0) / (t1 - t0);
      return c0.map((x, k) => Math.round(x + (c1[k] - x) * u));
    }
  }
  return HEAT_STOPS[HEAT_STOPS.length - 1][1];
}
function drawHeat(c, m, h, tx, seconds, perSecond) {
  const pts = tx.length / 2, limited = seconds > 0 && seconds !== Infinity;
  const n = limited ? Math.min(pts, Math.round(seconds * perSecond)) : pts;
  if (n < 2) return;
  const N = HEAT_BINS, cell = 2 * h / N, counts = new Float32Array(N * N);
  for (let i = pts - n; i < pts; i++) {
    const gx = Math.floor((tx[2 * i] + h) / cell), gy = Math.floor((h - tx[2 * i + 1]) / cell);
    if (gx >= 0 && gx < N && gy >= 0 && gy < N) counts[gy * N + gx] += 1;
  }
  const smooth = new Float32Array(N * N);
  let max = 0;
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    let sum = 0, w = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const yy = y + dy, xx = x + dx;
      if (yy < 0 || yy >= N || xx < 0 || xx >= N) continue;
      const k = dx === 0 && dy === 0 ? 4 : dx === 0 || dy === 0 ? 2 : 1;
      sum += counts[yy * N + xx] * k; w += k;
    }
    smooth[y * N + x] = sum / w;
    if (sum / w > max) max = sum / w;
  }
  if (max <= 0) return;
  const off = sb.heatCanvas || (sb.heatCanvas = document.createElement('canvas'));
  off.width = N; off.height = N;
  const oc = off.getContext('2d'), img = oc.createImageData(N, N);
  for (let i = 0; i < N * N; i++) {
    const v = Math.sqrt(smooth[i] / max);
    if (v < 0.03) continue;
    const [r, g, b] = heatColour(v);
    img.data[4 * i] = r; img.data[4 * i + 1] = g; img.data[4 * i + 2] = b;
    img.data[4 * i + 3] = Math.round(255 * Math.min(0.85, 0.2 + 0.65 * v));
  }
  oc.putImageData(img, 0, 0);
  c.save();
  c.imageSmoothingEnabled = true;
  c.drawImage(off, m.ox - h * m.s, m.oy - h * m.s, 2 * h * m.s, 2 * h * m.s);
  c.restore();
  // Legend, bottom left.
  const lx = 8, ly = c.canvas.height - 22, lw = 96;
  const g = c.createLinearGradient(lx, 0, lx + lw, 0);
  HEAT_STOPS.forEach(([v, col]) => g.addColorStop(v, `rgb(${col[0]},${col[1]},${col[2]})`));
  c.fillStyle = 'rgba(13,13,13,0.8)'; c.fillRect(lx - 4, ly - 14, lw + 124, 26);
  c.fillStyle = g; c.fillRect(lx, ly - 4, lw, 8);
  c.font = '11px ui-sans-serif,system-ui,sans-serif'; c.fillStyle = '#ddd';
  c.fillText('잠깐', lx, ly - 6 + 0);
  c.fillText('오래 머묾', lx + lw + 6, ly + 4);
}
function decodeF32(b64) {
  const bin = atob(b64), bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}
// The server keeps the trail since the fly was put down (up to a day) and sends
// the newest 50 points with every /state. Anything older than those -- a page
// just opened, or a tab that slept -- is fetched once from /trail.
function syncTrail(s) {
  if (s.trail_epoch === undefined) return;
  if (s.trail_epoch !== sb.tEpoch) { sb.tEpoch = s.trail_epoch; sb.tx = []; sb.tTotal = 0; }
  const missing = s.trail_total - sb.tTotal, tail = s.trail || [], k = tail.length / 2;
  if (missing <= 0) return;
  if (missing <= k) {
    for (let i = k - missing; i < k; i++) sb.tx.push(tail[2 * i], tail[2 * i + 1]);
    sb.tTotal = s.trail_total;
  } else if (!sb.tFetching) {
    sb.tFetching = true;
    const epoch = sb.tEpoch;
    fetch('/trail?since=' + sb.tTotal).then(r => r.json()).then(d => {
      if (d.epoch !== epoch || epoch !== sb.tEpoch) return;
      const xy = decodeF32(d.xy);
      // Points older than the server keeps were dropped: start again from its first.
      if (d.start !== sb.tTotal) sb.tx = [];
      for (let i = 0; i < xy.length; i++) sb.tx.push(xy[i]);
      sb.tTotal = d.start + xy.length / 2;
    }).catch(() => {}).finally(() => { sb.tFetching = false; });
  }
}
const KIND_KO = {sugar:'설탕', shock:'전기', odour:'냄새', patch:'색 바닥', obstacle:'장애물'};

function mapScale(c) {
  const W = c.canvas.width, H = c.canvas.height;
  const s = Math.min(W, H) / (2 * sb.half * 1.06) * sb.view.zoom;
  return {s, ox: W / 2 - sb.view.cx * s, oy: H / 2 + sb.view.cy * s};
}
const toCanvas = (c, x, y) => { const m = mapScale(c); return [m.ox + x * m.s, m.oy - y * m.s]; };
const toWorld = (c, px, py) => { const m = mapScale(c); return [(px - m.ox) / m.s, (m.oy - py) / m.s]; };

// Half-extents on the map: walls are long boxes, odour markers a small dot.
const extentOf = it => it.kind === 'obstacle' ? [it.hx, it.hy] : it.kind === 'odour' ? [2.5, 2.5] : [it.half, it.half];
// Topmost first: what the pointer should grab when things overlap.
const HIT_ORDER = {sugar:0, odour:1, obstacle:2, shock:3, patch:4};
function hitItems(x, y) {
  return [...sb.items].sort((a, b) => HIT_ORDER[a.kind] - HIT_ORDER[b.kind]).filter(it => {
    const [hx, hy] = extentOf(it);
    return Math.abs(x - it.x) <= hx && Math.abs(y - it.y) <= hy;
  });
}
const hitItem = (x, y) => hitItems(x, y)[0];

function describe(it) {
  const at = '위치 (' + it.x.toFixed(0) + ', ' + it.y.toFixed(0) + ')';
  if (it.kind === 'sugar') return [(SUGAR_KO[it.sugar] || it.sugar) + ' ' + it.molar.toFixed(2) + ' M',
    '남은 양 ' + Math.round(it.left) + ' / ' + Math.round(it.volume) + ' nl', at];
  if (it.kind === 'shock') return ['전기 ' + it.volts.toFixed(0) + ' V', '크기 ' + (2 * it.half).toFixed(0) + ' mm', at];
  if (it.kind === 'odour') return ['냄새 ' + (ODOUR_KO[it.odour] || it.odour), '세기 ' + it.strength.toFixed(2), at];
  if (it.kind === 'patch') return [(it.colour === 'green' ? '초록' : '파랑') + ' 바닥', '크기 ' + (2 * it.half).toFixed(0) + ' mm', at];
  const w = 2 * it.hx, h = 2 * it.hy;
  return [(Math.min(w, h) <= 2.01 && Math.max(w, h) > 2.01 ? '벽 ' : '블록 ') + w.toFixed(0) + ' x ' + h.toFixed(0) + ' mm', at];
}

function drawMap(c) {
  clear(c, '#0d0d0d');
  const m = mapScale(c), h = sb.half;
  const [x0, y0] = toCanvas(c, -h, h);
  c.fillStyle = '#161618'; c.fillRect(x0, y0, 2 * h * m.s, 2 * h * m.s);
  c.strokeStyle = '#8c8c94'; c.lineWidth = Math.max(2, 2 * m.s); c.strokeRect(x0, y0, 2 * h * m.s, 2 * h * m.s);
  const box = (it, fill, stroke) => {
    const [cx, cy] = toCanvas(c, it.x, it.y), [hx, hy] = extentOf(it);
    c.fillStyle = fill; c.fillRect(cx - hx * m.s, cy - hy * m.s, 2 * hx * m.s, 2 * hy * m.s);
    if (stroke) { c.strokeStyle = stroke; c.lineWidth = 1.5; c.strokeRect(cx - hx * m.s, cy - hy * m.s, 2 * hx * m.s, 2 * hy * m.s); }
  };
  const label = (it, textStr, col) => {
    const [cx, cy] = toCanvas(c, it.x, it.y);
    c.font = '11px ui-sans-serif,system-ui,sans-serif'; c.fillStyle = col;
    c.fillText(textStr, cx - c.measureText(textStr).width / 2, cy - (extentOf(it)[1] + 3) * m.s);
  };
  const full = sb.labels === 'full', simple = sb.labels === 'simple';
  sb.items.filter(i => i.kind === 'patch').forEach(it =>
    box(it, it.colour === 'green' ? 'rgba(58,168,67,0.75)' : 'rgba(47,95,214,0.8)'));
  sb.items.filter(i => i.kind === 'odour').forEach(it => {
    const [cx, cy] = toCanvas(c, it.x, it.y), r = 22 * m.s * it.strength;
    const g = c.createRadialGradient(cx, cy, 0, cx, cy, r);
    g.addColorStop(0, (ODOUR_COL[it.odour] || '#ccc') + 'aa'); g.addColorStop(1, '#00000000');
    c.fillStyle = g; c.beginPath(); c.arc(cx, cy, r, 0, 2 * Math.PI); c.fill();
  });
  // In a replay the recorded path (5 points a second) stands in for the live trail.
  const replaying = !!sb.replayFlat, trailXY = replaying ? sb.replayFlat : sb.tx, perSecond = replaying ? 5 : 10;
  if (sb.heat) drawHeat(c, m, h, trailXY, replaying ? Infinity : sb.trailSeconds, perSecond);
  // Shock zones have no colour anywhere (user request): a dashed outline and
  // the voltage, so they cannot be mistaken for floor colour the fly sees.
  sb.items.filter(i => i.kind === 'shock').forEach(it => {
    const [cx, cy] = toCanvas(c, it.x, it.y), r = it.half * m.s;
    c.setLineDash([5, 4]); c.strokeStyle = '#ff6040'; c.lineWidth = 1.5;
    c.strokeRect(cx - r, cy - r, 2 * r, 2 * r); c.setLineDash([]);
    if (full) label(it, '⚡ ' + it.volts.toFixed(0) + ' V', '#ff8a70');
    else if (simple) label(it, '⚡', '#ff8a70');
  });
  sb.items.filter(i => i.kind === 'obstacle').forEach(it => box(it, '#6b6b72', '#9a9aa2'));
  // Trail: the last trailSeconds (10 points a second), older parts fainter. Drawn
  // as 24 paths of one alpha each, not a stroke per segment: an hour is 36,000
  // segments. With no limit the oldest part stays visible instead of fading out.
  const pts = trailXY.length / 2, forever = replaying || sb.trailSeconds === Infinity;
  const want = forever ? pts : Math.min(pts, Math.round(sb.trailSeconds * perSecond));
  if (want > 1) {
    const first = pts - want, K = 24, tx = trailXY;
    c.lineWidth = 1.2;
    for (let b = 0; b < K; b++) {
      const i0 = first + Math.floor(b * (want - 1) / K), i1 = first + Math.floor((b + 1) * (want - 1) / K);
      if (i1 <= i0) continue;
      const alpha = forever ? 0.25 + 0.5 * (b + 1) / K : 0.7 * (b + 1) / K;
      c.strokeStyle = `rgba(240,160,85,${alpha.toFixed(3)})`;
      c.beginPath();
      let lx = m.ox + tx[2 * i0] * m.s, ly = m.oy - tx[2 * i0 + 1] * m.s;
      c.moveTo(lx, ly);
      for (let i = i0 + 1; i <= i1; i++) {
        const px = m.ox + tx[2 * i] * m.s, py = m.oy - tx[2 * i + 1] * m.s;
        if (i === i1 || Math.abs(px - lx) + Math.abs(py - ly) >= 0.7) { c.lineTo(px, py); lx = px; ly = py; }
      }
      c.stroke();
    }
  }
  sb.items.filter(i => i.kind === 'sugar').forEach(it => {
    const [cx, cy] = toCanvas(c, it.x, it.y);
    c.fillStyle = SUGAR_COL[it.sugar] || '#fff';
    c.beginPath(); c.arc(cx, cy, Math.max(4, it.half * m.s), 0, 2 * Math.PI); c.fill();
    if (full) label(it, (SUGAR_KO[it.sugar] || it.sugar).split(' ')[0] + ' ' + it.molar.toFixed(2) + 'M · ' +
                    Math.round(it.left) + '/' + Math.round(it.volume) + 'nl', '#e8e8d8');
  });
  sb.items.filter(i => i.kind === 'odour').forEach(it => {
    const [cx, cy] = toCanvas(c, it.x, it.y);
    c.fillStyle = ODOUR_COL[it.odour] || '#ccc';
    c.beginPath(); c.arc(cx, cy, 3, 0, 2 * Math.PI); c.fill();
    if (full) { c.font = '11px ui-sans-serif,system-ui,sans-serif'; c.fillText(ODOUR_KO[it.odour] || it.odour, cx + 6, cy + 12); }
  });
  if (sb.selected !== null) {
    const it = sb.items.find(i => i.id === sb.selected);
    if (it) {
      const [cx, cy] = toCanvas(c, it.x, it.y), [hx, hy] = extentOf(it);
      c.setLineDash([4, 3]); c.strokeStyle = '#ffffff'; c.lineWidth = 1.2;
      c.strokeRect(cx - hx * m.s - 4, cy - hy * m.s - 4, 2 * hx * m.s + 8, 2 * hy * m.s + 8); c.setLineDash([]);
    }
  }
  if (sb.fly && sb.flyDrag) {
    const [gx, gy] = toCanvas(c, sb.flyDrag.x, sb.flyDrag.y), a = -sb.fly.yaw, L = Math.max(7, 3.2 * m.s);
    c.fillStyle = 'rgba(240,160,85,0.45)'; c.strokeStyle = '#ffffff'; c.lineWidth = 1; c.beginPath();
    c.moveTo(gx + L * Math.cos(a), gy + L * Math.sin(a));
    c.lineTo(gx + 0.6 * L * Math.cos(a + 2.5), gy + 0.6 * L * Math.sin(a + 2.5));
    c.lineTo(gx + 0.6 * L * Math.cos(a - 2.5), gy + 0.6 * L * Math.sin(a - 2.5));
    c.closePath(); c.fill(); c.stroke();
  }
  if (sb.fly) {
    const [fx, fy] = toCanvas(c, sb.fly.x, sb.fly.y), a = -sb.fly.yaw, L = Math.max(7, 3.2 * m.s);
    c.fillStyle = '#f0a055'; c.beginPath();
    c.moveTo(fx + L * Math.cos(a), fy + L * Math.sin(a));
    c.lineTo(fx + 0.6 * L * Math.cos(a + 2.5), fy + 0.6 * L * Math.sin(a + 2.5));
    c.lineTo(fx + 0.6 * L * Math.cos(a - 2.5), fy + 0.6 * L * Math.sin(a - 2.5));
    c.closePath(); c.fill();
  }
  // Details for the item under the pointer, whatever the label setting.
  const hovered = sb.hover && !sb.drag && !sb.pan ? sb.items.find(i => i.id === sb.hover.id) : null;
  if (hovered && sb.labels !== 'none') {
    const lines = describe(hovered);
    c.font = '12px ui-sans-serif,system-ui,sans-serif';
    const w = Math.max(...lines.map(l => c.measureText(l).width)) + 14, hh = lines.length * 16 + 8;
    let bx = sb.hover.px + 14, by = sb.hover.py + 14;
    if (bx + w > c.canvas.width) bx = sb.hover.px - w - 10;
    if (by + hh > c.canvas.height) by = sb.hover.py - hh - 10;
    c.fillStyle = 'rgba(18,18,22,0.93)'; c.fillRect(bx, by, w, hh);
    c.strokeStyle = '#5a5a60'; c.lineWidth = 1; c.strokeRect(bx, by, w, hh);
    c.fillStyle = '#eeeeee'; lines.forEach((l, i) => c.fillText(l, bx + 7, by + 17 + i * 16));
  }
}

// Learned value per odour and colour, one bar each around zero.
function learnBars(c, tel) {
  clear(c);
  const rows = [
    ...Object.entries(tel.odour_valence || {}).map(([k, v]) => ['냄새 ' + (ODOUR_KO[k] || k), v, ODOUR_COL[k]]),
    ...Object.entries(tel.colour_valence || {}).map(([k, v]) => [k === 'blue' ? '색 파랑' : '색 초록', v,
      k === 'blue' ? '#5b8fe8' : '#4fbf5a']),
  ];
  const W = c.canvas.width, H = c.canvas.height, left = 86, right = 46;
  const mid = left + (W - left - right) / 2, span = (W - left - right) / 2;
  const rh = Math.max(12, (H - 16) / Math.max(1, rows.length));
  c.strokeStyle = '#444'; c.beginPath(); c.moveTo(mid, 4); c.lineTo(mid, H - 12); c.stroke();
  rows.forEach(([name, v, col], i) => {
    const y = 6 + i * rh, clipped = Math.max(-1, Math.min(1, v));
    c.fillStyle = v >= 0 ? '#3f7f4a' : '#8a3a3a';
    c.fillRect(Math.min(mid, mid + clipped * span), y + 2, Math.abs(clipped * span), rh - 6);
    text(c, name, 4, y + rh / 2 + 3, col || '#ccc', 11);
    text(c, (v >= 0 ? '+' : '') + v.toFixed(2), W - right + 4, y + rh / 2 + 3, '#d0d0d0', 11);
  });
  text(c, '−1', left - 2, H - 1, '#6e6e6e', 9); text(c, '0', mid - 3, H - 1, '#6e6e6e', 9);
  text(c, '+1', W - right - 12, H - 1, '#6e6e6e', 9);
}

function obstacleParams() {
  const len = parseFloat(p_length.value);
  if (p_shape.value === 'wall_h') return {hx: len / 2, hy: 1};
  if (p_shape.value === 'wall_v') return {hx: 1, hy: len / 2};
  return {hx: len / 2, hy: len / 2};
}
function currentParams(kind) {
  if (kind === 'sugar') return {sugar: p_sugar.value, molar: parseFloat(p_molar.value),
                                volume: parseFloat(p_volume.value)};
  if (kind === 'shock') return {volts: parseFloat(p_volts.value), half: parseFloat(p_half_shock.value)};
  if (kind === 'odour') return {odour: p_odour.value, strength: parseFloat(p_strength.value)};
  if (kind === 'patch') return {colour: p_colour.value, half: parseFloat(p_half_patch.value)};
  if (kind === 'obstacle') return obstacleParams();
  return {};
}
const labelsFor = {
  p_molar: v => v.toFixed(2) + ' M', p_volume: v => v.toFixed(0) + ' nl', p_volts: v => v.toFixed(0) + ' V',
  p_half_shock: v => (2 * v).toFixed(0) + ' mm', p_strength: v => v.toFixed(2),
  p_half_patch: v => (2 * v).toFixed(0) + ' mm', p_length: v => v.toFixed(0) + ' mm',
};
const refreshLabel = id => { const out = document.getElementById(id + '_v');
  if (out && labelsFor[id]) out.textContent = labelsFor[id](parseFloat(document.getElementById(id).value)); };

// Put a selected item's own values into the palette, so editing one setting
// changes only that setting.
function loadParams(it) {
  const set = (id, v) => { const el = document.getElementById(id); el.value = v; refreshLabel(id); };
  if (it.kind === 'sugar') { p_sugar.value = it.sugar; set('p_molar', it.molar); set('p_volume', it.volume); }
  if (it.kind === 'shock') { set('p_volts', it.volts); set('p_half_shock', it.half); }
  if (it.kind === 'odour') { p_odour.value = it.odour; set('p_strength', it.strength); }
  if (it.kind === 'patch') { p_colour.value = it.colour; set('p_half_patch', it.half); }
  if (it.kind === 'obstacle') {
    p_shape.value = it.hx === it.hy ? 'block' : it.hx > it.hy ? 'wall_h' : 'wall_v';
    set('p_length', 2 * Math.max(it.hx, it.hy));
  }
}
function showOpts() {
  const sel = sb.items.find(i => i.id === sb.selected);
  // One row at a time: the selected item's settings in select mode, else the
  // tool's. (toggle(name, undefined) flips instead of clearing: with nothing
  // selected every row once opened together and the map shrank to 175 px.)
  const kind = sb.tool === 'select' ? (sel ? sel.kind : 'select') : sb.tool;
  document.querySelectorAll('.palette .opts').forEach(o => o.classList.toggle('on', o.dataset.for === kind));
}
function selectItem(it) {
  sb.selected = it ? it.id : null;
  if (it) loadParams(it);
  showOpts();
}
function setTool(tool) {
  sb.tool = tool;
  document.querySelectorAll('.palette .tool').forEach(x => x.classList.toggle('on', x.dataset.tool === tool));
  if (tool !== 'select') sb.selected = null;
  showOpts();
}
function presetOptions(s, first) {
  const groups = (s.palette.preset_groups || [['', Object.keys(s.palette.presets)]]);
  return first + groups.map(([g, names]) => `<optgroup label="${esc(g)}">` +
    names.map(k => `<option value="${k}">${esc(s.palette.presets[k])}</option>`).join('') + '</optgroup>').join('');
}

function csvCell(v) {
  const sv = String(v ?? '');
  return /[",]/.test(sv) || sv.includes(String.fromCharCode(10)) ? '"' + sv.split('"').join('""') + '"' : sv;
}
function downloadCsv(rows) {
  if (!rows.length) return;
  const keys = Object.keys(rows[0]);
  const lines = [keys.map(csvCell).join(','), ...rows.map(r => keys.map(k => csvCell(r[k])).join(','))];
  const blob = new Blob([String.fromCharCode(0xFEFF) + lines.join(String.fromCharCode(13, 10))], {type:'text/csv;charset=utf-8'});
  const a = document.createElement('a'), d = new Date(), pad2 = x => String(x).padStart(2, '0');
  a.href = URL.createObjectURL(blob);
  a.download = `초파리_실험결과_${d.getFullYear()}${pad2(d.getMonth() + 1)}${pad2(d.getDate())}_${pad2(d.getHours())}${pad2(d.getMinutes())}.csv`;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
function setSbTab(tab) {
  sb.tab = tab;
  document.querySelectorAll('#sbtabs button').forEach(b => b.classList.toggle('on', b.dataset.tab === tab));
  intro.style.display = tab === 'explain' ? '' : 'none';
  explainnow.style.display = tab === 'explain' ? '' : 'none';
  trialpane.style.display = tab === 'trial' ? '' : 'none';
  exppane.style.display = tab === 'exp' ? '' : 'none';
  if (tab === 'exp') labRefresh(true);
  fitCanvases();
}

function setupPalette(s) {
  if (sb.paletteReady || !s.palette) return;
  sb.paletteReady = true;
  p_sugar.innerHTML = s.palette.sugars.map(k => `<option value="${k}">${SUGAR_KO[k] || k}</option>`).join('');
  p_odour.innerHTML = s.palette.odours.map(k => `<option value="${k}">${ODOUR_KO[k] || k}</option>`).join('');
  p_preset.innerHTML = presetOptions(s, '<option value="">고르기...</option>');
  t_preset.innerHTML = presetOptions(s, '<option value="">지금 방 그대로</option>');
  document.querySelectorAll('.palette .tool').forEach(b => b.onclick = () => setTool(b.dataset.tool));
  // A setting changed while an item is selected edits that item, that setting only.
  const edit = (el, build) => {
    const handler = () => {
      refreshLabel(el.id);
      const it = sb.items.find(i => i.id === sb.selected);
      if (sb.tool === 'select' && it) {
        const change = build(it);
        if (!el._gesture) el._gesture = 'slide' + Date.now() + Math.random();
        if (change) sbPost({op:'update', id: it.id, ...change, gesture: el._gesture});
      }
    };
    el.addEventListener(el.tagName === 'SELECT' ? 'change' : 'input', handler);
    // A slider's pull ends on release; a select's single change is its own record.
    el.addEventListener('change', () => { el._gesture = null; });
  };
  const only = (kind, fn) => it => it.kind === kind ? fn() : null;
  edit(p_sugar, only('sugar', () => ({sugar: p_sugar.value})));
  edit(p_molar, only('sugar', () => ({molar: parseFloat(p_molar.value)})));
  edit(p_volume, only('sugar', () => ({volume: parseFloat(p_volume.value)})));
  edit(p_volts, only('shock', () => ({volts: parseFloat(p_volts.value)})));
  edit(p_half_shock, only('shock', () => ({half: parseFloat(p_half_shock.value)})));
  edit(p_odour, only('odour', () => ({odour: p_odour.value})));
  edit(p_strength, only('odour', () => ({strength: parseFloat(p_strength.value)})));
  edit(p_colour, only('patch', () => ({colour: p_colour.value})));
  edit(p_half_patch, only('patch', () => ({half: parseFloat(p_half_patch.value)})));
  p_shape.addEventListener('change', () => {
    // A wall starts at a useful length, a block at the default 8 mm.
    if (p_shape.value === 'block' && parseFloat(p_length.value) > 20) p_length.value = 8;
    if (p_shape.value !== 'block' && parseFloat(p_length.value) < 16) p_length.value = 30;
  });
  edit(p_shape, only('obstacle', obstacleParams));
  edit(p_length, only('obstacle', obstacleParams));
  Object.keys(labelsFor).forEach(refreshLabel);
  p_labels.onchange = () => { sb.labels = p_labels.value; drawMap(document.getElementById('c_map').getContext('2d')); };
  p_heat.onclick = () => { sb.heat = !sb.heat; p_heat.classList.toggle('on', sb.heat);
    drawMap(document.getElementById('c_map').getContext('2d')); };
  p_trail.addEventListener('input', () => { sb.trailSeconds = TRAIL_STEPS[parseInt(p_trail.value)];
    p_trail_v.textContent = trailLabel(sb.trailSeconds);
    drawMap(document.getElementById('c_map').getContext('2d')); });
  p_preset.onchange = () => { if (p_preset.value) sbPost({op:'preset', name: p_preset.value}); selectItem(null); };
  b_deplete.onclick = () => sbPost({op:'sugar_depletes', value: !b_deplete.classList.contains('on')});
  b_clear.onclick = () => { sbPost({op:'clear'}); selectItem(null); };
  b_deselect.onclick = () => selectItem(null);
  b_undo.onclick = () => sbPost({op:'undo'});
  b_redo.onclick = () => sbPost({op:'redo'});
  addEventListener('keydown', e => {
    if (laidOut !== 'sandbox' || lab.open || (e.target.closest && e.target.closest('input, select, textarea'))) return;
    if (e.key === 'Escape') { selectItem(null); drawMap(document.getElementById('c_map').getContext('2d')); return; }
    const k = e.key.toLowerCase();
    if ((e.ctrlKey || e.metaKey) && (k === 'z' || k === 'y')) {
      e.preventDefault();
      sbPost({op: k === 'y' || e.shiftKey ? 'redo' : 'undo'});
    }
  });
  p_hunger.addEventListener('input', () => { sb.hungerDrag = true; p_hunger_v.textContent = parseFloat(p_hunger.value).toFixed(2);
                                            sbPost({op:'hunger', value: parseFloat(p_hunger.value)}); });
  p_hunger.addEventListener('change', () => { sb.hungerDrag = false; });

  // Trials.
  document.querySelectorAll('#sbtabs button').forEach(b => b.onclick = () => {
    setSbTab(b.dataset.tab);
    if (b.dataset.tab === 'exp') labOpen();
  });
  t_hunger.addEventListener('input', () => { t_hunger_v.textContent = parseFloat(t_hunger.value).toFixed(2); });
  t_start.onclick = () => sbPost({op:'trial_start', label: t_label.value, preset: t_preset.value,
    hunger: parseFloat(t_hunger.value), seconds: parseFloat(t_seconds.value),
    new_fly: t_newfly.checked, random_heading: t_random.checked});
  t_stop.onclick = () => sbPost({op:'trial_stop'});
  t_csv.onclick = () => downloadCsv(sb.rows || []);
  t_clear.onclick = () => { if (confirm('결과표를 모두 지울까요? (out/sandbox/trials.csv 파일에는 남아 있습니다)')) sbPost({op:'trial_clear'}); };

  const cv = document.getElementById('c_map');
  const canvasAt = e => { const r = cv.getBoundingClientRect();
    return [(e.clientX - r.left) * cv.width / r.width, (e.clientY - r.top) * cv.height / r.height]; };
  const worldAt = e => toWorld(cv.getContext('2d'), ...canvasAt(e));
  cv.addEventListener('contextmenu', e => e.preventDefault());
  cv.addEventListener('pointerdown', e => {
    e.preventDefault();
    const [x, y] = worldAt(e), hits = hitItems(x, y), hit = hits[0];
    if (e.button === 2) {
      // Right button: drag to slide the map. Released without moving, it is
      // a right-click and deletes what is under the pointer.
      sb.pan = {px: e.clientX, py: e.clientY, cx: sb.view.cx, cy: sb.view.cy, moved: false, hit};
      cv.setPointerCapture(e.pointerId);
      return;
    }
    if (sb.tool === 'fly') {
      sbPost({op:'fly_move', x, y, yaw: p_fly_yaw.value === 'keep' ? null : parseFloat(p_fly_yaw.value) * Math.PI / 180});
      return;
    }
    if (sb.tool === 'select' && sb.fly) {
      // The fly is drawn on top, so it is grabbed before anything under it.
      const m = mapScale(cv.getContext('2d')), r = cv.getBoundingClientRect();
      const reach = Math.max(3, 12 * cv.width / r.width / m.s);
      if (Math.hypot(x - sb.fly.x, y - sb.fly.y) < reach) {
        sb.flyDrag = {x: sb.fly.x, y: sb.fly.y, dx: sb.fly.x - x, dy: sb.fly.y - y, moved: false};
        cv.setPointerCapture(e.pointerId);
        drawMap(cv.getContext('2d'));
        return;
      }
    }
    if (sb.tool === 'erase') {
      if (hit) { sbPost({op:'remove', id: hit.id}); sb.items = sb.items.filter(i => i !== hit);
                 if (sb.selected === hit.id) selectItem(null); }
      return;
    }
    if (sb.tool === 'select') {
      if (!hits.length) { selectItem(null); sb.cycle = null; drawMap(cv.getContext('2d')); return; }
      // Clicking the same spot again steps to the next item underneath.
      let pick = hits[0];
      if (sb.cycle && Math.hypot(x - sb.cycle.x, y - sb.cycle.y) < 1.5) {
        const k = hits.findIndex(i => i.id === sb.selected);
        if (k >= 0) pick = hits[(k + 1) % hits.length];
      }
      sb.cycle = {x, y};
      selectItem(pick);
      sb.drag = {id: pick.id, dx: pick.x - x, dy: pick.y - y, gesture: 'drag' + Date.now() + Math.random()};
      cv.setPointerCapture(e.pointerId);
      drawMap(cv.getContext('2d'));
      return;
    }
    // Placing tools always place, even on top of something else.
    sbPost({op:'place', kind: sb.tool, x, y, ...currentParams(sb.tool)});
  });
  cv.addEventListener('pointermove', e => {
    if (sb.pan) {
      const r = cv.getBoundingClientRect(), s = mapScale(cv.getContext('2d')).s * r.width / cv.width;
      const dx = e.clientX - sb.pan.px, dy = e.clientY - sb.pan.py;
      if (Math.abs(dx) + Math.abs(dy) > 4) sb.pan.moved = true;
      sb.view.cx = sb.pan.cx - dx / s;
      sb.view.cy = sb.pan.cy + dy / s;
      drawMap(cv.getContext('2d'));
      return;
    }
    if (sb.flyDrag) {
      const [x, y] = worldAt(e);
      sb.flyDrag.x = x + sb.flyDrag.dx; sb.flyDrag.y = y + sb.flyDrag.dy; sb.flyDrag.moved = true;
      drawMap(cv.getContext('2d'));
      return;
    }
    if (!sb.drag) {
      const [px, py] = canvasAt(e), it = hitItem(...worldAt(e));
      const next = it ? {id: it.id, px, py} : null;
      if ((next && next.id) !== (sb.hover && sb.hover.id) || next) { sb.hover = next; drawMap(cv.getContext('2d')); }
      return;
    }
    const [x, y] = worldAt(e), it = sb.items.find(i => i.id === sb.drag.id);
    if (!it) return;
    it.x = x + sb.drag.dx; it.y = y + sb.drag.dy;
    const now = performance.now();
    if (now - sb.lastMove > 60) { sb.lastMove = now; sbPost({op:'move', id: it.id, x: it.x, y: it.y, gesture: sb.drag.gesture}); }
    drawMap(cv.getContext('2d'));
  });
  cv.addEventListener('pointerleave', () => { if (sb.hover) { sb.hover = null; drawMap(cv.getContext('2d')); } });
  const endDrag = () => {
    if (sb.pan) {
      const hit = sb.pan.hit;
      if (!sb.pan.moved && hit) {
        sbPost({op:'remove', id: hit.id}); sb.items = sb.items.filter(i => i !== hit);
        if (sb.selected === hit.id) selectItem(null);
      }
      sb.pan = null;
      return;
    }
    if (sb.flyDrag) {
      // One command at release: every move settles the body for 0.2 s.
      if (sb.flyDrag.moved) sbPost({op:'fly_move', x: sb.flyDrag.x, y: sb.flyDrag.y, yaw: null});
      sb.flyDrag = null;
      drawMap(cv.getContext('2d'));
      return;
    }
    if (!sb.drag) return;
    const it = sb.items.find(i => i.id === sb.drag.id);
    if (it) sbPost({op:'move', id: it.id, x: it.x, y: it.y, gesture: sb.drag.gesture});
    sb.drag = null;
  };
  cv.addEventListener('pointerup', endDrag);
  cv.addEventListener('pointercancel', endDrag);
  // Wheel zooms about the pointer: the world point under it stays put.
  cv.addEventListener('wheel', e => {
    e.preventDefault();
    const [wx, wy] = worldAt(e);
    const zoom = Math.max(1, Math.min(8, sb.view.zoom * (e.deltaY > 0 ? 1 / 1.15 : 1.15)));
    const ratio = sb.view.zoom / zoom;
    sb.view.cx = wx - (wx - sb.view.cx) * ratio;
    sb.view.cy = wy - (wy - sb.view.cy) * ratio;
    sb.view.zoom = zoom;
    if (zoom === 1) { sb.view.cx = 0; sb.view.cy = 0; }
    drawMap(cv.getContext('2d'));
  }, {passive:false});
  cv.addEventListener('dblclick', e => {
    const [x, y] = worldAt(e);
    if (!hitItem(x, y)) { sb.view = {cx:0, cy:0, zoom:1}; drawMap(cv.getContext('2d')); }
  });
  setTool(sb.tool);
}


// ---------- A/B lab: choose a question, predict, run in the background, read the result ----------
// A dialog over the whole window in three steps (see the CSS). Replays play in
// the lab's own video; the replay bar moves in with it and back out on close.
const lab = {open:false, view:'home', sets:null, byKey:{}, rooms:{}, list:[], loaded:false, listSig:'', miniHtml:'',
             headHtml:'', detail:null, key:null, choice:null, flies:8, id:null, starting:null, showAll:false,
             homeBuilt:false, busy:false, lastPoll:0, replay:null};
const STATE_KO = {starting:'준비 중', running:'진행 중', reporting:'정리 중', done:'끝남', stopped:'멈춤', failed:'실패'};
const ROLE_KO = {train:'훈련', test:'시험', repeat:'반복'};
const VERDICT_CLASS = {supported:'good', leaning:'lean', opposite:'bad'};
const UNITS = ['초', '%', 'nl', 'mm', 'mm/s'];
const fmtNum = v => Number.isInteger(v) ? String(v) : Math.abs(v) >= 10 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(1) : v.toFixed(2);
const isRunning = e => ['starting', 'running', 'reporting'].includes(e.state);
const donePct = e => e.total ? Math.round(100 * e.done / e.total) : 0;
const progressBar = e => '<span class="progress"><i style="width:' + donePct(e) + '%"></i></span>';
function unitOf(column) {
  const open = column.lastIndexOf('(');
  const unit = open >= 0 ? column.slice(open + 1, -1) : '';
  return UNITS.includes(unit) ? unit : '';
}
function timeLeft(seconds) {
  if (seconds === null || seconds === undefined) return '';
  return seconds < 45 ? '곧 끝남' : '약 ' + Math.max(1, Math.round(seconds / 60)) + '분 남음';
}
function etaMinutes(set, flies) {
  const jobs = 2 * flies;
  return Math.max(1, Math.round(Math.ceil(jobs / lab.sets.workers) * set.seconds_per_fly / lab.sets.speed / 60));
}

function labOpen(where, arg) {
  if (!lab.open) {
    lab.open = true;
    labroot.style.display = '';
    // Below 1100 px the page itself scrolls, and its scrollbar showed beside the lab's.
    document.body.style.overflow = 'hidden';
    // One video stream per tab: each holds a connection for as long as it
    // plays, and a browser allows six to this server across all of its tabs.
    view.removeAttribute('src');
  }
  labGo(where || lab.view, arg);
  labRefresh(true);
}
function labClose() {
  if (!lab.open) return;
  lab.open = false;
  labroot.style.display = 'none';
  document.body.style.overflow = '';
  labview.removeAttribute('src');
  view.after(replaybar);
  view.src = '/stream';
}
function labGo(where, arg) {
  if (where === 'set' && arg && arg !== lab.key) {
    lab.key = arg; lab.choice = null; lab.flies = (lab.byKey[arg] || {}).flies || 8;
  }
  if (where === 'result' && arg && arg !== lab.id) { lab.id = arg; lab.detail = null; }
  if ((where === 'set' && !lab.key) || (where === 'result' && !lab.id)) where = 'home';
  lab.view = where;
  lab_home.style.display = where === 'home' ? '' : 'none';
  lab_set.style.display = where === 'set' ? '' : 'none';
  lab_result.style.display = where === 'result' ? '' : 'none';
  labsteps.querySelectorAll('li').forEach(li => {
    const step = li.dataset.step;
    li.classList.toggle('now', step === where);
    li.classList.toggle('done', step !== where &&
      (step === 'home' || (step === 'set' && !!lab.key) || (step === 'result' && !!lab.id)));
  });
  if (where === 'result') {
    labvideo.appendChild(replaybar);
    if (!labview.getAttribute('src')) labview.src = '/stream';
    lab.headHtml = '';
    renderLabResult();
  } else {
    view.after(replaybar);
    labview.removeAttribute('src');
  }
  if (where === 'home') renderLabHome();
  if (where === 'set') renderLabSet();
  labbody.scrollTop = 0;
}

async function labRefresh(force) {
  if (lab.busy) return;
  lab.busy = true;
  lab.lastPoll = Date.now();
  try {
    if (!lab.sets) {
      const sets = await (await fetch('/sets')).json();
      sets.groups.forEach(g => g.sets.forEach(s => { lab.byKey[s.key] = s; }));
      lab.rooms = sets.rooms || {};
      lab.sets = sets;
      if (lab.open && lab.view === 'home') renderLabHome();
      if (lab.open && lab.view === 'set') renderLabSet();
    }
    const list = await (await fetch('/experiments')).json();
    lab.list = list;
    lab.loaded = true;
    if (lab.starting) {
      const fresh = list.find(e => e.set === lab.starting.key && !lab.starting.before.has(e.id));
      if (fresh) { lab.starting = null; labGo('result', fresh.id); force = true; }
      else if (Date.now() - lab.starting.t > 15000) { lab.starting = null; labStartFailed(); }
    }
    const sig = JSON.stringify(list.map(e => [e.id, e.state, e.done, e.verdict_code, e.stop_requested]));
    if (force || sig !== lab.listSig) {
      lab.listSig = sig;
      if (lab.open && lab.view === 'home') renderLabExps();
      if (lab.open && lab.view === 'set') renderLabBusy();
    }
    renderLabMini();
    renderSab();
    if (lab.open && lab.view === 'result' && lab.id) {
      const now = list.find(e => e.id === lab.id), d = lab.detail;
      if (force || !d || d.id !== lab.id || !now || d.done !== now.done || d.state !== now.state) {
        lab.detail = await (await fetch('/experiment?id=' + encodeURIComponent(lab.id))).json();
        renderLabResult();
      } else {
        renderLabHead();
      }
    }
  } catch (e) {
  } finally {
    lab.busy = false;
  }
}
setInterval(() => {
  if (typeof laidOut === 'undefined' || laidOut !== 'sandbox') return;
  // Every 2 s with the lab open; every 4 s otherwise, for the progress under the video.
  if (lab.open || Date.now() - lab.lastPoll > 4000) labRefresh(false);
}, 2000);

function renderSab() {
  const running = lab.list.filter(isRunning);
  sab.textContent = running.length ? running.map(e => e.done + '/' + e.total).join(', ') + ' 진행 중' : '돌고 있지 않음';
  sablab.classList.toggle('on', running.length > 0);
}
function renderLabMini() {
  const rows = lab.list.slice(0, 3);
  const html = !lab.loaded ? '' : rows.length ? rows.map(e =>
      '<div class="row"><span class="badge ' + e.state + '">' + (STATE_KO[e.state] || e.state) + '</span>' + progressBar(e) +
      '<span>' + esc(e.title) + ' · ' + e.done + '/' + e.total +
      (isRunning(e) ? ' · ' + timeLeft(e.eta_s) : e.verdict ? ' · ' + esc(e.verdict) : '') + '</span>' +
      '<button class="linkbtn" data-open="' + esc(e.id) + '">결과 보기</button></div>').join('')
    : '<span class="lbl">아직 실험 기록이 없습니다. 실험실에서 질문을 고르면 시작할 수 있습니다.</span>';
  if (html !== lab.miniHtml) { lab.miniHtml = html; labmini.innerHTML = html; }
}

// ---- step 1: questions, and the experiments already run
function renderLabHome() {
  if (!lab.sets) { lab_home.innerHTML = '<p class="lbl">질문을 불러오는 중…</p>'; lab.homeBuilt = false; return; }
  if (!lab.homeBuilt) {
    lab.homeBuilt = true;
    lab_home.innerHTML =
      '<p class="labintro"><b>A/B 실험</b>은 한 가지만 다른 두 조건 A와 B에 쌍둥이 초파리를 여러 마리씩 넣어 배경에서 돌리고, ' +
      '두 무리의 차이로 내 가설이 맞았는지 판정합니다. 실험이 도는 동안에도 지금 초파리는 그대로 볼 수 있습니다.</p>' +
      '<div id="lab_exps"></div>' +
      '<h3>① 궁금한 질문을 고르세요</h3>' +
      lab.sets.groups.map(g => '<div class="labgroup"><div class="labgrouptitle">' + esc(g.group) + '</div><div class="qgrid">' +
        g.sets.map(s => '<button class="qcard" data-set="' + esc(s.key) + '">' +
          '<span class="qthumb">' + (lab.rooms[s.conditions[0].steps[0].thumb] || '') + '</span>' +
          '<span class="qtext"><b>' + esc(s.title) + '</b>' +
          '<span><i class="tagA">A</i>' + esc(s.a) + '</span><span><i class="tagB">B</i>' + esc(s.b) + '</span>' +
          '<small>재는 것: ' + esc(s.metric) + ' · 약 ' + etaMinutes(s, s.flies) + '분</small>' +
          '<small class="tried" data-tried="' + esc(s.key) + '"></small></span></button>').join('') +
        '</div></div>').join('');
  }
  renderLabExps();
}
function renderLabExps() {
  const box = document.getElementById('lab_exps');
  if (!box) return;
  const shown = lab.showAll ? lab.list : lab.list.slice(0, 6);
  box.innerHTML = !lab.list.length ? '' :
    '<h3>내 실험 <small class="lbl">눌러서 결과와 3D 다시 보기</small></h3><div class="labexps">' +
    shown.map(e => '<button class="expcard" data-open="' + esc(e.id) + '">' +
      '<span class="expcardtop"><span class="badge ' + e.state + '">' + (STATE_KO[e.state] || e.state) + '</span>' +
      '<small>' + esc((e.created || '').slice(5, 16)) + '</small></span>' +
      '<b>' + esc(e.title) + '</b>' + progressBar(e) +
      '<small>' + e.done + '/' + e.total + ' 시행 · 조건마다 파리 ' + e.flies + '마리' +
      (isRunning(e) ? ' · ' + timeLeft(e.eta_s) : '') + '</small>' +
      (e.verdict ? '<small class="vmini ' + (VERDICT_CLASS[e.verdict_code] || '') + '">' + esc(e.verdict) + '</small>' : '') +
      '</button>').join('') + '</div>' +
    (lab.list.length > 6 ? '<button class="linkbtn" id="lab_showall">' +
      (lab.showAll ? '최근 6개만 보기' : '모두 보기 (' + lab.list.length + '개)') + '</button>' : '');
  const tried = {};
  lab.list.forEach(e => { tried[e.set] = (tried[e.set] || 0) + 1; });
  lab_home.querySelectorAll('[data-tried]').forEach(el => {
    const n = tried[el.dataset.tried];
    el.textContent = n ? '해 본 실험 ' + n + '번' : '';
  });
}

// ---- step 2: what A and B are, the hypothesis, and the start button
function labSeq(c) {
  const parts = [];
  c.steps.forEach((st, i) => {
    if (i) parts.push('<span class="seqarrow">→</span>');
    parts.push('<div class="seqstep"><div class="thumb">' + (lab.rooms[st.thumb] || '') + '</div>' +
      '<b>' + esc(st.role_ko) + '</b> ' + esc(st.seconds_label) + (st.repeats > 1 ? ' × ' + st.repeats + '번' : '') +
      (st.gap_label ? '<div class="lbl">사이사이 ' + esc(st.gap_label) + ' 쉬기</div>' : '') +
      '<div class="lbl">' + esc(st.room_label) + '</div></div>');
    if (st.rest_label) parts.push('<span class="seqarrow">→</span><span class="seqrest">' + esc(st.rest_label) + ' 쉬기</span>');
  });
  return '<div class="seq">' + parts.join('') + '</div>';
}
function renderLabSet() {
  const s = lab.byKey[lab.key];
  if (!s) { lab_set.innerHTML = '<p class="lbl">불러오는 중…</p>'; return; }
  const card = (c, i) => {
    const tag = i ? 'B' : 'A';
    const notes = [];
    if (c.changes.length) notes.push('<b>' + esc(c.changes.join(', ')) + '</b>');
    notes.push(c.same_fly ? '같은 파리가 모든 시행을 이어서 함' : '시행마다 새 파리');
    notes.push('시작 배고픔 ' + c.hunger.toFixed(2));
    return '<div class="labcard cond ' + tag.toLowerCase() + '">' +
      '<div class="labcardtitle"><i class="tag' + tag + '">' + tag + '</i>' + esc(i ? s.b : s.a) + '</div>' +
      labSeq(c) + '<p class="condnote">' + notes.join(' · ') + '</p>' +
      '<button class="linkbtn" data-preview="' + i + '" title="지금 초파리가 있는 방을 이 조건의 첫 방으로 꾸밉니다. ' +
      '실험실이 닫히고, Ctrl+Z로 되돌릴 수 있습니다.">이 방을 지금 초파리 방에 꾸며 보기</button></div>';
  };
  lab_set.innerHTML =
    '<button class="linkbtn" data-go="home">← 질문 다시 고르기</button>' +
    '<div class="sethead"><div class="lbl">' + esc(s.group) + '</div><h2>' + esc(s.title) + '</h2><p>' + esc(s.question) + '</p></div>' +
    '<div class="setcols">' + s.conditions.map(card).join('') + '</div>' +
    '<div class="labcard"><dl class="vars">' +
      '<dt>바꾸는 것<span class="lbl">조작 변인</span></dt><dd>' + esc(s.changed) + '</dd>' +
      '<dt>재는 것<span class="lbl">종속 변인</span></dt><dd><b>' + esc(s.metric) + '</b> — ' + esc(s.meaning) +
        ' <span class="lbl">' + esc(s.phase) + '의 평균으로 A와 B를 비교합니다.</span></dd>' +
      '<dt>같게 하는 것<span class="lbl">통제 변인</span></dt><dd>' + s.held.map(esc).join(' · ') + '</dd>' +
    '</dl></div>' +
    '<div class="labcard labstep"><div class="labcardtitle">② 내 가설 고르기 <small>실험하기 전에 결과를 예상하세요. ' +
      '실험이 끝나면 이 예상이 맞았는지 판정합니다.</small></div>' +
      s.choices.map(c => '<label class="choice' + (lab.choice === c.value ? ' on' : '') + '"><input type="radio" name="labchoice" value="' +
        esc(c.value) + '"' + (lab.choice === c.value ? ' checked' : '') + '><span>' + esc(c.text) + '</span></label>').join('') +
    '</div>' +
    '<div class="labcard labstep"><div class="labcardtitle">③ 실험 시작</div>' +
      '<div class="runrow"><span>조건마다 파리</span><input type="range" id="lab_flies" min="3" max="30" step="1" value="' + lab.flies + '">' +
      '<b id="lab_flies_v"></b></div>' +
      '<p class="lbl">파리가 많을수록 결과를 믿을 만하지만 오래 걸립니다. 같은 번호의 A·B 파리는 뇌 배선과 출발 방향이 같은 ' +
      '쌍둥이이고, 쌍이 3쌍보다 적으면 판정하지 않습니다.</p>' +
      '<div id="lab_busy" class="warn"></div>' +
      '<div class="runrow"><button id="lab_start" class="primary big">▶ 실험 시작</button><span id="lab_hint" class="lbl"></span></div>' +
    '</div>' +
    '<details class="labmore"><summary>배경 지식: 실제 연구에서는</summary><p>' + esc(s.basis) + '</p></details>';
  renderLabRun();
  renderLabBusy();
}
function renderLabRun() {
  const s = lab.byKey[lab.key], out = document.getElementById('lab_flies_v');
  if (!s || !out) return;
  out.textContent = lab.flies + '마리씩, 모두 ' + 2 * lab.flies + '마리 · 약 ' + etaMinutes(s, lab.flies) + '분';
  document.getElementById('lab_start').disabled = !lab.choice || !!lab.starting;
  document.getElementById('lab_hint').textContent = lab.starting ? '실험을 준비하고 있습니다…'
    : !lab.choice ? '먼저 위에서 가설을 고르세요' : '시작하면 결과 화면으로 넘어가고, 실험은 배경에서 돕니다.';
}
function renderLabBusy() {
  const box = document.getElementById('lab_busy');
  if (!box) return;
  const n = lab.list.filter(isRunning).length;
  box.textContent = n ? '지금 실험 ' + n + '개가 돌고 있습니다. CPU를 나눠 쓰므로 더 오래 걸립니다.' : '';
}
function labStart() {
  if (!lab.key || !lab.choice || lab.starting) return;
  // The server answers before the run exists, so the new run is recognised
  // as the first experiment of this set that was not in the list before.
  lab.starting = {key: lab.key, before: new Set(lab.list.map(e => e.id)), t: Date.now()};
  renderLabRun();
  sbPost({op:'exp_start', set: lab.key, flies: lab.flies, prediction: lab.choice});
  setTimeout(() => labRefresh(false), 700);
}
function labStartFailed() {
  renderLabRun();
  const hint = document.getElementById('lab_hint');
  if (hint) hint.textContent = '실험이 시작되지 않았습니다. 같은 실험을 방금 시작했다면 10초 뒤에 다시 눌러 보세요.';
}

// ---- step 3: progress, verdict, chart, and replays
function renderLabResult() {
  renderLabHead();
  renderLabVerdict();
  renderLabTrials();
  renderLabGrid();
  const d = lab.detail;
  lr_why.textContent = d ? d.why : '';
  lr_basis.textContent = d ? d.basis : '';
}
function renderLabHead() {
  const d = lab.detail;
  let html;
  if (!d) {
    html = '<p class="lbl">실험 기록을 불러오는 중…</p>';
  } else {
    const e = lab.list.find(x => x.id === d.id) || d;
    html = '<div><div class="lbl">' + esc((d.created || '').slice(0, 16)) + ' 시작 · 조건마다 파리 ' + d.flies + '마리</div>' +
      '<h2>' + esc(d.title) + '</h2>' +
      '<div><i class="tagA">A</i>' + esc(d.a) + ' <span class="lbl">와</span> <i class="tagB">B</i>' + esc(d.b) + '</div></div>' +
      '<div class="resheadside"><div><span class="badge ' + e.state + '">' + (STATE_KO[e.state] || e.state) + '</span> ' +
      progressBar(e) + ' ' + e.done + '/' + e.total + ' 시행' + (isRunning(e) ? ' · ' + timeLeft(e.eta_s) : '') +
      (isRunning(e) && e.stop_requested ? ' · 멈추는 중' : '') + '</div>' +
      '<div class="resheadbtns">' + (isRunning(e) && !e.stop_requested ? '<button id="lr_stop">■ 멈추기</button>' : '') +
      '<button id="lr_report" class="primary">탐구 보고서 열기</button>' +
      '<button id="lr_again">같은 질문으로 다시 실험</button></div></div>';
  }
  if (html !== lab.headHtml) { lab.headHtml = html; lr_head.innerHTML = html; }
}
function renderLabVerdict() {
  const d = lab.detail;
  if (!d) { lr_verdict.innerHTML = ''; return; }
  const sm = d.summary, unit = unitOf(d.metric);
  let html = '<div class="labcardtitle">가설 판정</div>' +
    '<div class="myhyp"><span class="lbl">내 가설</span><br>' + esc(d.prediction_text) + '</div>';
  if (!sm || !sm.a || !sm.b) {
    html += '<div class="verdictbox">아직 결과를 기다리고 있습니다</div>' +
      '<p class="lbl">A와 B의 파리가 시행을 마치면 여기에 판정이 나옵니다.</p>';
  } else {
    html += '<div class="verdictbox ' + (VERDICT_CLASS[sm.code] || '') + '">' + esc(sm.verdict) + '</div>';
    html += '<p>' + esc(d.phase) + '의 <b>' + esc(d.metric) + '</b> 평균: <b class="ab-a">A ' + fmtNum(sm.a.mean) + unit + '</b> · ' +
      '<b class="ab-b">B ' + fmtNum(sm.b.mean) + unit + '</b>' +
      (sm.pairs ? '<br>쌍둥이 ' + sm.pairs + '쌍을 하나씩 비교하면 A가 더 큼 <b>' + sm.a_higher + '</b>쌍 · 비슷함 <b>' + sm.ties +
        '</b>쌍 · A가 더 작음 <b>' + sm.a_lower + '</b>쌍 <span class="lbl">(차이가 ' + fmtNum(sm.tolerance) + unit +
        ' 안이면 비슷함)</span>' : '') + '</p>';
    if (sm.code !== 'too_few' && sm.a_higher + sm.a_lower > 0)
      html += '<p class="lbl">A와 B가 사실 똑같다면, 동전을 ' + (sm.a_higher + sm.a_lower) + '번 던져 이만큼 한쪽으로 몰릴 확률은 약 ' +
        Math.round(100 * sm.p_sign) + '%입니다. 작을수록 우연이 아닐 가능성이 큽니다.</p>';
    html += '<div class="labchart">' + labChart(d) + '</div>' +
      '<p class="lbl">점 하나가 파리 한 마리(' + esc(d.phase) + '의 평균), 굵은 가로선이 평균, 회색 선이 같은 번호의 쌍둥이입니다.' +
      (d.latency ? ' 시행 시간 안에 일어나지 않았으면 시행 시간으로 셉니다(표에서 + 표시).' : '') + '</p>';
  }
  lr_verdict.innerHTML = html;
}
function niceStep(span, n) {
  const raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / mag;
  return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * mag;
}
function labChart(d) {
  const sm = d.summary;
  if (!sm || !sm.scores) return '';
  const a = sm.scores.a || {}, b = sm.scores.b || {};
  const values = Object.values(a).concat(Object.values(b));
  if (!values.length) return '';
  let lo = Math.min(...values), hi = Math.max(...values);
  if (d.metric_key.endsWith('_pi')) { lo = -1; hi = 1; }
  else {
    lo = Math.min(lo, 0);
    if (hi <= lo) hi = lo + 1;
    hi += (hi - lo) * 0.08;
  }
  const W = 460, H = 250, L = 52, R = 14, T = 12, B = 30;
  const y = v => T + (hi - v) / (hi - lo) * (H - T - B);
  const xa = L + (W - L - R) * 0.3, xb = L + (W - L - R) * 0.7;
  const out = ['<svg viewBox="0 0 ' + W + ' ' + H + '" role="img">'];
  const step = niceStep(hi - lo, 5);
  for (let i = Math.ceil(lo / step); i * step <= hi + step * 0.001; i++) {
    const v = i * step, yy = y(v).toFixed(1);
    out.push('<line x1="' + L + '" x2="' + (W - R) + '" y1="' + yy + '" y2="' + yy + '" stroke="' + (i === 0 ? '#666' : '#2c2c2c') + '"/>');
    out.push('<text x="' + (L - 6) + '" y="' + (y(v) + 4).toFixed(1) + '" text-anchor="end" font-size="11" fill="#8a8a8a">' +
      (i === 0 ? '0' : +v.toFixed(2)) + '</text>');
  }
  Object.keys(a).forEach(f => {
    if (b[f] !== undefined)
      out.push('<line x1="' + xa + '" y1="' + y(a[f]).toFixed(1) + '" x2="' + xb + '" y2="' + y(b[f]).toFixed(1) + '" stroke="#555"/>');
  });
  [[xa, a, '#e0874f', 'A'], [xb, b, '#6f9fe8', 'B']].forEach(([x, scores, colour, name]) => {
    const entries = Object.entries(scores);
    entries.forEach(([f, v]) => {
      const jitter = ((f * 37) % 11 - 5) * 1.4;
      out.push('<circle cx="' + (x + jitter).toFixed(1) + '" cy="' + y(v).toFixed(1) + '" r="4.5" fill="' + colour + '" fill-opacity="0.8"/>');
    });
    if (entries.length) {
      const m = entries.reduce((sum, e) => sum + e[1], 0) / entries.length;
      out.push('<line x1="' + (x - 34) + '" x2="' + (x + 34) + '" y1="' + y(m).toFixed(1) + '" y2="' + y(m).toFixed(1) +
        '" stroke="' + colour + '" stroke-width="3.5"/>');
    }
    out.push('<text x="' + x + '" y="' + (H - 8) + '" text-anchor="middle" font-size="13" fill="' + colour + '">' +
      name + ' (' + entries.length + '마리)</text>');
  });
  if (d.sides) {
    out.push('<text x="' + (W - R - 4) + '" y="' + (y(1) + 14).toFixed(1) + '" text-anchor="end" font-size="11" fill="#9a9a9a">+1: 내내 ' + esc(d.sides[0]) + ' 쪽</text>');
    out.push('<text x="' + (W - R - 4) + '" y="' + (y(-1) - 6).toFixed(1) + '" text-anchor="end" font-size="11" fill="#9a9a9a">-1: 내내 ' + esc(d.sides[1]) + ' 쪽</text>');
  }
  out.push('<text transform="translate(13 ' + ((T + H - B) / 2).toFixed(1) + ') rotate(-90)" text-anchor="middle" font-size="11" fill="#9a9a9a">' +
    esc(d.metric) + '</text></svg>');
  return out.join('');
}
function renderLabTrials() {
  const d = lab.detail;
  if (!d || !d.grid) { lr_trials.innerHTML = ''; return; }
  const rp = lab.replay, unit = unitOf(d.metric);
  const html = d.grid.map((g, c) => {
    const rows = [];
    g.flies.forEach((cells, f) => {
      const buttons = [];
      cells.forEach((cell, i) => {
        if (!cell || !cell.replay) return;
        const playing = rp && rp.id === d.id && rp.condition === c && rp.fly === f + 1 && rp.seq === i + 1;
        const role = g.roles[i] || 'repeat';
        const name = cells.length < 2 ? '' : (role === 'repeat' ? (i + 1) + '회' : (i + 1) + ' ' + ROLE_KO[role]) + ' · ';
        const value = cell.v === null ? '-' : fmtNum(cell.v) + (cell.censored ? '+' : '') + unit;
        buttons.push('<button data-c="' + c + '" data-f="' + (f + 1) + '" data-s="' + (i + 1) + '"' +
          (playing ? ' class="playing"' : '') + '>' + name + value + ' ▶</button>');
      });
      if (buttons.length) rows.push('<div class="row"><span>파리 ' + (f + 1) + '</span>' + buttons.join('') + '</div>');
    });
    return '<div class="trialcond"><b class="' + (c ? 'ab-b' : 'ab-a') + '">' + esc(g.label) + '</b>' +
      (rows.length ? rows.join('') : '<div class="lbl">아직 끝난 시행이 없습니다</div>') + '</div>';
  }).join('');
  const who = d.replay_flies < 0 ? '모든 파리' : '파리 1~' + d.replay_flies + '번';
  lr_trials.innerHTML = html + '<p class="lbl">3D 기록은 조건마다 ' + who + '만 남깁니다. 버튼의 숫자는 그 시행의 ' +
    esc(d.metric) + '입니다.</p>';
}
function renderLabGrid() {
  const d = lab.detail;
  if (!d || !d.grid) { lr_grid.innerHTML = ''; return; }
  const n = Math.max(...d.grid.map(g => g.roles.length));
  const values = d.grid.flatMap(g => g.flies.flat()).filter(cell => cell && cell.v !== null).map(cell => cell.v);
  const lo = values.length ? Math.min(...values) : 0, hi = values.length ? Math.max(...values) : 1;
  const roles = d.grid.reduce((a, g) => g.roles.length > a.length ? g.roles : a, []);
  let html = '<thead><tr><th class="cond" title="칸마다 그 시행의 값입니다. 60+처럼 +가 붙으면 시행 시간 안에 일어나지 않았다는 뜻이고, ▶ 칸은 3D로 다시 볼 수 있습니다.">' +
    esc(d.metric) + ' ⓘ</th>';
  for (let i = 0; i < n; i++) html += '<th>' + (i + 1) + (roles[i] ? ' ' + ROLE_KO[roles[i]] : '') + '</th>';
  html += '</tr></thead><tbody>';
  // '60+' is a latency that never ended: the fly did not get there in the trial's 60 s.
  d.grid.forEach((g, c) => g.flies.forEach((cells, f) => {
    html += '<tr><td class="cond ' + (c ? 'ab-b' : 'ab-a') + '">' + esc(g.label.split(':')[0]) + ' 파리 ' + (f + 1) + '</td>';
    for (let i = 0; i < n; i++) {
      const cell = cells[i];
      if (!cell) { html += '<td class="nr">·</td>'; continue; }
      const u = cell.v === null || hi === lo ? 0.5 : (cell.v - lo) / (hi - lo);
      const bg = cell.v === null ? '' : 'background:rgba(' + (c ? '91,143,232' : '224,135,79') + ',' + (0.1 + 0.45 * u).toFixed(2) + ')';
      const text = cell.v === null ? '-' : fmtNum(cell.v) + (cell.censored ? '+' : '');
      html += cell.replay
        ? '<td class="rp" data-c="' + c + '" data-f="' + (f + 1) + '" data-s="' + (i + 1) + '" style="' + bg + '" title="눌러서 이 시행 다시 보기">' + text + ' ▶</td>'
        : '<td style="' + bg + '">' + text + '</td>';
    }
    html += '</tr>';
  }));
  lr_grid.innerHTML = html + '</tbody>';
}

// ---- wiring
lab_open.onclick = () => labOpen();
sablab.onclick = () => {
  const running = lab.list.find(isRunning);
  if (running) labOpen('result', running.id); else labOpen('home');
};
lab_close.onclick = labClose;
addEventListener('keydown', e => { if (lab.open && e.key === 'Escape') { e.preventDefault(); labClose(); } });
labsteps.onclick = ev => { const li = ev.target.closest('li.done'); if (li) labGo(li.dataset.step); };
labmini.onclick = ev => { const b = ev.target.closest('[data-open]'); if (b) labOpen('result', b.dataset.open); };
lab_home.onclick = ev => {
  const open = ev.target.closest('[data-open]');
  if (open) { labGo('result', open.dataset.open); labRefresh(true); return; }
  const question = ev.target.closest('[data-set]');
  if (question) { labGo('set', question.dataset.set); return; }
  if (ev.target.closest('#lab_showall')) { lab.showAll = !lab.showAll; renderLabExps(); }
};
lab_set.onclick = ev => {
  if (ev.target.closest('[data-go]')) { labGo('home'); return; }
  const preview = ev.target.closest('[data-preview]');
  if (preview) { sbPost({op:'exp_preview', set: lab.key, condition: +preview.dataset.preview}); labClose(); return; }
  if (ev.target.closest('#lab_start')) labStart();
};
lab_set.addEventListener('change', ev => {
  if (ev.target.name !== 'labchoice') return;
  lab.choice = ev.target.value;
  lab_set.querySelectorAll('.choice').forEach(el => el.classList.toggle('on', el.querySelector('input').checked));
  renderLabRun();
});
lab_set.addEventListener('input', ev => {
  if (ev.target.id !== 'lab_flies') return;
  lab.flies = parseInt(ev.target.value) || 8;
  renderLabRun();
});
lab_result.onclick = ev => {
  const el = ev.target;
  if (el.closest('[data-go]')) { labGo('home'); return; }
  const trial = el.closest('button[data-c], td.rp');
  if (trial && lab.detail) {
    sbPost({op:'replay', id: lab.detail.id, condition: +trial.dataset.c, fly: +trial.dataset.f, seq: +trial.dataset.s});
    return;
  }
  const camera = el.closest('[data-labview]');
  if (camera) { post({view: camera.dataset.labview}); return; }
  if (el.closest('#lr_stop') && confirm('이 실험을 멈출까요? 지금까지 끝난 시행은 남고, 그것으로 판정합니다.')) {
    sbPost({op:'exp_stop', id: lab.id});
    setTimeout(() => labRefresh(true), 800);
  }
  if (el.closest('#lr_report')) window.open('/report?id=' + encodeURIComponent(lab.id), '_blank');
  if (el.closest('#lr_again') && lab.detail) {
    lab.key = lab.detail.set; lab.flies = lab.detail.flies; lab.choice = lab.detail.prediction;
    labGo('set');
  }
};
lr_follow.onchange = () => sbPost({op:'follow', id: lab.id, on: lr_follow.checked});
r_play.onclick = () => sbPost({op:'replay_ctl', action: sb.replayPlaying ? 'pause' : 'play'});
r_speed.onchange = () => sbPost({op:'replay_ctl', action:'speed', value: parseFloat(r_speed.value)});
r_seek.addEventListener('input', () => {
  sb.seeking = true;
  if (sb.replayDuration) sbPost({op:'replay_ctl', action:'seek', value: parseFloat(r_seek.value) / 1000 * sb.replayDuration});
});
r_seek.addEventListener('change', () => { sb.seeking = false; });
r_stop.onclick = () => sbPost({op:'replay_ctl', action:'stop'});

function renderReplay(s) {
  const rp = s.replay;
  replaybar.style.display = rp ? '' : 'none';
  sb.replayFlat = rp ? rp.path.flatMap(q => [q[1], q[2]]) : null;
  const before = lab.replay ? [lab.replay.id, lab.replay.condition, lab.replay.fly, lab.replay.seq].join() : '';
  lab.replay = rp ? {id: rp.id, condition: rp.condition, fly: rp.fly, seq: rp.seq} : null;
  if (lab.open) {
    if ((rp ? [rp.id, rp.condition, rp.fly, rp.seq].join() : '') !== before && lab.view === 'result') renderLabTrials();
    lr_hint.style.display = rp ? 'none' : '';
    const following = !!s.follow && s.follow === lab.id;
    if (lr_follow.checked !== following) lr_follow.checked = following;
  }
  if (!rp) return;
  sb.replayPlaying = rp.playing; sb.replayDuration = rp.duration;
  r_label.textContent = rp.label;
  r_play.textContent = rp.playing ? '⏸' : '▶';
  r_time.textContent = rp.t.toFixed(1) + ' / ' + rp.duration.toFixed(0) + '초';
  if (!sb.seeking) r_seek.value = Math.round(1000 * rp.t / Math.max(rp.duration, 0.01));
  if (parseFloat(r_speed.value) !== rp.speed) r_speed.value = String(rp.speed);
}

function renderTrials(s) {
  const trial = s.sb_trial, rows = s.trial_rows || [];
  sb.rows = rows;
  t_start.disabled = !!trial;
  t_stop.disabled = !trial;
  t_csv.disabled = !rows.length;
  t_status.textContent = trial
    ? `실험 ${trial.number} 진행 중 · ${trial.elapsed.toFixed(1)} / ${trial.seconds.toFixed(0)}초`
    : rows.length ? `기록 ${rows.length}줄 · out/sandbox/trials.csv에도 저장됨` : '아직 기록이 없습니다';
  strial.textContent = trial ? `${trial.number}번 ${trial.elapsed.toFixed(0)}/${trial.seconds.toFixed(0)}초` : '없음';
  if (rows.length === sb.rowsShown) return;
  sb.rowsShown = rows.length;
  if (!rows.length) { t_table.innerHTML = ''; return; }
  const keys = Object.keys(rows[0]);
  t_table.innerHTML = '<thead><tr>' + keys.map(k => `<th>${esc(k)}</th>`).join('') + '</tr></thead><tbody>' +
    [...rows].reverse().map(r => '<tr>' + keys.map(k => `<td>${esc(r[k])}</td>`).join('') + '</tr>').join('') + '</tbody>';
}

function renderSandbox(s) {
  setupPalette(s);
  // Keep the dragged item where the pointer is, not where the last poll put it.
  const dragged = sb.drag ? sb.items.find(i => i.id === sb.drag.id) : null;
  sb.items = (s.items || []).map(it => dragged && it.id === dragged.id ? {...it, x: dragged.x, y: dragged.y} : it);
  if (sb.selected !== null && !sb.items.some(i => i.id === sb.selected)) selectItem(null);
  sb.fly = s.fly; syncTrail(s); renderReplay(s); sb.half = s.room_half || 50;
  b_deplete.classList.toggle('on', !!s.sugar_depletes);
  b_undo.disabled = !s.undo;
  b_undo.title = s.undo ? '되돌리기: ' + s.undo + ' (Ctrl+Z)' : '되돌릴 편집이 없습니다';
  b_redo.disabled = !s.redo;
  b_redo.title = s.redo ? '다시 하기: ' + s.redo + ' (Ctrl+Y)' : '다시 할 편집이 없습니다';
  b_deplete.textContent = '먹으면 줄어듦: ' + (s.sugar_depletes ? '켜짐' : '꺼짐');
  const sel = sb.items.find(i => i.id === sb.selected);
  selbar.classList.toggle('on', !!sel);
  p_selected.textContent = sel ? '고름: ' + KIND_KO[sel.kind] + ' (' + sel.x.toFixed(0) + ', ' + sel.y.toFixed(0) + ')' : '-';
  const cv = document.getElementById('c_map');
  drawMap(cv.getContext('2d'));
  const tel = s.telemetry || {};
  learnBars(document.getElementById('c_learn').getContext('2d'), tel);
  smode.textContent = MODE_KO[tel.mode] || tel.mode || '-';
  smode.style.color = tel.mode === 'escape' ? '#ff8a70' : tel.mode === 'feed' ? '#e8e8a0' : '#7fb2f0';
  shunger.textContent = (tel.hunger ?? 0).toFixed(2);
  const counts = tel.counts || {};
  sshocks.textContent = (counts.shocks || 0) + '회 · ' + (counts.shock_seconds || 0) + 's';
  sfeed.textContent = (counts.feed_seconds || 0) + 's';
  sspeed.textContent = s.speed_factor ? '×' + s.speed_factor.toFixed(2) : (s.paused ? '멈춤' : '-');
  if (!sb.hungerDrag) { p_hunger.value = tel.hunger ?? 0; p_hunger_v.textContent = (tel.hunger ?? 0).toFixed(2); }
  const bySugar = Object.entries(counts.by_sugar || {}).map(([k, v]) => (SUGAR_KO[k] || k).split(' ')[0] + ' ' + v + 's').join(', ') || '-';
  bodykv.innerHTML =
    `<i>행동</i><span>${MODE_KO[tel.mode] || '-'}</span>` +
    `<i>설탕 위 다리</i><span>${tel.legs_on_sugar ?? 0}${tel.sugar_under ? ' · ' + esc(SUGAR_KO[tel.sugar_under] || tel.sugar_under) + ' 먹는 중' : ''}</span>` +
    `<i>전기 닿음</i><span>${tel.legs_on_shock ? '예' : '아니오'}</span>` +
    `<i>전기 맞은 횟수</i><span>${counts.shocks || 0}회 (${counts.shock_seconds || 0}초)</span>` +
    `<i>먹은 횟수</i><span>${counts.feed_bouts || 0}번 (${counts.feed_seconds || 0}초)</span>` +
    `<i>설탕별 먹은 시간</i><span>${esc(bySugar)}</span>`;
  eventlog.innerHTML = [...(s.events || [])].reverse()
    .map(e => `<li><b>${e.t.toFixed(1)}s</b>${esc(e.text)}</li>`).join('');
  renderTrials(s);
}

// The map follows the fly faster than the rest of the page refreshes.
async function sandboxLoop() {
  if (laidOut === 'sandbox') {
    try {
      const s = await (await fetch('/state')).json();
      if (s && s.sandbox) renderSandbox(s);
    } catch (e) {}
  }
  setTimeout(sandboxLoop, 150);
}
// Started by timer, not called here: `laidOut` is declared further down, and
// reading it before that line runs throws and would end the loop for good.
setTimeout(sandboxLoop, 300);

const cs = {odor: ctx('c_odor'), attr: ctx('c_attr'), attrhist: ctx('c_attrhist'),
            kc: ctx('c_kc'), mbon: ctx('c_mbon'), val: ctx('c_val'),
            vkc: ctx('c_vkc'), vpn: ctx('c_vpn'), eyeval: ctx('c_eyeval'),
            dan: ctx('c_dan')};

setInterval(async () => {
  let n;
  try { n = await (await fetch('/neuro')).json(); } catch (e) { return; }
  // Empty until the simulation thread publishes its first snapshot, which is
  // after the HTTP server is already answering.
  if (!n || !n.traces) return;
  fitCanvases();

  traces(cs.odor, [
    {data:[...n.traces.odor_l], color:'#4d8fd6', label:'왼쪽'},
    {data:[...n.traces.odor_r], color:'#d68f4d', label:'오른쪽'},
  ]);

  if (n.sandbox) {
    traces(cs.dan, [
      {data:n.traces.hunger,   color:'#9a9a9a', label:'배고픔'},
      {data:n.traces.nutrient, color:'#5fc6d8', label:'영양 보상'},
      {data:n.traces.sweet,    color:'#e58fd0', label:'단맛 보상'},
      {data:n.traces.punish,   color:'#e06c6c', label:'벌'},
    ], {lo:0, hi:1.05});
  } else if (n.conditioning && n.modality === 'colour') {
    vkcRaster(cs.vkc, n.vkc_active, n.kc_ref, n.stimuli, n.vkc_n, n.under);
    vpnBars(cs.vpn, n.vpn, n.vpn_labels);
    // In units of a resting synapse: the visual code is scaled on the server,
    // so 1.0 is an untouched MBON and 0 a silenced one.
    traces(cs.mbon, [
      {data:n.traces.mbon_avoid,    color:'#d66a6a', label:'회피'},
      {data:n.traces.mbon_approach, color:'#6ad07a', label:'접근'},
    ], {lo:0, hi:1.1, marks:n.dan_marks, mcolor:'#9a6a20'});
    const other = n.stimuli.find(name => name !== n.focus);
    traces(cs.val, [
      {data:n.traces.valence_other, color:'#9a9a9a', label:ko(other) + ' (CS−)'},
      {data:n.traces.valence,       color:'#c98fd6', label:ko(n.focus) + ' (CS+)'},
    ], {lo:-1.05, hi:1.05, hline:0, hcolor:'#555', marks:n.phase_marks});
    gainval.textContent = n.gain;
    traces(cs.eyeval, [
      {data:n.traces.eye_left,  color:'#4d8fd6', label:'왼쪽 눈'},
      {data:n.traces.eye_right, color:'#d68f4d', label:'오른쪽 눈'},
      {data:n.traces.turn,      color:'#e8e8e8', label:'회전'},
    ], {lo:-1.05, hi:1.05});
  } else if (n.conditioning) {
    kcRaster(cs.kc, n.kc_active, n.kc_ref, n.kc_odour, n.kc_n);
    // 회피 first so 접근 draws over it: the two sit on top of each other until
    // dopamine has taught them apart, and that overlap is the "before" picture.
    traces(cs.mbon, [
      {data:n.traces.mbon_avoid,    color:'#d66a6a', label:'회피'},
      {data:n.traces.mbon_approach, color:'#6ad07a', label:'접근'},
    ], {lo:0, hi:1.1, marks:n.dan_marks, mcolor:'#9a6a20'});
    // Bounds pinned so the zero line stays put; a fitted axis would slide the
    // one crossing the panel is about.
    traces(cs.val, [
      {data:n.traces.valence, color:'#c98fd6', label:'가치'},
      {data:n.traces.drive,   color:'#7fb2f0', label:'접근 구동 = 선천 ' + n.innate + ' + 가치'},
    ], {lo:-0.85, hi:0.45, hline:0, marks:n.phase_marks});
    valcap.textContent = '지금 따라가는 냄새: ' + (n.focus || '-') + (memoryName
      ? ' (불러온 기억이 더 싫어하는 냄새).'
      : ' (반전 전에는 A, 반전 후에는 새로 벌받는 B).');
  } else if (n.has_brain) {
    traces(cs.attr, Object.entries(n.attribution).map(([label, data], i) =>
      ({data, color: ATTR_COLORS[i % ATTR_COLORS.length], label})),
      {lo: 0, marks: n.episode_marks});
    attrBars(cs.attrhist, n.attr_history || [], n.channels, n.probe_ready);
  } else {
    const msg = '정책이 운전할 때만 표시됩니다 (--follow 또는 --policy)';
    clear(cs.attr); text(cs.attr, msg, 12, 24);
    clear(cs.attrhist); text(cs.attrhist, msg, 12, 24);
  }
}, 250);

// Which panels this run has anything to put in. Re-applied whenever the mode
// changes rather than settled once: the first /state can arrive before the
// simulation thread has published anything, and an empty status read as "not
// conditioning" used to lock the page into the RL layout for good. The same
// thing happened to a tab left open across a server restart into another mode.
let laidOut = null;
//: Captions that change with the task. The odour ones are what the page loads with.
const CAPS = {};
function applyMode(s) {
  const mode = s.sandbox ? 'sandbox' : !s.conditioning ? 'rl'
    : s.modality === 'colour' ? 'colour' : 'odour';
  if (laidOut === mode) return;
  laidOut = mode;
  if (mode === 'sandbox') {
    const show = new Set(['mapcard', 'learncard', 'dancard', 'bodycard', 'logcard',
                          'explaincard', 'sandbar']);
    ['eyecard', 'kccard', 'odorcard', 'vpncard', 'vkccard', 'attrcard', 'attrhistcard', 'mboncard',
     'valcard', 'eyevalcard', 'condbar', 'explaincard', 'stagelegend', 'mapcard', 'learncard',
     'dancard', 'bodycard', 'logcard', 'sandbar'].forEach(id =>
      document.getElementById(id).style.display = show.has(id) ? '' : 'none');
    document.querySelector('main').classList.add('sandbox');
    document.querySelector('footer').classList.add('four');
    document.querySelector('footer').insertBefore(learncard, dancard);
    protocol.style.display = 'none';
    if (CAPS.intro === undefined) CAPS.intro = intro.innerHTML;
    intro.innerHTML = '<b>샌드박스</b> — 사각 밀폐 방에 설탕·전기·냄새·색 바닥·장애물을 마음대로 놓고 초파리의 반응을 봅니다. ' +
      '<b>타고난 반응</b>: 배고프면 단맛 나는 설탕 위에서 멈춰 먹고, 전기에 닿으면 돌아서 달아나고, 벽에는 닿기 전에 돌아섭니다. ' +
      '<b>학습</b>: 도파민이 나오는 순간 맡고 있던 냄새·보고 있던 색이 기억되어, 다음부터 멀리서도 다가가거나 피합니다.';
    explainnow.innerHTML = '해 볼 것: 전기 구역에 <b>파란 바닥</b>을 겹쳐 두면 몇 번 맞은 뒤 파랑을 피하는지, 설탕에 <b>식초</b>를 ' +
      '겹쳐 두면 식초 쪽으로 곧장 가는지, 배고픔을 0으로 내리면 설탕을 지나치는지 보세요. 한계: 단서 없는 <b>장소</b>는 기억하지 ' +
      '못하고(중심복합체 몫, 아직 없음), 냄새는 지금 장애물을 통과해 퍼집니다. 소르비톨은 맛이 없어 초파리가 먹기 시작하지 않습니다.';
    eyeview.removeAttribute('src');
    document.querySelector('[data-act=reset]').textContent = '새 파리 (방은 그대로)';
    document.querySelector('[data-act=reset]').title = '기억과 배고픔까지 모두 새로 시작하는 새 파리입니다. 방은 그대로입니다.';
    b_home.style.display = '';
    sbtabs.style.display = '';
    lbl_sig.textContent = '다리 신호 좌/우'; lbl_up.textContent = '몸 기울기(1=똑바로)';
    lbl_legs.textContent = '바닥 닿은 다리'; lbl_goal.textContent = '가장 가까운 설탕까지';
    lbl_touch.textContent = '벽 접촉';
    setSbTab(sb.tab);
    fitCanvases();
    return;
  }
  sbtabs.style.display = 'none'; trialpane.style.display = 'none';
  intro.style.display = ''; explainnow.style.display = '';
  document.querySelector('main').classList.remove('sandbox');
  ['mapcard', 'learncard', 'dancard', 'bodycard', 'logcard', 'sandbar'].forEach(id =>
    document.getElementById(id).style.display = 'none');
  protocol.style.display = '';
  const cond = mode !== 'rl', colour = mode === 'colour';
  [['eyecard', !cond || colour], ['kccard', mode === 'odour'], ['odorcard', !colour],
   ['vpncard', colour], ['vkccard', colour], ['attrcard', !cond],
   ['attrhistcard', !cond], ['mboncard', cond], ['valcard', cond], ['eyevalcard', colour],
   ['condbar', cond], ['explaincard', cond], ['stagelegend', cond],
   ['legendodour', mode === 'odour'], ['legendcolour', colour]].forEach(([id, show]) => {
    document.getElementById(id).style.display = show ? (id.startsWith('legend') ? 'contents' : '') : 'none';
  });
  document.querySelector('footer').classList.toggle('three', colour);
  for (const id of ['eyecap', 'mboncap', 'valcapdiv', 'intro'])
    if (CAPS[id] === undefined) CAPS[id] = document.getElementById(id).innerHTML;
  const first = (s.stimuli || [])[0], second = (s.stimuli || [])[1];
  const reward = s.reinforcer === 'reward';
  if (colour) {
    eyecap.innerHTML = '겹눈이 보는 화면 <small>왼쪽 눈 | 오른쪽 눈. 낱눈마다 읽는 채널의 색으로 칠했습니다 — ' +
      '<b style="color:#4fbf5a">노랑형</b>(전체의 70%)은 초록을, <b style="color:#5b8fe8">옅은형</b>(30%)은 파랑을 읽고, ' +
      '빨강은 어느 낱눈도 읽지 못합니다. 아래쪽 절반이 바닥, 위쪽이 흰 하늘입니다.</small>';
    mboncap.innerHTML = 'MBON 출력 (시각) <small>색을 보는 동안에는 시각 케니언 세포가 늘 켜져 있어서 값이 끊기지 않습니다' +
      '(1.0 = 휴지 가중치). 주황 세로줄은 ' + (reward ? '설탕(보상)' : '벌') + ' 도파민 펄스로, 훈련 중 5초마다 1초씩 들어옵니다. ' +
      (reward ? '보상은 <b style="color:#d66a6a">회피</b> 쪽 시냅스를 눌러' : '벌은 <b style="color:#6ad07a">접근</b> 쪽 시냅스를 눌러') +
      ' 그 색을 볼 때의 출력이 내려갑니다. 다른 색을 보는 구간에는 휴지값으로 돌아옵니다.</small>';
    valcapdiv.innerHTML = '색별 학습된 가치 <small>버섯체가 각 색에 매긴 값입니다(−1 = 그 색의 시냅스가 모두 억압됨, ' +
      '+1 = 반대 구획이 모두 억압됨). 시각에는 타고난 끌림 항이 없어서 처음엔 둘 다 0이고, ' +
      (reward ? '보상받는 색이 + 쪽으로' : '벌받는 색만 − 쪽으로') + ' 움직여야 합니다. 시행 사이 어둠(12초)과 ' +
      '시험 전 60초 동안에는 복원항으로 조금 되돌아갑니다. 세로 점선은 블록 경계입니다.</small>';
    intro.innerHTML = '<b>실험 설명</b> — Vogt et al. (2014)의 색 학습을 재현합니다. 바닥 색 하나에만 ' +
      (reward ? '설탕(보상 도파민)' : '벌(도파민)') + '을 짝지으면 초파리가 그 색을 ' + (reward ? '찾아가는지' : '피하는지') +
      ' 봅니다. 학습되는 것은 <b>시각 케니언 세포 → MBON 시냅스</b>뿐이고, 냄새 기억과 <b>같은</b> 구획과 도파민을 씁니다. ' +
      '선호도는 +1이 ' + ko(second) + ' 위, −1이 ' + ko(first) + ' 위에 머문 것입니다.';
    ccslabel.textContent = reward ? '보상받는 색' : '벌받는 색';
    cintlabel.textContent = '발밑';
  } else {
    for (const id of ['eyecap', 'mboncap', 'valcapdiv', 'intro'])
      document.getElementById(id).innerHTML = CAPS[id];
    ccslabel.textContent = '벌받는 냄새';
    cintlabel.textContent = '세기';
  }
  // Only ask for the eye stream when something is rendering eyes. The odour
  // task runs vision=False, and an MJPEG request that never gets a frame hangs.
  if (mode === 'odour') eyeview.removeAttribute('src');
  else eyeview.src = '/eyes';
  document.querySelector('[data-act=reset]').textContent = cond ? '새 파리' : '리셋';
  b_home.style.display = 'none';
  fitCanvases();
}

const signed = v => (v > 0 ? '+' : '') + v.toFixed(3);

// What the current block is for and what to watch while it runs. `cs` is the
// odour being punished in this phase of the session -- A until the reversal,
// B after it -- and `other` the one that is not.
const EXPLAIN = {
  naive: () =>
    '처음 보는 냄새 A와 B를 좌우에 놓고 어느 쪽으로 가는지 봅니다. 아직 아무것도 배우지 ' +
    '않았으니 선호도가 <b>0 근처</b>여야 하고, 이 기준선이 있어야 뒤의 변화를 학습이라고 ' +
    '부를 수 있습니다. 초파리는 처음 잡은 냄새 쪽으로 곧장 가버리기 때문에 시행 하나하나는 ' +
    '±0.5씩 크게 흔들립니다. 블록 평균을 보세요.',
  train: (cs, other) =>
    '냄새 <b>' + cs + '</b> 하나만 놓고, 초파리가 그 냄새 속에 0.5초 넘게 머물면 도파민(벌)을 ' +
    '줍니다. 그 순간 켜져 있던 케니언 세포에서 <b>접근 MBON</b>으로 가는 시냅스만 약해집니다. ' +
    'MBON 그래프의 초록 봉우리가 시행마다 낮아지고, 가치 그래프의 파란 선(접근 구동)이 내려갈수록 ' +
    cs + '에 덜 끌리다가 0 아래로 내려가면 ' + cs + '에서 멀어지는 쪽으로 돕니다. 피하기 시작하면 ' +
    '냄새 속에 덜 들어가니 벌도 덜 받습니다.',
  acquisition: (cs, other) =>
    '다시 두 냄새를 좌우에 놓고 이번에는 벌을 주지 않습니다. 배운 대로라면 <b>' + cs +
    '를 피하고 ' + other + ' 쪽으로</b> 가야 하고, 선호도가 ' + (other === 'B' ? '+' : '−') +
    ' 쪽으로 나와야 합니다. 버섯체에 생긴 기억이 실제 행동으로 옮겨졌는지 보는 시험입니다.',
  reversal_train: (cs, other) =>
    '규칙을 뒤집어 이번에는 <b>' + cs + '</b>에 벌을 줍니다. ' + cs + '의 접근 시냅스가 새로 ' +
    '약해지는 동안, ' + other + '에 대한 옛 기억은 도파민 없이 <b>약 100초 시정수로만</b> ' +
    '천천히 풀립니다. 배우는 속도는 도파민이 정하고 잊는 속도는 이 시정수가 정하기 때문에, ' +
    '이 구간이 끝나도 옛 기억이 다 지워지지는 않습니다.',
  reversal: (cs, other) =>
    '다시 두 냄새로 시험합니다. 역전되었다면 <b>' + cs + '를 피하고 ' + other + ' 쪽으로</b> ' +
    '가야 하고 선호도가 ' + (other === 'B' ? '+' : '−') + ' 쪽으로 나와야 합니다. 다만 ' + other +
    '에 대한 옛 기억이 덜 풀려서 두 냄새가 모두 조금씩 싫은 상태라 획득 때보다 약하게 ' +
    '나옵니다. 5마리를 20시행씩 훈련한 실험에서 획득 후 +0.49, 역전 후 −0.30이었습니다.',
  memory_test: (cs, other, s) => {
    const m = s.memory, filed = m.filed || {};
    const fmt = v => (v === undefined ? '-' : signed(v));
    return '저장된 기억 <b>' + esc(m.name) + '</b>을 불러온 초파리입니다. ' + esc(m.note) +
      ' 저장 당시 가치는 A ' + fmt(filed.A) + ', B ' + fmt(filed.B) + '였습니다. 이 시험에서는 ' +
      '벌을 주지 않고, 두 냄새를 좌우에 놓아 그 기억대로 움직이는지만 봅니다. 선호도가 ' +
      '+면 B 쪽, −면 A 쪽입니다.' +
      (m.extinction === 'avoid'
        ? ' 이 초파리는 소거가 켜져 있어서, 벌이 오지 않는 시험을 거치는 동안 싫어하던 냄새의 ' +
          '기억이 조금씩 상쇄됩니다. 헤더의 가치가 저장값에서 움직이는 이유입니다.'
        : m.extinction === 'approach'
        ? ' 이 초파리는 소거 신호가 접근 구획으로 잘못 연결돼 있어서, 시험을 거칠수록 두 냄새가 ' +
          '모두 더 싫어집니다.'
        : '');
  },
  done: (cs, other, s) => s.memory
    ? '시험이 끝났습니다. 아래 <b>새 파리</b>를 누르면 같은 기억을 파일에서 다시 불러와 ' +
      '처음부터 시험합니다.'
    : '세션이 끝났습니다. 아래 <b>새 파리</b>를 누르면 기억을 초기화한 새 개체로 처음부터 ' +
      '다시 시작합니다.',
};

// The same blocks, for the colour task. `cs` is the colour paired with the
// reinforcer, `other` the one presented without it.
const EXPLAIN_COLOUR = {
  naive: (cs, other, s) =>
    '파랑·초록 체커보드 위에 90초 동안 놓고 각 색 위에 머문 시간을 잽니다. 아직 배운 것이 없어 두 눈이 보는 ' +
    '가치가 모두 0이고 색에 따른 조향이 없으므로, 선호도가 <b>0 근처</b>여야 합니다. 대각선 칸이 같은 색이라 ' +
    '어느 교차점에서 보든 Vogt의 사분면 배치와 같습니다.',
  train: (cs, other, s) => s.trial_kind === 'train'
    ? '바닥 전체가 <b>' + ko(cs) + '</b>입니다. 60초 동안 5초마다 1초씩 ' +
      (s.reinforcer === 'reward' ? '보상' : '벌') + ' 도파민이 들어옵니다(Vogt: 1초 전기 충격 12번). ' +
      '그 순간 켜져 있는 시각 케니언 세포의 ' + (s.reinforcer === 'reward' ? '회피' : '접근') +
      ' 시냅스만 약해지고, 냄새 세포는 꺼져 있으니 냄새 기억은 그대로입니다. 첫 60초 만에 거의 포화합니다 ' +
      '(Vogt도 훈련 1회로 이미 유의한 기억이 생겼습니다).'
    : '바닥 전체가 <b>' + ko(s.under || other) + '</b>이고 도파민은 없습니다(CS−). 이 색의 시각 세포는 ' + ko(cs) +
      '과 겹치지 않아서 기억이 옮겨붙지 않습니다. 시행 사이 12초 어둠 동안에는 학습 없이 복원항만 작용합니다.',
  acquisition: (cs, other, s) =>
    '훈련 60초 뒤 다시 체커보드로 90초 시험합니다. 배운 대로라면 <b>' + ko(cs) + ' 칸을 ' +
    (s.reinforcer === 'reward' ? '찾아가야' : '피해 ' + ko(other) + ' 칸 위에 머물러야') + '</b> 합니다. ' +
    '아래 오른쪽 그래프에서 두 눈 중 ' + ko(cs) + '을 더 많이 보는 쪽의 가치가 ' +
    (s.reinforcer === 'reward' ? '높아 그쪽으로' : '낮아 반대쪽으로') + ' 도는 것이 보입니다.',
  memory_test: (cs, other, s) => {
    const m = s.memory, filed = m.filed || {};
    const fmt = v => (v === undefined ? '-' : signed(v));
    return '저장된 기억 <b>' + esc(m.name) + '</b>을 불러온 초파리입니다. ' + esc(m.note) + ' 저장 당시 가치는 ' +
      (s.stimuli || []).map(n => ko(n) + ' ' + fmt(filed[n])).join(', ') + '였습니다. 강화 없이 체커보드 위에서 ' +
      '그 기억대로 움직이는지만 봅니다.';
  },
  done: (cs, other, s) => EXPLAIN.done(cs, other, s),
};

// Memory notes come from files on disk and land in innerHTML.
const esc = t => String(t ?? '').replace(/[&<>"]/g,
  c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
let memoryName = null;

function renderExplanation(s) {
  const colour = s.modality === 'colour';
  const names = s.stimuli || ['A', 'B'];
  const cs = s.focus || names[0];
  const other = names.find(n => n !== cs) || names[1];
  const blocks = s.protocol || [];
  const current = blocks.findIndex(b => b.key === s.block);
  const reinforcer = s.reinforcer === 'reward' ? '보상' : '벌';
  protocol.innerHTML = blocks.map((b, i) => {
    const name = b.kind === 'train'
      ? b.label + ' (' + (colour ? ko(b.odour) : b.odour) + '에 ' + reinforcer + ')'
      : b.label + ' 시험';
    // A finished test block shows its mean; the running one shows its mean so
    // far, and the highlighted block is what says it is still in progress.
    let pi = '';
    if (b.preference !== undefined) pi = '선호도 ' + signed(b.preference);
    else if (i === current && b.kind === 'test' && s.preference !== null)
      pi = '선호도 ' + signed(s.preference);
    const cls = s.done || i < current ? 'past' : i === current ? 'now' : '';
    return '<li class="' + cls + '"><span class="name">' + (i + 1) + '. ' + name + '</span>' +
           '<span class="meta"><span>시행 ' + b.first + '–' + b.last + '</span>' +
           '<span class="pi">' + pi + '</span></span></li>';
  }).join('');
  const key = s.done ? 'done' : s.block;
  const texts = colour ? EXPLAIN_COLOUR : EXPLAIN;
  explainnow.innerHTML = texts[key] ? texts[key](cs, other, s) : '';
}

setInterval(async () => {
  let s;
  try { s = await (await fetch('/state')).json(); } catch (e) { return; }
  // An empty status means the server is up but has not rendered yet. Deciding
  // the layout from it is the bug applyMode's comment describes.
  if (!s || s.terrain === undefined) return;
  applyMode(s);
  pad.style.display = s.driver === 'keyboard' ? '' : 'none';
  terrain.textContent = '— ' + s.terrain +
    (s.sandbox ? ' · 샌드박스' + (s.preset ? ' · ' + (s.palette.presets[s.preset] || '') : '') :
     s.conditioning ? (s.modality === 'colour' ? ' · 색 조건화' : ' · 후각 조건화') +
                      (s.memory ? ' · 기억 ' + s.memory.name : '')
                    : s.driver !== 'keyboard' ? ' · 정책 ' + s.driver : '') +
    (s.paused ? ' · 일시정지' : '');
  time.textContent = s.time + ' s';
  pos.textContent  = s.x + ', ' + s.y + ' mm';
  spd.textContent  = s.speed_mms + ' mm/s';
  sig.textContent  = s.left.toFixed(2) + ' / ' + s.right.toFixed(2);
  up.textContent   = s.upright;
  legs.textContent = s.legs_down + ' / 6';
  distlbl.textContent = s.distance + ' mm';
  if (s.training_steps !== undefined) {
    learn.style.display = '';
    lsteps.textContent  = s.training_steps.toLocaleString();
    lrecent.textContent = s.recent_n;
    lfound.textContent  = s.recent_found;
    leps.textContent    = s.episodes;
    lreload.textContent = s.reloads;
  }
  if (s.conditioning) {
    memoryName = s.memory ? s.memory.name : null;
    ctrial.textContent = s.trial;
    ctotal.textContent = s.trials_total;
    const kindLabel = s.trial_kind === 'train' ? '훈련' : '시험';
    // The first training block's label is the kind itself; saying it twice
    // ("훈련 (훈련)") reads as a glitch. The others add information.
    ckind.textContent = s.done ? '세션 완료'
      : kindLabel + (s.block_label && s.block_label !== kindLabel
          ? ' (' + s.block_label + ')' : '');
    if (s.modality === 'colour') {
      ccs.textContent = s.cs_plus ? ko(s.cs_plus) : (s.trial_kind === 'expose' ? '없음 (CS−)' : '없음');
      const on = s.time_on || {}, names = s.stimuli || [];
      cint.textContent = ko(s.under) + (s.dopamine ? ' · 도파민' : '') +
        (s.trial_kind === 'test' ? ' · ' + names.map(n => ko(n) + ' ' + (on[n] || 0) + 's').join(' / ') : '');
      // The running test's own score, before the block closes: PI toward the second colour.
      const a = on[names[0]] || 0, b = on[names[1]] || 0;
      const live = s.trial_kind === 'test' && a + b > 0 ? (b - a) / (a + b) : null;
      cpref.textContent = live !== null ? signed(live) + ' (진행 중)'
        : s.preference === null ? '-' : signed(s.preference);
    } else {
      ccs.textContent = s.cs_plus || '없음';
      cint.textContent = s.intensity.toFixed(4) +
        (s.dopamine ? ' · 도파민' : s.smelling ? ' · 감지' : '');
      cpref.textContent = s.preference === null ? '-' : signed(s.preference);
    }
    cint.style.color = s.dopamine ? '#e0a24d' : '#7fb2f0';
    renderExplanation(s);
  }
  if (s.odor !== undefined) {
    forage.style.display = '';
    odor.textContent  = s.odor;
    goaldist.textContent = s.goal_dist !== undefined ? s.goal_dist + ' mm' : '-';
    touch.textContent = s.touching ? '접촉' : '없음';
    touch.style.color = s.touching ? '#e06c6c' : '#eee';
  }
  if (s.odor_asym !== undefined || s.vis_asym !== undefined) {
    senses.style.display = '';
    const side = v => v > 0.002 ? ' (왼쪽)' : v < -0.002 ? ' (오른쪽)' : '';
    if (s.odor_asym !== undefined)
      oasym.textContent = s.odor_asym.toFixed(4) + side(s.odor_asym);
    vissense.style.display = s.vis_asym !== undefined ? '' : 'none';
    if (s.vis_asym !== undefined) {
      vasym.textContent = s.vis_asym.toFixed(4) + side(s.vis_asym);
      vtot.textContent  = s.vis_total.toFixed(4);
    }
  }
  if (!dragging) dist.value = s.distance;
  document.getElementById('pause').textContent = s.paused ? '재생' : '일시정지';
  document.querySelectorAll('.view').forEach(b =>
    b.classList.toggle('on', b.dataset.view === s.view));
  document.querySelectorAll('.spd').forEach(b =>
    b.classList.toggle('on', parseFloat(b.dataset.spd) === s.playback));
}, 400);
</script>
"""


def make_handler(server: FlyServer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # quiet
            pass

        def do_GET(self):
            if self.path == "/":
                body = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path in ("/state", "/neuro"):
                data = (
                    server.status if self.path == "/state" else server.neuro_snapshot
                )
                body = json.dumps(data).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path.startswith(("/sets", "/experiments", "/experiment?", "/report?")):
                parsed = urlparse(self.path)
                query = parse_qs(parsed.query)
                try:
                    if parsed.path == "/sets":
                        body, kind = json.dumps(server.sets_payload()).encode(), "application/json"
                    elif parsed.path == "/experiments":
                        body, kind = json.dumps(server.experiments_list()).encode(), "application/json"
                    elif parsed.path == "/experiment":
                        body = json.dumps(server.experiment_detail(query["id"][0])).encode()
                        kind = "application/json"
                    else:
                        body = build_report(server._experiment_dir(query["id"][0])).encode("utf-8")
                        kind = "text/html; charset=utf-8"
                except (KeyError, ValueError, OSError):
                    self.send_error(404, "no such experiment")
                    return
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path.startswith("/trail"):
                query = parse_qs(urlparse(self.path).query)
                try:
                    since = int(query.get("since", ["0"])[0])
                except ValueError:
                    since = 0
                body = json.dumps(server.trail_since(since)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/eyes":
                self.send_response(200)
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                )
                self.end_headers()
                try:
                    while True:
                        frame = server.next_eye_frame()
                        if frame is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(frame)}\r\n\r\n".encode()
                        )
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                except ConnectionError:
                    # A closed tab. The base class, not the two POSIX names:
                    # on Windows the same event is ConnectionAbortedError
                    # (WinError 10053), which slipped past those and printed a
                    # full traceback every time a browser tab was closed.
                    pass
            elif self.path == "/stream":
                self.send_response(200)
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                )
                self.end_headers()
                try:
                    while True:
                        frame = server.next_frame()
                        if frame is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(frame)}\r\n\r\n".encode()
                        )
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                except ConnectionError:
                    # A closed tab. The base class, not the two POSIX names:
                    # on Windows the same event is ConnectionAbortedError
                    # (WinError 10053), which slipped past those and printed a
                    # full traceback every time a browser tab was closed.
                    pass
            else:
                self.send_error(404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                cmd = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                cmd = {}

            if self.path == "/sandbox":
                if server.sandbox_mode and isinstance(cmd, dict):
                    server.commands.put(cmd)
                self.send_response(204)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            key = cmd.get("key")
            if key == "a":
                server.turn = min(server.turn + 0.15, MAX_TURN)
            elif key == "d":
                server.turn = max(server.turn - 0.15, -MAX_TURN)
            elif key == "w":
                server.drive = min(server.drive + 0.15, SIGNAL_HIGH)
            elif key == "s":
                server.drive = max(server.drive - 0.15, SIGNAL_LOW)
            elif key == "c":
                server.turn = 0.0

            if cmd.get("act") == "reset":
                server.reset_requested = True
            elif cmd.get("act") == "pause":
                if server.sandbox_mode and server.replay is not None:
                    server.commands.put({"op": "replay_ctl", "action": "toggle"})
                else:
                    server.paused = not server.paused
            if cmd.get("view") == "room" or cmd.get("view") in VIEWS:
                server.set_view(cmd["view"])
            pan = cmd.get("pan")
            if isinstance(pan, list) and len(pan) == 2 and all(
                isinstance(v, (int, float)) for v in pan
            ):
                server.pan(float(pan[0]), float(pan[1]))
            if cmd.get("zoom") == "in":
                server.cam.distance = max(server.cam.distance / ZOOM_STEP, 2.0)
            elif cmd.get("zoom") == "out":
                server.cam.distance = min(server.cam.distance * ZOOM_STEP, 300.0)
            if isinstance(cmd.get("distance"), (int, float)):
                server.cam.distance = float(np.clip(cmd["distance"], 2.0, 300.0))
            if isinstance(cmd.get("orbit"), (int, float)):
                server.cam.azimuth = (server.cam.azimuth + float(cmd["orbit"])) % 360
            if isinstance(cmd.get("elevation"), (int, float)):
                server.cam.elevation = float(np.clip(cmd["elevation"], -89.0, 20.0))
            if isinstance(cmd.get("elevation_delta"), (int, float)):
                server.cam.elevation = float(
                    np.clip(server.cam.elevation + cmd["elevation_delta"], -89.0, 20.0)
                )
            if isinstance(cmd.get("speed"), (int, float)):
                server.speed = float(cmd["speed"])

            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--terrain", default="flat", choices=sorted(TERRAINS))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Playback speed. Defaults to 0.5 so the legs are watchable, or to "
        "1.0 with --conditioning, where the story is a memory built over "
        "dozens of trials rather than a gait.",
    )
    parser.add_argument("--fps", type=int, default=25)
    # 16:9, because the page's video area is wide: at 800x600 the picture was a
    # 4:3 island in a black frame. The model's offscreen buffer is 2048x2048
    # (measured), so this is nowhere near its limit.
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--pillars",
        type=int,
        default=0,
        help="Scatter this many visible, solid obstacles around the fly.",
    )
    parser.add_argument(
        "--odor",
        action="store_true",
        help="Place a food source 22 mm ahead and show the live odour readings. "
        "The orange marker is in geom group 2, so the fly cannot see it -- "
        "it can only smell it.",
    )
    parser.add_argument(
        "--vision",
        action="store_true",
        help="Also show the live visual features. Costs 25 ms per rendered "
        "frame on top of the view itself, so playback slows down.",
    )
    parser.add_argument(
        "--eye-width",
        type=int,
        default=360,
        help="Width of the compound-eye mosaic in the page.",
    )
    parser.add_argument(
        "--neuro-decim",
        type=int,
        default=4,
        help="Policy steps between neural-dashboard samples (4 = 25 Hz).",
    )
    parser.add_argument("--decimation", type=int, default=DEFAULT_DECIMATION)
    parser.add_argument(
        "--policy",
        default=None,
        help="Run name under out/rl/ - let the finished policy drive instead "
        "of the keyboard.",
    )
    parser.add_argument(
        "--follow",
        default=None,
        metavar="RUN",
        help="Watch a run learn: play its newest checkpoint and swap in a "
        "fresher one as training produces them. Works while training runs.",
    )
    parser.add_argument(
        "--conditioning",
        action="store_true",
        help="Run the olfactory conditioning protocol from flyplay.conditioning "
        "instead of a policy, and show the mushroom body doing it. Vision is "
        "off in this mode, so there is no compound-eye panel.",
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="The sealed room of flyplay.sandbox, furnished live from the page: "
        "sugar, shock zones, odours, coloured floor and obstacles.",
    )
    parser.add_argument(
        "--preset",
        default="scented_sugar",
        choices=sorted(PRESETS),
        help="Room the sandbox starts with (default: %(default)s).",
    )
    parser.add_argument(
        "--modality",
        default="odour",
        choices=("odour", "colour"),
        help="With --conditioning: the odour T-maze task, or the floor-colour "
        "assay of Vogt et al. (2014) with the eyes on.",
    )
    parser.add_argument(
        "--cs-plus",
        default=None,
        help="Stimulus paired with punishment first: A or B for odour (default "
        "A; reversal trains the other), blue or green for colour (default blue).",
    )
    parser.add_argument(
        "--train-trials",
        type=int,
        default=None,
        help="Training trials per block. Odour default 12: one trial drives the "
        "paired odour's valence to about -0.16 and the behavioural flip needs "
        "-0.35, so 12 clears it with room to spare, while run_session's 20 would "
        "push the session past the width of the trace window. Colour default 8, "
        "Vogt's four CS+ and four CS- periods.",
    )
    parser.add_argument(
        "--test-trials",
        type=int,
        default=None,
        help="Unreinforced test trials per block, averaged into one preference "
        "index (default 4 odour, 1 colour).",
    )
    parser.add_argument(
        "--no-reversal",
        action="store_true",
        help="Stop after acquisition instead of swapping the contingency and "
        "training again.",
    )
    parser.add_argument(
        "--memory",
        default=None,
        metavar="NAME",
        help="Restore a filed memory (memories/NAME.npz, see scripts/16_memory.py) "
        "and watch the fly act on it over test trials. Implies --conditioning.",
    )
    parser.add_argument(
        "--memory-trials",
        type=int,
        default=None,
        help="Test trials in a --memory session (default 12 for an odour "
        "memory, 2 for a colour one: a colour test is 90 s).",
    )
    parser.add_argument(
        "--reload-every",
        type=float,
        default=20.0,
        help="Seconds between checkpoint checks in --follow mode.",
    )
    parser.add_argument(
        "--wait-for-model",
        type=float,
        default=300.0,
        help="Seconds to wait for a freshly started run to write its first "
        "checkpoint before giving up.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.memory:
        args.conditioning = True
    if args.conditioning and (args.policy or args.follow):
        raise SystemExit("--conditioning cannot be combined with --policy or --follow.")
    if args.sandbox and (args.conditioning or args.policy or args.follow):
        raise SystemExit("--sandbox runs on its own; drop the other modes.")
    if args.speed is None:
        args.speed = 1.0 if (args.conditioning or args.sandbox) else 0.5

    server = FlyServer(args)
    sim_thread = threading.Thread(target=server.run, daemon=True)
    sim_thread.start()
    server.ready.wait()  # the model is built on that thread; wait for it
    if server.failure is not None:
        raise SystemExit(f"simulation failed to start: {server.failure}")
    if server.policy is not None:
        # Measuring every checkpoint means loading dozens of models; keep it
        # off the simulation thread so playback stays smooth.
        threading.Thread(target=server.scan_checkpoints, daemon=True).start()

    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(server))
    # Keep console output ASCII: a Windows console on the cp949 codepage
    # raises UnicodeEncodeError on an em dash and takes the server down with it.
    print(f"terrain '{args.terrain}' -> open  http://localhost:{args.port}")
    if args.sandbox:
        print(f"sandbox: preset '{args.preset}'; edit the room from the page")
    if args.memory:
        print(f"memory '{args.memory}' restored "
              f"({'colour' if server.colour else 'odour'}); {len(server.plan)} test trials")
    elif args.conditioning:
        blocks = []
        for key, kind, _odour, _focus in server.plan:
            if not blocks or blocks[-1][0] != key:
                blocks.append([key, 0])
            blocks[-1][1] += 1
        print(
            f"conditioning ({'colour' if server.colour else 'odour'}): "
            + ", ".join(f"{key} x{n}" for key, n in blocks)
        )
        print(f"{len(server.plan)} trials in the session; reset = a fresh fly")
    print("Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        httpd.server_close()


if __name__ == "__main__":
    main()

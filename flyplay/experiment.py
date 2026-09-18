"""A/B experiments that repeat themselves, and the records a report is written from.

A student's question -- "does the fly learn to avoid a smell paired with shock?"
-- becomes an *experiment set*: a question, a hypothesis, and two conditions
that differ in exactly one thing (A: shock on, B: shock off). Each condition is
a protocol of trials in order (train three times, then test twice), run by
many flies. At the end the hypothesis is checked by comparing A with B on one
measure, which is what makes the result a test rather than an anecdote.

    set          question, hypothesis, the deciding measure       `EXPERIMENT_SETS`
    condition    A or B: its steps and what it changes            `Condition`
    step         a room, a trial length, how many repeats         `Step`
    protocol     the set as it is run: conditions, flies, seed    `Protocol`

Fly *k* of condition A and fly *k* of condition B share a seed, so they have
the same mushroom-body wiring, the same exploratory random numbers and the same
start headings until the one difference between A and B makes them diverge.
Paired like that, a difference between A and B is the manipulation's doing.

`FlyRun` drives one fly through one condition on a `Sandbox`. Worker processes
(`scripts/17_run_experiment.py`) run many at once and write every trial to the
experiment's directory as it finishes; the viewer reads that directory to show
progress, replay chosen trials and build the report. There is one trial loop,
this one.
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from flyplay.sandbox import (
    METRICS_BY_KEY,
    ODOURS,
    PRESETS,
    TRIAL_METRICS,
    Sandbox,
    duration_ko,
    load_preset,
)

ROOT = Path(__file__).resolve().parents[1]
#: Every experiment gets a directory here (see `new_experiment_dir`).
EXPERIMENTS_DIR = ROOT / "out" / "experiments"

ROLE_LABELS = {"test": "시험", "train": "훈련", "repeat": "반복"}
#: Behaviour modes as small integers, for replay files.
MODES = ("explore", "search", "feed", "retract", "escape", "wall", "freeze")
#: Replay samples per simulated second. Walking cycles at about 12 Hz, so 25 Hz
#: would show legs jumping between poses at slow playback; 50 Hz costs 0.85 MB
#: per simulated minute compressed (measured) and replays 0.006 mm from the
#: recorded thorax on a model built from another seed.
REPLAY_HZ = 50


# --- what an experiment is made of ------------------------------------------------


@dataclass
class Step:
    """Trials of one kind in a condition's sequence."""

    role: str = "repeat"
    #: A preset name, or "current": the room saved with the protocol.
    room: str = "current"
    seconds: float = 60.0
    repeats: int = 1
    #: Seconds off-stage after this step's last trial (`Sandbox.rest`).
    rest_after: float = 0.0
    #: Seconds off-stage between this step's repeats.
    gap: float = 0.0


@dataclass
class Condition:
    """One arm of an A/B comparison."""

    label: str
    steps: list[Step]
    #: Keep the fly (and what it learned) from trial to trial; otherwise every
    #: trial starts a new fly.
    same_fly: bool = True
    hunger: float = 0.8
    #: Set hunger back to `hunger` at the start of every trial. Off, a fly that
    #: ate stays fed into the next trial.
    hold_hunger: bool = True
    #: Hunger during test trials only, if set.
    test_hunger: float | None = None
    #: Item kinds left out of every room of this condition.
    without: list[str] = field(default_factory=list)
    #: Overrides for every item of the kind, if set.
    volts: float | None = None
    molar: float | None = None
    volume: float | None = None
    strength: float | None = None

    def changes(self) -> list[str]:
        """This condition's overrides, in words."""
        kinds = {"sugar": "설탕", "shock": "전기", "odour": "냄새", "patch": "색 바닥", "obstacle": "벽·장애물",
                 "drum": "줄무늬 원통", "shadow": "그림자", "heat": "뜨거운 바닥", "light": "빛"}
        out = [f"{kinds.get(k, k)} 없음" for k in self.without]
        if self.volts is not None:
            out.append(f"전압 {self.volts:.0f} V")
        if self.molar is not None:
            out.append(f"설탕 농도 {self.molar:g} M")
        if self.volume is not None:
            out.append(f"설탕 양 {self.volume:.0f} nl")
        if self.strength is not None:
            out.append(f"냄새 세기 {self.strength:g}")
        if self.test_hunger is not None:
            out.append(f"시험 때 배고픔 {self.test_hunger:.2f}")
        return out


@dataclass
class Protocol:
    """An experiment set as it is actually run."""

    title: str
    set_key: str
    conditions: list[Condition]
    #: Flies per condition.
    flies: int = 8
    random_heading: bool = True
    seed: int = 1
    #: The room steps named "current" use, as item specs.
    layout: list[dict] = field(default_factory=list)
    #: Flies per condition whose trials are recorded for replay; -1 for all.
    replay_flies: int = 3
    #: The deciding measure and the expected direction, "A<B" or "A>B".
    metric: str = ""
    phase: str = "all"
    expect: str = ""
    #: What the student predicted before running: "A<B", "A>B" or "same".
    prediction: str = ""
    created: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Protocol:
        d = copy.deepcopy(d)
        d["conditions"] = [Condition(**{**c, "steps": [Step(**s) for s in c["steps"]]})
                           for c in d["conditions"]]
        return cls(**d)

    def trials_per_fly(self, condition: int) -> int:
        return sum(s.repeats for s in self.conditions[condition].steps)

    def total_trials(self) -> int:
        return self.flies * sum(self.trials_per_fly(c) for c in range(len(self.conditions)))

    def simulated_seconds(self) -> float:
        return self.flies * sum(s.seconds * s.repeats for c in self.conditions for s in c.steps)


@dataclass(frozen=True)
class TrialPlan:
    condition: int
    fly: int
    #: 1-based position in this fly's sequence.
    seq: int
    step: int
    #: 1-based within the step.
    repeat: int
    role: str
    room: str
    seconds: float
    #: Off-stage time just before this trial.
    rest_before: float
    new_fly: bool
    hunger: float | None


def fly_plan(protocol: Protocol, condition: int, fly: int) -> list[TrialPlan]:
    """The trials fly `fly` (1-based) runs in `condition`, in order."""
    cond = protocol.conditions[condition]
    plan, seq, rest = [], 0, 0.0
    for s, step in enumerate(cond.steps):
        for r in range(1, max(1, int(step.repeats)) + 1):
            seq += 1
            if step.role == "test" and cond.test_hunger is not None:
                hunger = cond.test_hunger
            elif seq == 1 or cond.hold_hunger or not cond.same_fly:
                hunger = cond.hunger
            else:
                hunger = None
            plan.append(TrialPlan(condition, fly, seq, s, r, step.role, step.room, float(step.seconds),
                                  rest, seq == 1 or not cond.same_fly, hunger))
            rest = step.gap if r < step.repeats else 0.0
        rest += step.rest_after
    return plan


def furnish(sandbox: Sandbox, room: str, protocol: Protocol, condition: Condition) -> None:
    """Empty the room and set it up for a trial: the preset or saved layout,
    minus what the condition leaves out, with its overrides applied."""
    if room == "current":
        sandbox.clear()
        for spec in protocol.layout:
            sandbox.place(spec["kind"], spec["x"], spec["y"], **spec.get("params", {}))
    else:
        load_preset(sandbox, room)
    for item in list(sandbox.room.items.values()):
        if item.kind in condition.without:
            sandbox.remove(item.id)
    for item in list(sandbox.room.items.values()):
        change = {}
        if item.kind == "shock" and condition.volts is not None:
            change["volts"] = condition.volts
        if item.kind == "sugar":
            if condition.molar is not None:
                change["molar"] = condition.molar
            if condition.volume is not None:
                change["volume"] = condition.volume
        if item.kind == "odour" and condition.strength is not None:
            change["strength"] = condition.strength
        if change:
            sandbox.update(item.id, **change)


def room_label(room: str) -> str:
    return "직접 꾸민 방" if room == "current" else PRESETS[room][0]


# --- one fly through one condition ----------------------------------------------------


class FlyRun:
    """Drive one fly through its trial sequence on `sandbox`.

    Call `begin` to set up the next trial, `Sandbox.step` repeatedly, and
    `tick` after every step: it returns the finished trial's record when the
    trial's time is up. `run` does all of that for a worker process.
    """

    def __init__(self, sandbox: Sandbox, protocol: Protocol, condition: int, fly: int, *,
                 replay_dir: Path | None = None):
        self.sb = sandbox
        self.protocol = protocol
        self.c = condition
        self.cond = protocol.conditions[condition]
        self.fly = fly
        self.plan = fly_plan(protocol, condition, fly)
        # Headings drawn from the fly's number alone: fly k of A and fly k of B
        # start every trial facing the same way.
        rng = np.random.default_rng([protocol.seed, fly, 7919])
        self.headings = [float(rng.uniform(0.0, 2.0 * np.pi)) for _ in self.plan]
        self.replay_dir = replay_dir
        self.records: list[dict] = []
        self.current: TrialPlan | None = None
        self._next = 0
        self.stopped = False

    @property
    def done(self) -> bool:
        return self.current is None and (self.stopped or self._next >= len(self.plan))

    def begin(self) -> bool:
        """Set up the next planned trial. False when none is left."""
        if self.stopped or self._next >= len(self.plan):
            return False
        pt = self.plan[self._next]
        heading = self.headings[self._next] if self.protocol.random_heading else None
        self._next += 1
        sb = self.sb
        if pt.rest_before > 0.0 and not pt.new_fly:
            sb.rest(pt.rest_before)
        furnish(sb, pt.room, self.protocol, self.cond)
        if pt.new_fly:
            sb.reset_fly(yaw=heading)
        else:
            sb.return_home(yaw=heading)
        if pt.hunger is not None:
            sb.hunger = float(np.clip(pt.hunger, 0.0, 1.0))
        sb.reset_counts()
        sb.refresh_telemetry()
        self.current = pt
        self._t0 = sb.time
        self._wall0 = time.perf_counter()
        self._heading = heading
        self._start_hunger = sb.hunger
        self._items = [sb.item_spec(i) for i in sb.room.items]
        self._events: list[list] = []
        self._last_event = sb.events[-1] if sb.events else None
        self._steps = 0
        self._frames: list[np.ndarray] = []
        self._tel: list[list[float]] = []
        self._radii: list[list[float]] = []
        self._sugar_ids = [i for i in sb.room.items if sb.room.items[i].kind == "sugar"]
        role = ROLE_LABELS.get(pt.role, pt.role)
        sb._event(f"{self.cond.label} · 파리 {self.fly} · {pt.seq}번째 시행 ({role} {pt.repeat}) 시작")
        return True

    def tick(self) -> dict | None:
        """Call after each `Sandbox.step`. The trial's record once its time is up."""
        if self.current is None:
            return None
        self._steps += 1
        self._collect()
        if self.sb.time - self._t0 >= self.current.seconds - 1e-9:
            return self._finish(stopped=False)
        return None

    def stop(self) -> dict | None:
        """End here: the running trial is recorded as stopped early, and no
        further trial begins."""
        self.stopped = True
        return self._finish(stopped=True) if self.current is not None else None

    def run(self, on_record=None, should_stop=None) -> list[dict]:
        """Every remaining trial, headless."""
        while self.begin():
            while True:
                self.sb.step()
                record = self.tick()
                if record is not None:
                    if on_record:
                        on_record(record)
                    break
                if should_stop is not None and self._steps % 100 == 0 and should_stop():
                    record = self.stop()
                    if record is not None and on_record:
                        on_record(record)
                    return self.records
        return self.records

    # --- recording ---------------------------------------------------------------

    def _collect(self) -> None:
        sb = self.sb
        if sb.events and sb.events[-1] is not self._last_event:
            fresh = []
            for event in reversed(sb.events):
                if event is self._last_event:
                    break
                fresh.append(event)
            self._events.extend([round(e["t"] - self._t0, 1), e["text"]] for e in reversed(fresh))
            self._last_event = sb.events[-1]
        if self.replay_dir is None:
            return
        steps_per_frame = max(1, round(sb.config.action_hz / REPLAY_HZ))
        if self._steps % steps_per_frame == 0:
            self._frames.append(sb.fs.sim.mj_data.qpos.astype(np.float32))
        if self._steps % 10 == 0:
            t = sb.telemetry
            dopamine = t.dopamine or {}
            values = sb.odour_valences()
            self._tel.append([
                round(sb.time - self._t0, 2), float(MODES.index(t.mode) if t.mode in MODES else 0),
                sb.hunger, dopamine.get("punish", 0.0), dopamine.get("sweet", 0.0),
                dopamine.get("nutrient", 0.0), *[float(v) for v in values],
                float(t.colour_valence.get("blue", 0.0)), float(t.colour_valence.get("green", 0.0)),
            ])
            self._radii.append([sb.room.items[i].half if i in sb.room.items else -1.0
                                for i in self._sugar_ids])

    def _finish(self, stopped: bool) -> dict:
        sb, pt = self.sb, self.current
        replay = None
        if self.replay_dir is not None and self._frames:
            self.replay_dir.mkdir(parents=True, exist_ok=True)
            name = f"c{self.c}_f{self.fly:03d}_t{pt.seq:02d}.npz"
            np.savez_compressed(
                self.replay_dir / name, qpos=np.stack(self._frames), hz=REPLAY_HZ,
                telemetry=np.asarray(self._tel, dtype=np.float32),
                sugar_ids=np.asarray(self._sugar_ids, dtype=np.int32),
                sugar_radii=np.asarray(self._radii, dtype=np.float32).reshape(len(self._radii), -1),
                nq=sb.fs.sim.mj_model.nq,
            )
            replay = f"replay/{name}"
        record = {
            "condition": self.c,
            "condition_label": self.cond.label,
            "fly": self.fly,
            "seed": self.protocol.seed + self.fly,
            "seq": pt.seq,
            "step": pt.step,
            "repeat": pt.repeat,
            "role": pt.role,
            "room": pt.room,
            "room_label": room_label(pt.room),
            "seconds": round(sb.time - self._t0, 2),
            "planned_seconds": pt.seconds,
            "stopped": stopped,
            "new_fly": pt.new_fly,
            "heading_deg": None if self._heading is None else round(math.degrees(self._heading) % 360.0, 1),
            "start_hunger": round(self._start_hunger, 3),
            "rest_before_s": pt.rest_before,
            "metrics": sb.trial_metrics(),
            "seen": sb.seen_contents(),
            "path": [list(p) for p in sb.path],
            "items": self._items,
            "events": self._events,
            "replay": replay,
            "wall_s": round(time.perf_counter() - self._wall0, 1),
            #: Wall-clock time the trial ended, for the viewer's "follow new trials".
            "finished": round(time.time(), 1),
        }
        self.records.append(record)
        self.current = None
        return record


# --- the experiment directory ------------------------------------------------------------


def new_experiment_dir(root: Path = EXPERIMENTS_DIR, tag: str = "exp") -> Path:
    """A fresh, ASCII-named directory: out/experiments/20260917-183512-exp."""
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = root / f"{stamp}-{tag}"
    n = 2
    while path.exists():
        path = root / f"{stamp}-{tag}{n}"
        n += 1
    path.mkdir()
    (path / "records").mkdir()
    return path


def write_json(path: Path, data) -> None:
    """Written whole and renamed into place, so a reader never sees half a file.

    Windows refuses the rename while another process has the file open, and the
    viewer opens status.json every few seconds to show progress: a run died on
    exactly that after 20 of 20 trials (PermissionError, WinError 5), before its
    CSV and report were written. The rename is retried for up to a second."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    for attempt in range(40):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.025)


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def read_records(directory: Path) -> list[dict]:
    """Every finished trial so far, ordered by condition, fly and sequence."""
    records = []
    for path in sorted((directory / "records").glob("*.jsonl")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                records.append(json.loads(line))
            except ValueError:
                pass  # a line still being written
    return sorted(records, key=lambda r: (r["condition"], r["fly"], r["seq"]))


def append_record(directory: Path, record: dict) -> None:
    path = directory / "records" / f"c{record['condition']}_f{record['fly']:03d}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_fly_job(job: dict) -> dict:
    """Worker entry point: one fly through one condition, every trial appended
    to the experiment directory as it finishes. Stops early, keeping what was
    recorded, when a file named STOP appears in the directory."""
    directory = Path(job["dir"])
    protocol = Protocol.from_dict(read_json(directory / "protocol.json"))
    c, fly = int(job["condition"]), int(job["fly"])
    started = time.perf_counter()
    stop_file = directory / "STOP"
    if stop_file.exists():
        return {"condition": c, "fly": fly, "trials": 0, "wall_s": 0.0, "stopped": True}
    sandbox = Sandbox(seed=protocol.seed + fly)
    replay = protocol.replay_flies < 0 or fly <= protocol.replay_flies
    run = FlyRun(sandbox, protocol, c, fly, replay_dir=directory / "replay" if replay else None)
    count = [0]

    def keep(record: dict) -> None:
        append_record(directory, record)
        count[0] += 1

    try:
        run.run(on_record=keep, should_stop=stop_file.exists)
    finally:
        sandbox.close()
    return {"condition": c, "fly": fly, "trials": count[0],
            "wall_s": round(time.perf_counter() - started, 1), "stopped": run.stopped}


def jobs_for(protocol: Protocol, directory: Path) -> list[dict]:
    """Fly 1 of every condition, then fly 2 of every condition...: a run stopped
    halfway still has matched pairs."""
    return [{"dir": str(directory), "condition": c, "fly": f}
            for f in range(1, protocol.flies + 1) for c in range(len(protocol.conditions))]


# --- was the hypothesis right? --------------------------------------------------------


#: Pairs of flies needed before the report gives any verdict. With one or two
#: pairs every split is a coin flip's worth (sign test p = 1.0 or 0.5).
MIN_PAIRS = 3

PHASE_LABELS = {
    "all": "모든 시행",
    "test": "훈련 뒤 시험 시행",
    "last": "마지막 두 시행",
    "first": "첫 시행",
}


def phase_records(records: list[dict], phase: str) -> list[dict]:
    """One fly's records (ordered by sequence) that a phase selects."""
    if phase == "test":
        trains = [r["seq"] for r in records if r["role"] == "train"]
        after = max(trains) if trains else 0
        chosen = [r for r in records if r["role"] == "test" and r["seq"] > after]
        return chosen or [r for r in records if r["role"] == "test"]
    if phase == "last":
        return records[-2:]
    if phase == "first":
        return records[:1]
    return records


def metric_value(record: dict, key: str) -> float | None:
    """A record's value for a measure, reading a latency that never ended as
    the whole trial (the fly did not get there in the time it had)."""
    metric = METRICS_BY_KEY[key]
    value = record["metrics"].get(key)
    if value is None and metric.latency and all(n in record.get("seen", []) for n in metric.needs):
        return float(record["seconds"])
    return None if value is None else float(value)


def fly_scores(records: list[dict], key: str, phase: str) -> dict[int, dict[int, float]]:
    """condition -> fly -> the fly's mean over its phase trials."""
    by_fly: dict[tuple[int, int], list[dict]] = {}
    for r in records:
        by_fly.setdefault((r["condition"], r["fly"]), []).append(r)
    out: dict[int, dict[int, float]] = {}
    for (c, fly), rs in by_fly.items():
        rs.sort(key=lambda r: r["seq"])
        values = [v for v in (metric_value(r, key) for r in phase_records(rs, phase)) if v is not None]
        if values:
            out.setdefault(c, {})[fly] = float(np.mean(values))
    return out


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact binomial probability of a split at least this uneven,
    if either direction were equally likely -- how often coin flips would do it."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = max(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2.0**n
    return min(1.0, 2.0 * tail)


#: What "believable" means to `pairs_needed`: the sign test's probability under
#: 5% in 8 of 10 reruns of the same experiment. The textbook convention (alpha
#: 0.05, power 0.8), picked for wording like the verdict's thresholds.
BELIEVE_P = 0.05
BELIEVE_REPEATS = 0.8
#: Where `pairs_needed` gives up: past this the difference is too small, or too
#: mixed, for any run the lab offers (at most 30 flies per condition, 50 from
#: the command line).
PAIRS_NEEDED_CAP = 200
#: Effect sizes as words, upper bounds of |d_z|: Cohen's small, medium and large.
EFFECT_WORDS = ((0.2, "거의 없음"), (0.5, "작음"), (0.8, "보통"), (math.inf, "큼"))


def paired_effect(diffs: list[float]) -> float | None:
    """How big the difference is next to how much it wobbles from pair to pair:
    the mean pair difference over its standard deviation (Cohen's d_z). None
    under `MIN_PAIRS` pairs. When every pair differs by exactly the same amount
    the ratio is infinite; it is reported as 99 so the payload stays JSON
    (``Infinity`` is not, and the page's JSON.parse rejects it)."""
    if len(diffs) < MIN_PAIRS:
        return None
    arr = np.asarray(diffs, dtype=float)
    sd = float(arr.std(ddof=1))
    mean = float(arr.mean())
    if sd == 0.0:
        return 0.0 if mean == 0.0 else math.copysign(99.0, mean)
    return float(np.clip(mean / sd, -99.0, 99.0))


def effect_words(effect: float | None) -> str | None:
    if effect is None:
        return None
    return next(words for bound, words in EFFECT_WORDS if abs(effect) < bound)


def _upper_tails(n: int, p: float) -> list[float]:
    """P(X >= k) for k = 0 .. n+1, X ~ Binomial(n, p), from log-space terms."""
    if p <= 0.0:
        return [1.0] + [0.0] * (n + 1)
    if p >= 1.0:
        return [1.0] * (n + 1) + [0.0]
    lp, lq, lf = math.log(p), math.log1p(-p), math.lgamma(n + 1)
    tails = [0.0] * (n + 2)
    for k in range(n, -1, -1):
        tails[k] = tails[k + 1] + math.exp(lf - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * lp + (n - k) * lq)
    return [min(1.0, t) for t in tails]


def pairs_needed(agree: int, disagree: int, ties: int) -> int | None:
    """Pairs a rerun would need to show a split like this one believably.

    Suppose each pair of the rerun goes this run's way with probability
    ``(agree + 1) / (agree + disagree + 2)`` -- this run's share pulled toward a
    coin flip, since three pairs out of three is weak evidence that every pair
    would -- and ties as often as here. Returns the fewest pairs for which
    `sign_test` would come out under `BELIEVE_P` in `BELIEVE_REPEATS` of reruns,
    or None when this run points nowhere (no more agreeing pairs than
    disagreeing) or would need more than `PAIRS_NEEDED_CAP`.

    Computed: 3 agreeing pairs of 3 need 20 pairs, 8 of 8 need 12, 6 of 8
    need 49, 5 of 8 need 199; 4 of 4 with 4 ties need 36. About 11 ms a call.
    """
    decided = agree + disagree
    if decided == 0 or agree <= disagree:
        return None
    q = (agree + 1) / (decided + 2)
    decide_share = decided / (decided + ties)
    # Power of the sign test with d decided pairs, for every d up to the cap.
    power = [0.0] * (PAIRS_NEEDED_CAP + 1)
    for d in range(1, PAIRS_NEEDED_CAP + 1):
        null = _upper_tails(d, 0.5)
        k = next((k for k in range(d + 1) if 2.0 * null[k] <= BELIEVE_P), None)
        if k is not None:
            power[d] = _upper_tails(d, q)[k]
    for n in range(MIN_PAIRS, PAIRS_NEEDED_CAP + 1):
        # How many of n pairs are decided is itself binomial.
        decided_at_least = _upper_tails(n, decide_share)
        chance = sum((decided_at_least[d] - decided_at_least[d + 1]) * power[d] for d in range(1, n + 1))
        if chance >= BELIEVE_REPEATS:
            return n
    return None


def evaluate(protocol: Protocol, records: list[dict], expect: str | None = None) -> dict:
    """Compare condition A (0) with B (1) on the protocol's deciding measure.

    Each fly's score is its mean over the phase's trials; flies pair up by
    number. The verdict needs both the size of the difference (beyond the
    measure's tolerance) and its consistency (at least 70% of pairs going that
    way). Both thresholds are choices for wording a middle-school report, not
    a statistical test; the sign test's probability is reported beside them.
    """
    key, phase = protocol.metric, protocol.phase
    expect = expect or protocol.expect
    metric = METRICS_BY_KEY[key]
    scores = fly_scores(records, key, phase)
    a, b = scores.get(0, {}), scores.get(1, {})

    def summary(values: list[float]) -> dict | None:
        if not values:
            return None
        arr = np.asarray(values)
        return {"n": len(arr), "mean": float(arr.mean()), "min": float(arr.min()), "max": float(arr.max()),
                "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0}

    pairs = sorted(set(a) & set(b))
    diffs = [a[f] - b[f] for f in pairs]
    tol = metric.tolerance
    higher = sum(d > tol for d in diffs)
    lower = sum(d < -tol for d in diffs)
    result = {
        "metric": key, "column": metric.column, "phase": phase, "phase_label": PHASE_LABELS.get(phase, phase),
        "expect": expect, "tolerance": tol,
        "a": summary(list(a.values())), "b": summary(list(b.values())),
        "pairs": len(pairs), "a_higher": higher, "a_lower": lower, "ties": len(pairs) - higher - lower,
        "p_sign": sign_test(higher, lower),
        "scores": {"a": a, "b": b},
    }
    if not pairs or result["a"] is None or result["b"] is None:
        result["verdict"] = "not_enough"
        return result
    d = result["a"]["mean"] - result["b"]["mean"]
    result["difference"] = d
    direction = "A>B" if d > tol else "A<B" if d < -tol else "same"
    result["direction"] = direction
    consistent = (higher if direction == "A>B" else lower if direction == "A<B" else len(pairs) - higher - lower)
    result["consistent"] = consistent
    result["effect"] = paired_effect(diffs)
    result["effect_words"] = effect_words(result["effect"])
    # Flies per condition to show this run's difference believably; nothing to
    # show when the means are within the tolerance.
    result["pairs_needed"] = None
    if direction != "same" and len(pairs) >= MIN_PAIRS:
        agree, disagree = (higher, lower) if direction == "A>B" else (lower, higher)
        result["pairs_needed"] = pairs_needed(agree, disagree, len(pairs) - higher - lower)
    steady = consistent >= math.ceil(0.7 * len(pairs))
    if len(pairs) < MIN_PAIRS:
        result["verdict"] = "too_few"
    elif expect == "same":
        result["verdict"] = "supported" if direction == "same" else ("opposite" if steady else "unclear")
    elif direction == expect:
        result["verdict"] = "supported" if steady else "leaning"
    elif direction == "same":
        result["verdict"] = "no_difference"
    else:
        result["verdict"] = "opposite" if steady else "unclear"
    return result


VERDICT_TEXT = {
    "too_few": "아직 비교한 쌍이 3쌍보다 적어 판단하기 이릅니다",
    "supported": "가설이 맞았습니다",
    "leaning": "평균은 가설 쪽이지만 파리마다 달라 확실하지 않습니다",
    "no_difference": "A와 B의 차이가 거의 없어 가설이 맞았다고 보기 어렵습니다",
    "opposite": "가설과 반대로 나왔습니다",
    "unclear": "파리마다 결과가 엇갈려 판단하기 어렵습니다",
    "not_enough": "아직 비교할 기록이 부족합니다",
}


# --- the sets -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentSet:
    key: str
    group: str
    title: str
    #: The inquiry question, as a student would write it.
    question: str
    #: What A and B are, in a few words each.
    a: str
    b: str
    #: The one thing that differs (the manipulated variable), in words.
    changed: str
    conditions: tuple[Condition, Condition]
    metric: str
    phase: str
    expect: str
    hypothesis: str
    #: Why the hypothesis is expected: the model's mechanism, plainly.
    why: str
    #: How real experiments did it, with sources.
    basis: str
    flies: int = 8
    #: Other measures worth charting beside the deciding one.
    also: tuple[str, ...] = ()
    #: The hypotheses a student picks from, in the order the page lists them:
    #: (direction, sentence) for "A<B" and for "A>B". The measure's words
    #: ("A의 '처음 먹기(초)' 값이 B보다 작을 것이다") were what the page first
    #: offered, and a student had to work out which letter was which room.
    outcomes: tuple[tuple[str, str], ...] = ()


def _steps(*items) -> list[Step]:
    return [Step(*i) if isinstance(i, tuple) else i for i in items]


def _learning(train_room: str, test_room: str, trains: int = 3, tests: int = 2,
              seconds: float = 60.0, rest: float = 0.0) -> list[Step]:
    return [Step("train", train_room, seconds, trains, rest_after=rest),
            Step("test", test_room, seconds, tests)]


EXPERIMENT_SETS: dict[str, ExperimentSet] = {}


def _add(s: ExperimentSet) -> None:
    if s.outcomes:
        assert sorted(d for d, _ in s.outcomes) == ["A<B", "A>B"], s.key
    EXPERIMENT_SETS[s.key] = s


# ---- finding food
_add(ExperimentSet(
    "smell_guides", "먹이 찾기", "냄새가 먹이 찾기를 도울까?",
    "설탕물에 식초 냄새가 나면 초파리가 설탕을 더 빨리 찾을까?",
    "식초 냄새 있음", "냄새 없음", "설탕 옆 식초 냄새가 있는가",
    (Condition("A: 식초 냄새 있음", [Step("repeat", "scented_sugar", 60.0, 1)], same_fly=False),
     Condition("B: 냄새 없음", [Step("repeat", "plain_sugar", 60.0, 1)], same_fly=False)),
    "first_feed_s", "all", "A<B",
    "식초 냄새가 있는 A가 냄새 없는 B보다 설탕을 먹기 시작하기까지 걸리는 시간이 짧을 것이다.",
    "식초에는 타고난 끌림이 있어서 초파리가 냄새가 진해지는 쪽으로 돕니다. 냄새가 없으면 우연히 밟을 때까지 "
    "돌아다녀야 합니다.",
    "걷는 초파리는 먹이 냄새를 맡으면 그쪽으로 방향을 바꿉니다(Álvarez-Salvado 등 2018, eLife). 학교 실험에서는 "
    "양쪽 끝에 다른 물질을 둔 선택 상자에 초파리를 넣고 몇 분 뒤 양쪽 마릿수를 셉니다(AP Biology Lab 12).",
    flies=10, also=("first_touch_s", "feed_s"),
    outcomes=(
        ("A<B", "냄새가 있는 A가 설탕을 더 빨리 먹기 시작할 것이다"),
        ("A>B", "냄새가 없는 B가 설탕을 더 빨리 먹기 시작할 것이다"),
    ),
))
_add(ExperimentSet(
    "hunger_feeds_longer", "먹이 찾기", "배고플수록 더 오래 먹을까?",
    "배고픈 초파리가 배부른 초파리보다 설탕을 더 오래 먹을까?",
    "배고픔 0.9", "배고픔 0.3", "시작할 때 배고픔",
    (Condition("A: 배고픔 0.9", [Step("repeat", "big_scented_sugar", 60.0, 1)], same_fly=False, hunger=0.9),
     Condition("B: 배고픔 0.3", [Step("repeat", "big_scented_sugar", 60.0, 1)], same_fly=False, hunger=0.3)),
    "feed_s", "all", "A>B",
    "배고픔이 0.9인 A가 0.3인 B보다 설탕을 먹은 시간이 길 것이다.",
    "이 모델에서는 배고플수록 먹다가 멈출 확률이 낮습니다(굶주리면 초당 0.05번, 배부르면 1번). 식초의 타고난 "
    "끌림도 배고픔만큼 커집니다.",
    "flyPAD는 초파리가 먹는 양을 한 모금(약 1 nl) 단위로 세는 장치입니다(Itskov 등 2014, Nat Commun). 이런 장치로 "
    "굶긴 시간에 따라 먹는 양이 어떻게 달라지는지 잽니다.",
    flies=10, also=("first_feed_s", "end_hunger"),
    outcomes=(
        ("A>B", "배고픈 A가 설탕을 더 오래 먹을 것이다"),
        ("A<B", "덜 배고픈 B가 설탕을 더 오래 먹을 것이다"),
    ),
))
_add(ExperimentSet(
    "taste_needed", "먹이 찾기", "맛이 없으면 영양이 있어도 안 먹을까?",
    "단맛이 없는 설탕(소르비톨)은 영양이 있어도 초파리가 먹지 않을까?",
    "자당 (단맛+영양)", "소르비톨 (영양만)", "설탕의 종류",
    (Condition("A: 자당", [Step("repeat", "big_scented_sugar", 60.0, 1)], same_fly=False),
     Condition("B: 소르비톨", [Step("repeat", "sorbitol_only", 60.0, 1)], same_fly=False)),
    "feed_s", "all", "A>B",
    "단맛이 있는 자당(A)을 소르비톨(B)보다 오래 먹을 것이다.",
    "초파리는 발끝의 맛 감각으로 먹기를 시작합니다. 이 모델에서 소르비톨은 단맛이 0이라 먹기가 시작되지 않습니다.",
    "배고픈 초파리는 소르비톨을 물보다 더 마시지 않았습니다(Burke & Waddell 2011, Curr Biol).",
    flies=8, also=("first_touch_s",),
    outcomes=(
        ("A>B", "자당(A)을 소르비톨(B)보다 더 오래 먹을 것이다"),
        ("A<B", "소르비톨(B)을 자당(A)보다 더 오래 먹을 것이다"),
    ),
))
# No set asks whether a sweet sugar without calories fills the fly: hunger
# falls only with calories by construction, so both answers were written in
# advance (removed at the user's request).
_add(ExperimentSet(
    "drop_size", "먹이 찾기", "방울이 크면 더 빨리 찾을까?",
    "냄새가 없을 때, 큰 설탕 방울이 작은 방울보다 더 빨리 발에 닿을까?",
    "300 nl 방울 12개", "20 nl 방울 12개", "방울의 크기",
    (Condition("A: 300 nl", [Step("repeat", "scattered_plain", 60.0, 1)], same_fly=False, volume=300.0),
     Condition("B: 20 nl", [Step("repeat", "scattered_plain", 60.0, 1)], same_fly=False, volume=20.0)),
    "first_touch_s", "all", "A<B",
    "큰 방울(A)이 작은 방울(B)보다 처음 발에 닿기까지 걸리는 시간이 짧을 것이다.",
    "냄새가 없으면 우연히 밟아야 찾습니다. 방울이 넓을수록 밟힐 자리가 많습니다. 방울 하나만 두었더니 60초 안에 "
    "둘 다 못 찾아서(시범 실행) 12개를 흩어 놓았습니다.",
    "실제 연구에서 확인한 내용이 아니라 이 모델의 예측입니다. 이 방에서 방울은 보이도록 실제보다 훨씬 크게 그려집니다.",
    flies=10, also=("first_feed_s",),
    outcomes=(
        ("A<B", "큰 방울(A)에 더 빨리 닿을 것이다"),
        ("A>B", "작은 방울(B)에 더 빨리 닿을 것이다"),
    ),
))

# ---- learning
_add(ExperimentSet(
    "shock_odour", "배우기", "전기와 함께 맡은 냄새를 피하게 될까?",
    "MCH 냄새가 날 때만 전기를 주면, 시험에서 초파리가 MCH를 피할까?",
    "훈련 때 전기 켬", "훈련 때 전기 끔", "훈련 방의 전기",
    (Condition("A: 전기 켬", _learning("odour_shock", "odour_choice"), hunger=0.5),
     Condition("B: 전기 끔", _learning("odour_shock", "odour_choice"), hunger=0.5, volts=0.0)),
    "odour_pi", "test", "A>B",
    "훈련 때 전기를 받은 A가 B보다 시험에서 MCH를 더 피해 냄새 선호(옥탄올+)가 클 것이다.",
    "전기가 오는 순간 켜져 있던 냄새 세포에서 '다가가기' 출력으로 가는 시냅스가 약해집니다. 그래서 MCH의 가치가 "
    "음수가 되고, 시험 방에서 MCH 냄새가 진해지는 쪽을 피해 돕니다.",
    "T자 미로 실험: 한 냄새에 전기(60 V)를 주고 다른 냄새에는 주지 않은 뒤 두 냄새 사이에서 고르게 합니다"
    "(Tully & Quinn 1985, J Comp Physiol A). 교과서적인 초파리 학습 실험입니다.",
    flies=8, also=("value_mch", "shocks", "near_mch_pct"),
    outcomes=(
        ("A>B", "전기를 받은 A가 MCH를 더 피할 것이다"),
        ("A<B", "전기를 받지 않은 B가 MCH를 더 피할 것이다"),
    ),
))
_add(ExperimentSet(
    "mirror_odour", "배우기", "어느 냄새에 전기를 줘도 똑같이 배울까?",
    "전기를 MCH에 주든 옥탄올에 주든, 초파리는 전기를 받은 냄새를 피할까? (거울 실험)",
    "MCH에 전기", "옥탄올에 전기", "전기를 받은 냄새",
    (Condition("A: MCH에 전기", _learning("odour_shock", "odour_choice"), hunger=0.5),
     Condition("B: 옥탄올에 전기", _learning("octanol_shock", "odour_choice"), hunger=0.5)),
    "odour_pi", "test", "A>B",
    "A는 MCH를, B는 옥탄올을 피해서 냄새 선호(옥탄올+)가 A에서 크고 B에서 작을 것이다.",
    "냄새 자체를 원래 좋아하거나 싫어해서 생긴 차이라면 A와 B가 같은 쪽으로 기웁니다. 반대로 기울면 전기를 받은 "
    "냄새를 배운 것입니다.",
    "실제 학습 실험은 이렇게 짝지은 냄새를 바꾼 두 무리를 함께 돌려 타고난 선호를 지웁니다(Michels 등 2017, "
    "초보자용 초파리 유충 학습 실습서).",
    flies=8, also=("value_mch", "value_octanol"),
    outcomes=(
        ("A>B", "A는 MCH를, B는 옥탄올을 더 피할 것이다 (전기를 받은 냄새를 피함)"),
        ("A<B", "A는 옥탄올을, B는 MCH를 더 피할 것이다 (전기를 받지 않은 냄새를 피함)"),
    ),
))
_add(ExperimentSet(
    "shock_colour", "배우기", "전기와 함께 밟은 색을 피하게 될까?",
    "파란 바닥에서만 전기를 주면, 시험에서 초파리가 파랑을 피할까?",
    "훈련 때 전기 켬", "훈련 때 전기 끔", "훈련 방의 전기",
    (Condition("A: 전기 켬", _learning("blue_shock", "colour_quadrants")),
     Condition("B: 전기 끔", _learning("blue_shock", "colour_quadrants"), volts=0.0)),
    "colour_pi", "test", "A<B",
    "훈련 때 파랑에서 전기를 받은 A가 B보다 시험에서 파랑 쪽에 덜 머물러 색 선호(파랑+)가 작을 것이다.",
    "눈으로 본 바닥 색도 냄새처럼 케니언 세포를 켭니다. 전기가 오는 순간 본 파랑의 가치가 음수가 됩니다.",
    "걷는 초파리에게 파랑·초록 바닥과 전기를 짝지어 색을 배우게 했습니다(Vogt 등 2014, eLife).",
    flies=8, also=("value_blue", "shocks", "blue_pct"),
    outcomes=(
        ("A<B", "전기를 받은 A가 파랑을 더 피할 것이다"),
        ("A>B", "전기를 받지 않은 B가 파랑을 더 피할 것이다"),
    ),
))
_add(ExperimentSet(
    "sugar_odour", "배우기", "설탕과 함께 맡은 냄새를 좋아하게 될까?",
    "옥탄올 냄새가 나는 곳에서 설탕을 먹게 하면, 시험에서 초파리가 옥탄올 쪽을 좋아할까?",
    "훈련 때 설탕 있음", "훈련 때 설탕 없음", "훈련 방의 설탕",
    (Condition("A: 설탕 있음", _learning("octanol_sugar", "odour_choice"), hunger=0.9),
     Condition("B: 설탕 없음", _learning("octanol_sugar", "odour_choice"), hunger=0.9, without=["sugar"])),
    "odour_pi", "test", "A>B",
    "훈련 때 옥탄올 옆 설탕을 먹은 A가 B보다 시험에서 옥탄올 쪽에 더 머물 것이다.",
    "배고픈 파리가 설탕을 먹는 동안 켜져 있던 냄새 세포에서 '피하기' 출력으로 가는 시냅스가 약해져 옥탄올의 "
    "가치가 양수가 됩니다.",
    "냄새와 설탕을 짝지은 학습은 배고픈 초파리에서 오래가는 기억을 만듭니다(Krashes & Waddell 2008, J Neurosci).",
    flies=8, also=("value_octanol", "feed_s"),
    outcomes=(
        ("A>B", "설탕을 먹은 A가 옥탄올 쪽에 더 머물 것이다"),
        ("A<B", "설탕 없이 훈련한 B가 옥탄올 쪽에 더 머물 것이다"),
    ),
))
_add(ExperimentSet(
    "sugar_colour", "배우기", "설탕을 먹은 바닥 색을 좋아하게 될까?",
    "초록 바닥에서 설탕을 먹게 하면, 시험에서 초파리가 초록 쪽을 좋아할까?",
    "훈련 때 설탕 있음", "훈련 때 설탕 없음", "훈련 방의 설탕",
    (Condition("A: 설탕 있음", _learning("green_reward", "colour_quadrants"), hunger=0.9),
     Condition("B: 설탕 없음", _learning("green_reward", "colour_quadrants"), hunger=0.9, without=["sugar"])),
    "colour_pi", "test", "A<B",
    "초록에서 설탕을 먹은 A가 B보다 시험에서 초록 쪽에 더 머물러 색 선호(파랑+)가 작을 것이다.",
    "설탕을 먹는 동안 본 초록의 가치가 양수가 됩니다. 두 방 모두 옅은 식초 냄새는 같습니다.",
    "걷는 초파리의 색 학습은 설탕 보상으로도 되고, 훈련을 1·2·4·8번으로 늘릴수록 기억이 커졌습니다"
    "(Schnaitmann 등 2010, Front Behav Neurosci; Vogt 등 2014). 훈련하지 않은 초파리는 원래 초록을 조금 더 좋아했습니다.",
    flies=8, also=("value_green", "feed_s"),
    outcomes=(
        ("A<B", "초록에서 설탕을 먹은 A가 초록 쪽에 더 머물 것이다"),
        ("A>B", "설탕을 먹지 않은 B가 초록 쪽에 더 머물 것이다"),
    ),
))
_add(ExperimentSet(
    "fed_hides_memory", "배우기", "배가 부르면 설탕 기억을 쓰지 않을까?",
    "똑같이 설탕과 냄새를 배운 초파리도, 시험 때 배가 부르면 그 냄새를 덜 따라갈까?",
    "시험 때 배고픔", "시험 때 배부름", "시험할 때의 배고픔",
    (Condition("A: 시험 때 배고픔 0.9", _learning("octanol_sugar", "odour_choice"), hunger=0.9),
     Condition("B: 시험 때 배고픔 0.1", _learning("octanol_sugar", "odour_choice"), hunger=0.9, test_hunger=0.1)),
    "odour_pi", "test", "A>B",
    "훈련은 똑같이 받았어도 시험 때 배고픈 A가 배부른 B보다 옥탄올 쪽에 더 머물 것이다.",
    "이 모델에서 설탕 기억은 배고픈 만큼만 행동으로 나옵니다(가치 = 배고픔 × 설탕 기억 - 전기 기억).",
    "설탕 기억은 먹이를 먹이면 행동에서 사라졌다가 다시 굶기면 돌아옵니다(Krashes 등 2009, Cell).",
    flies=8, also=("value_octanol",),
    outcomes=(
        ("A>B", "시험 때 배고픈 A가 옥탄올 쪽에 더 머물 것이다"),
        ("A<B", "시험 때 배부른 B가 옥탄올 쪽에 더 머물 것이다"),
    ),
))
_add(ExperimentSet(
    "more_training", "배우기", "훈련을 많이 할수록 더 잘 배울까?",
    "전기 훈련을 네 번 받은 초파리가 한 번 받은 초파리보다 냄새를 더 피할까?",
    "훈련 4번", "훈련 1번", "훈련 횟수",
    (Condition("A: 훈련 4번", _learning("odour_shock", "odour_choice", trains=4), hunger=0.5, volts=20.0),
     Condition("B: 훈련 1번", _learning("odour_shock", "odour_choice", trains=1), hunger=0.5, volts=20.0)),
    "odour_pi", "test", "A>B",
    "훈련을 네 번 받은 A가 한 번 받은 B보다 시험에서 MCH를 더 피할 것이다.",
    "전기를 받을 때마다 시냅스가 조금씩 더 약해집니다. 한 번에 다 배우지 않도록 약한 20 V를 씁니다.",
    "T자 미로에서 전기 횟수를 늘리면 기억 점수가 커지다가 한 번의 훈련 안에서 포화합니다(Tully & Quinn 1985). "
    "한 번씩 따로 주는 훈련은 횟수가 늘수록 점수가 올랐습니다(Beck 등 2000, J Neurosci).",
    flies=8, also=("value_mch", "shocks"),
    outcomes=(
        ("A>B", "훈련을 4번 받은 A가 MCH를 더 피할 것이다"),
        ("A<B", "훈련을 1번 받은 B가 MCH를 더 피할 것이다"),
    ),
))
_add(ExperimentSet(
    "stronger_shock", "배우기", "전기가 셀수록 더 잘 배울까?",
    "센 전기(60 V)로 훈련한 초파리가 약한 전기(10 V)로 훈련한 초파리보다 파랑을 더 피할까?",
    "60 V", "10 V", "훈련 전기의 세기",
    (Condition("A: 60 V", _learning("blue_shock", "colour_quadrants")),
     Condition("B: 10 V", _learning("blue_shock", "colour_quadrants"), volts=10.0)),
    "colour_pi", "test", "A<B",
    "60 V로 훈련한 A가 10 V로 훈련한 B보다 시험에서 파랑 쪽에 덜 머물 것이다.",
    "이 모델에서 처벌 신호는 전압에 따라 커지다가 포화합니다(10 V 0.11, 60 V 0.96).",
    "걷는 초파리의 색 학습은 15 V부터 나타나고 30~120 V에서 비슷했습니다(Vogt 등 2014).",
    flies=8, also=("value_blue",),
    outcomes=(
        ("A<B", "센 전기(60 V)로 훈련한 A가 파랑을 더 피할 것이다"),
        ("A>B", "약한 전기(10 V)로 훈련한 B가 파랑을 더 피할 것이다"),
    ),
))
_add(ExperimentSet(
    "memory_fades", "배우기", "하루가 지나면 기억이 흐려질까?",
    "전기 훈련 바로 뒤와 하루 뒤에 시험하면, 하루 뒤에 냄새를 덜 피할까?",
    "바로 시험", "하루 쉬고 시험", "훈련과 시험 사이의 시간",
    (Condition("A: 바로 시험", _learning("odour_shock", "odour_choice"), hunger=0.5),
     Condition("B: 하루 뒤 시험", _learning("odour_shock", "odour_choice", rest=86400.0), hunger=0.5)),
    "odour_pi", "test", "A>B",
    "바로 시험한 A가 하루 쉬고 시험한 B보다 MCH를 더 피할 것이다.",
    "빠르게 배운 기억은 몇 시간에 걸쳐 되돌아가고, 천천히 배운 기억만 며칠 갑니다. 쉬는 시간은 계산으로만 "
    "흐르고 시뮬레이션하지 않습니다.",
    "T자 미로에서 한 번 훈련한 기억은 7시간에 걸쳐 줄고 24시간 뒤에는 조금만 남았습니다(Tully & Quinn 1985). "
    "여러 번 나눠 훈련하면 며칠 가는 기억이 생기지만(Tully 등 1994, Cell), 이 모델에는 그 과정이 없습니다.",
    flies=8, also=("value_mch",),
    outcomes=(
        ("A>B", "바로 시험한 A가 MCH를 더 피할 것이다"),
        ("A<B", "하루 뒤에 시험한 B가 MCH를 더 피할 것이다"),
    ),
))
_add(ExperimentSet(
    "same_fly_faster", "배우기", "같은 파리는 반복할수록 빨라질까?",
    "같은 초파리에게 옥탄올 냄새 나는 설탕 찾기를 여러 번 시키면, 매번 새 초파리보다 빨리 찾게 될까?",
    "같은 파리 (기억 유지)", "매번 새 파리", "기억이 이어지는가",
    (Condition("A: 같은 파리", [Step("repeat", "octanol_sugar_far", 45.0, 6)], hunger=0.9),
     Condition("B: 매번 새 파리", [Step("repeat", "octanol_sugar_far", 45.0, 6)], same_fly=False, hunger=0.9)),
    "first_feed_s", "last", "A<B",
    "마지막 두 시행에서 같은 파리(A)가 매번 새 파리(B)보다 설탕을 빨리 먹기 시작할 것이다.",
    "옥탄올은 원래 끌림이 없어 처음엔 우연히 설탕을 밟아야 합니다. 한 번 먹고 나면 옥탄올의 가치가 올라가 다음 "
    "시행부터 냄새가 진해지는 쪽으로 돕니다. 새 파리는 매번 처음부터 우연에 맡깁니다. 식초로 했더니 타고난 끌림 "
    "때문에 첫 시행부터 3초 만에 찾아 더 빨라질 여지가 없었습니다(시범 실행).",
    "냄새와 설탕을 짝지으면 그 냄새를 따라가게 됩니다(Krashes & Waddell 2008). 같은 파리와 새 파리를 비교하면 "
    "변화가 기억 때문인지, 방 자체 때문인지 가릴 수 있습니다.",
    flies=8, also=("value_octanol", "first_touch_s"),
    outcomes=(
        ("A<B", "같은 파리인 A가 설탕을 더 빨리 먹기 시작할 것이다"),
        ("A>B", "매번 새 파리인 B가 설탕을 더 빨리 먹기 시작할 것이다"),
    ),
))

# ---- conflicts
_add(ExperimentSet(
    "hunger_vs_pain", "갈등", "배고프면 전기를 참고 먹을까?",
    "전기(30 V) 구역 가운데 설탕이 있을 때, 배고픈 초파리가 더 오래 먹을까?",
    "배고픔 0.95", "배고픔 0.3", "시작할 때 배고픔",
    (Condition("A: 배고픔 0.95", [Step("repeat", "conflict", 60.0, 1)], same_fly=False, hunger=0.95),
     Condition("B: 배고픔 0.3", [Step("repeat", "conflict", 60.0, 1)], same_fly=False, hunger=0.3)),
    "feed_s", "all", "A>B",
    "배고픈 A가 B보다 전기 구역 안의 설탕을 오래 먹을 것이다.",
    "이 모델은 '먹이가 끄는 힘(배고픔 × 맛)'이 '전기가 미는 힘'보다 크면 전기를 견딥니다.",
    "배고픈 초파리는 먹이 냄새를 쫓는 행동을 더 끈질기게 이어갑니다(Sayin 등 2019, Neuron). 이 방의 견디기 규칙은 "
    "그 결과를 본뜬 모델의 선택입니다.",
    flies=10, also=("shocks", "first_feed_s"),
    outcomes=(
        ("A>B", "배고픈 A가 전기 구역 안의 설탕을 더 오래 먹을 것이다"),
        ("A<B", "덜 배고픈 B가 전기 구역 안의 설탕을 더 오래 먹을 것이다"),
    ),
))
_add(ExperimentSet(
    "learned_danger", "갈등", "전기와 함께 맡은 식초는 배고파도 피할까?",
    "식초 냄새와 전기를 함께 겪은 초파리는 식초 향 설탕을 늦게 찾을까?",
    "식초+전기 훈련", "식초만 훈련", "훈련 방의 전기",
    (Condition("A: 식초+전기", [Step("train", "vinegar_shock", 60.0, 2), Step("test", "scented_sugar", 60.0, 2)]),
     Condition("B: 식초만", [Step("train", "vinegar_shock", 60.0, 2), Step("test", "scented_sugar", 60.0, 2)],
               volts=0.0)),
    "first_feed_s", "test", "A>B",
    "식초와 전기를 함께 겪은 A가 B보다 시험에서 식초 향 설탕을 늦게 먹기 시작할 것이다.",
    "타고난 식초 끌림(+)에 배운 가치(-)가 더해져 식초 쪽으로 도는 힘이 약해지거나 반대가 됩니다.",
    "배운 기억은 타고난 냄새 선호를 바꿀 수 있습니다. 좋아하던 냄새도 처벌과 짝지으면 피합니다(Tully & Quinn 1985 "
    "방식의 응용).",
    flies=8, also=("value_vinegar", "first_touch_s"),
    outcomes=(
        ("A>B", "식초와 전기를 함께 겪은 A가 식초 향 설탕을 더 늦게 먹기 시작할 것이다"),
        ("A<B", "식초만 겪은 B가 식초 향 설탕을 더 늦게 먹기 시작할 것이다"),
    ),
))
_add(ExperimentSet(
    "cue_needed", "갈등", "냄새 단서가 있어야 전기를 미리 피할까?",
    "전기 구역에서 냄새가 날 때와 안 날 때, 같은 방을 여러 번 겪은 초파리가 전기를 덜 밟게 되는 쪽은 어디일까?",
    "전기 구역에 MCH 냄새 있음", "냄새 없음", "전기 구역의 냄새 단서",
    (Condition("A: 냄새 있음", [Step("repeat", "cued_shocks", 60.0, 5)]),
     Condition("B: 냄새 없음", [Step("repeat", "cued_shocks", 60.0, 5)], without=["odour"])),
    "shocks", "last", "A<B",
    "마지막 두 시행에서 냄새 단서가 있는 A가 냄새가 없는 B보다 전기를 적게 밟을 것이다.",
    "전기를 맞는 순간 맡고 있던 MCH의 가치가 음수가 되어, 다음에는 냄새가 진해지는 쪽을 피해 돕니다. 냄새가 "
    "없으면 버섯체가 전기와 짝지을 것이 없어 매번 밟고 나서야 도망칩니다. 이 모델의 중심복합체는 뜨거운 바닥의 시원한 곳만 "
    "목표로 기억하므로, 전기 구역의 자리는 외우지 않습니다.",
    "초파리는 냄새나 색 같은 단서와 짝지어진 벌을 배웁니다(Tully & Quinn 1985; Vogt 등 2014). 단서 없이 장소만으로 "
    "피하는 것은 heat box 실험처럼 따로 연구되며 버섯체가 필요 없습니다(Putz & Heisenberg 2002).",
    flies=8, also=("first_shock_s", "value_mch", "shock_s"),
    outcomes=(
        ("A<B", "냄새 단서가 있는 A가 전기를 덜 밟을 것이다"),
        ("A>B", "냄새가 없는 B가 전기를 덜 밟을 것이다"),
    ),
))

# ---- mazes
_add(ExperimentSet(
    "maze_smell", "미로", "미로에서도 냄새가 길잡이가 될까?",
    "T자 미로에서 설탕에 식초 냄새가 나면 더 빨리 찾을까?",
    "식초 냄새 있음", "냄새 없음", "설탕 옆 식초 냄새가 있는가",
    (Condition("A: 식초 냄새 있음", [Step("repeat", "t_maze", 90.0, 1)], same_fly=False),
     Condition("B: 냄새 없음", [Step("repeat", "t_maze", 90.0, 1)], same_fly=False, without=["odour"])),
    "first_feed_s", "all", "A<B",
    "냄새가 있는 A가 B보다 미로 속 설탕을 빨리 먹기 시작할 것이다.",
    "냄새는 벽을 돌아 퍼지므로 갈림길에서 설탕이 있는 팔 쪽이 더 진합니다.",
    "걷는 초파리가 벽을 돌아 냄새를 따라간 실험 자료는 찾지 못했습니다. 이 결과는 모델의 예측으로 봐야 합니다.",
    flies=10, also=("first_touch_s", "distance_mm"),
    outcomes=(
        ("A<B", "냄새가 있는 A가 미로 속 설탕을 더 빨리 먹기 시작할 것이다"),
        ("A>B", "냄새가 없는 B가 미로 속 설탕을 더 빨리 먹기 시작할 것이다"),
    ),
))
_add(ExperimentSet(
    "wall_detour", "미로", "벽이 가로막으면 늦게 찾을까?",
    "설탕 앞에 벽이 있으면 벽이 없을 때보다 늦게 찾을까?",
    "벽 있음", "벽 없음", "벽이 있는가",
    (Condition("A: 벽 있음", [Step("repeat", "wall_between", 60.0, 1)], same_fly=False),
     Condition("B: 벽 없음", [Step("repeat", "wall_between", 60.0, 1)], same_fly=False, without=["obstacle"])),
    "first_feed_s", "all", "A>B",
    "벽이 있는 A가 벽이 없는 B보다 설탕을 늦게 먹기 시작할 것이다.",
    "초파리는 벽에 닿기 전에 돌아서고 벽 끝을 찾아 돌아가야 합니다.",
    "이 모델의 예측입니다. 벽 앞에서 돌아서는 반사는 몸이 벽을 타면 뒤집히기 때문에 넣은 것입니다(CLAUDE.md 측정).",
    flies=10, also=("distance_mm",),
    outcomes=(
        ("A>B", "벽이 있는 A가 설탕을 더 늦게 먹기 시작할 것이다"),
        ("A<B", "벽이 없는 B가 설탕을 더 늦게 먹기 시작할 것이다"),
    ),
))
_add(ExperimentSet(
    "door_room", "미로", "문이 하나뿐인 방 안의 설탕은 늦게 찾을까?",
    "벽으로 둘러싸이고 문이 하나뿐인 방 안의 설탕은, 벽이 없을 때보다 늦게 찾을까?",
    "문 하나 달린 방", "벽 없음", "설탕을 둘러싼 벽",
    (Condition("A: 문 하나 달린 방", [Step("repeat", "room_in_room", 90.0, 1)], same_fly=False),
     Condition("B: 벽 없음", [Step("repeat", "room_in_room", 90.0, 1)], same_fly=False, without=["obstacle"])),
    "first_feed_s", "all", "A>B",
    "벽에 둘러싸인 A가 B보다 설탕을 늦게 먹기 시작할 것이다.",
    "냄새는 문으로만 새어 나오고, 초파리도 그 문을 찾아야 들어갈 수 있습니다.",
    "이 모델의 예측입니다.",
    flies=10, also=("distance_mm",),
    outcomes=(
        ("A>B", "벽에 둘러싸인 A가 설탕을 더 늦게 먹기 시작할 것이다"),
        ("A<B", "벽이 없는 B가 설탕을 더 늦게 먹기 시작할 것이다"),
    ),
))
_add(ExperimentSet(
    "maze_practice", "미로", "같은 파리는 미로를 반복할수록 빨라질까?",
    "같은 초파리가 옥탄올 설탕이 있는 T자 미로를 여러 번 풀면, 매번 새 초파리보다 빨라질까?",
    "같은 파리", "매번 새 파리", "기억이 이어지는가",
    (Condition("A: 같은 파리", [Step("repeat", "t_maze_octanol", 60.0, 5)], hunger=0.9),
     Condition("B: 매번 새 파리", [Step("repeat", "t_maze_octanol", 60.0, 5)], same_fly=False, hunger=0.9)),
    "first_feed_s", "last", "A<B",
    "마지막 두 시행에서 같은 파리(A)가 새 파리(B)보다 빨리 설탕을 먹기 시작할 것이다.",
    "빨라진다면 설탕과 함께 맡은 옥탄올을 좋아하게 되어 갈림길에서 그쪽으로 돌기 때문일 수 있습니다. 이 모델의 중심복합체는 "
    "뜨거운 바닥의 시원한 곳만 목표로 기억하므로, 이 미로의 길 자체는 외우지 않습니다.",
    "초파리의 장소 기억은 중심복합체가 맡습니다(Ofstad 등 2011). 이 미로에서는 버섯체의 냄새 기억이 주로 쓰입니다.",
    flies=8, also=("value_octanol", "distance_mm"),
    outcomes=(
        ("A<B", "같은 파리인 A가 미로 속 설탕을 더 빨리 먹기 시작할 것이다"),
        ("A>B", "매번 새 파리인 B가 미로 속 설탕을 더 빨리 먹기 시작할 것이다"),
    ),
))

# ---- places, heat, bitterness, light
_add(ExperimentSet(
    "place_memory", "장소 기억", "뜨거운 바닥에서 시원한 칸을 점점 빨리 찾을까?",
    "방 바깥에 표지 막대가 있으면, 같은 초파리가 시원한 칸을 시행마다 더 빨리 찾게 될까?",
    "표지 막대 있음", "표지 없음", "방 바깥의 표지 막대",
    (Condition("A: 표지 막대 있음", [Step("repeat", "heat_maze", 60.0, 6)], hunger=0.5),
     Condition("B: 표지 없음", [Step("repeat", "heat_maze_plain", 60.0, 6)], hunger=0.5)),
    "first_cool_s", "last", "A<B",
    "마지막 두 시행에서 표지 막대가 있는 A가 표지가 없는 B보다 시원한 칸을 빨리 찾을 것이다.",
    "중심복합체 모형이 방향 감각(나침반), 지나온 길의 합(경로 적분), 기억한 목표를 맡습니다. 시원한 칸에 닿으면 그 자리를 "
    "목표로 기억하고, 뜨거운 바닥에 서면 그쪽으로 돕니다. 그런데 초파리는 들어 올려졌다 놓일 때마다 방향 감각을 잃습니다. "
    "표지 막대가 보이면 그 모양으로 방향을 되찾아 기억한 자리가 맞는 곳을 가리키지만, 표지가 없으면 엉뚱한 곳을 가리킵니다.",
    "뜨거운 바닥(36 °C)의 시원한 칸 하나를 둘레의 막대 무늬로 기억해, 10번 시행하는 동안 찾는 시간이 절반으로 줄었습니다. "
    "무늬를 돌리면 돌린 쪽을 찾았고, 이 기억에는 버섯체가 아니라 타원체(중심복합체)의 고리 뉴런이 필요했습니다(Ofstad 등 2011, Nature).",
    flies=8, also=("hot_s", "distance_mm"),
    outcomes=(
        ("A<B", "표지 막대가 있는 A가 시원한 칸을 더 빨리 찾을 것이다"),
        ("A>B", "표지가 없는 B가 시원한 칸을 더 빨리 찾을 것이다"),
    ),
))
_add(ExperimentSet(
    "bitter_food", "먹이 찾기", "쓴맛이 섞이면 덜 먹을까?",
    "설탕물에 쓴맛을 섞으면 초파리가 덜 먹을까?",
    "쓴맛 섞음", "쓴맛 없음", "설탕물의 쓴맛",
    (Condition("A: 쓴맛 섞음", [Step("repeat", "bitter_sugar", 60.0, 1)], same_fly=False, hunger=0.9),
     Condition("B: 쓴맛 없음", [Step("repeat", "scented_sugar", 60.0, 1)], same_fly=False, hunger=0.9)),
    "feed_s", "all", "A<B",
    "쓴맛을 섞은 A가 쓴맛이 없는 B보다 설탕을 먹은 시간이 짧을 것이다.",
    "쓴맛은 같은 방울의 단맛 세포를 누르고(단맛 × (1 - 쓴맛)), 발에 닿는 동안 처벌 도파민을 냅니다. 남은 단맛이 약하면 먹기를 "
    "시작하지 않습니다. 그래서 먹는 시간이 줄고, 함께 맡은 식초 냄새의 가치도 떨어질 수 있습니다.",
    "쓴맛은 발의 단맛 세포 반응을 억누르고(Meunier 등 2003), 쓴맛(DEET)을 섞은 설탕과 짝지은 냄새는 곧바로 피하게 되었다가 "
    "30분 뒤에는 좋아하게 됩니다(Das 등 2014). 쓴맛 처벌에는 PPL1 도파민 뉴런이 필요합니다(Kirkhart & Scott 2015).",
    flies=10, also=("first_feed_s", "value_vinegar"),
    outcomes=(
        ("A<B", "쓴맛을 섞은 A를 더 짧게 먹을 것이다"),
        ("A>B", "쓴맛이 없는 B를 더 짧게 먹을 것이다"),
    ),
))
_add(ExperimentSet(
    "light_learning", "배우기", "빛으로 도파민 뉴런을 켜도 냄새를 배울까?",
    "전기 대신 빛으로 처벌 도파민 뉴런을 켜면서 옥탄올을 맡게 하면, 시험에서 옥탄올을 피할까?",
    "처벌 빛 켬", "빛 없음", "훈련 방의 빛",
    (Condition("A: 처벌 빛 켬", _learning("light_punish_odour", "odour_choice"), hunger=0.5),
     Condition("B: 빛 없음", _learning("light_punish_odour", "odour_choice"), hunger=0.5, without=["light"])),
    "odour_pi", "test", "A<B",
    "처벌 빛을 받은 A가 B보다 시험에서 옥탄올을 더 피해 냄새 선호(옥탄올+)가 낮을 것이다.",
    "빛 구역은 감각을 거치지 않고 처벌 도파민만 냅니다. 그때 켜져 있던 옥탄올 케니언 세포의 시냅스가 약해지는 것은 전기와 "
    "똑같습니다. 전기와 달리 몸이 놀라 도망치지는 않습니다.",
    "빛에 반응하는 이온 통로(CsChrimson)를 도파민 뉴런에 넣고 냄새와 함께 빛을 비추면, 전기나 설탕 없이도 기억이 생깁니다. "
    "어느 도파민 뉴런을 켜느냐에 따라 기억이 빨리 생기거나 오래갑니다(Aso & Rubin 2016; Claridge-Chang 등 2009).",
    flies=8, also=("value_octanol", "value_mch"),
    outcomes=(
        ("A<B", "처벌 빛을 받은 A가 옥탄올을 더 피할 것이다"),
        ("A>B", "빛이 없던 B가 옥탄올을 더 피할 것이다"),
    ),
))

# ---- seeing motion
_add(ExperimentSet(
    "optomotor", "움직임", "줄무늬가 돌면 초파리도 따라 돌까?",
    "방 바깥의 줄무늬 원통이 반시계 방향으로 돌면, 초파리도 반시계 방향으로 돌까?",
    "원통이 돎 (60°/s)", "원통이 멈춤", "줄무늬 원통이 도는가",
    (Condition("A: 원통이 돎", [Step("repeat", "drum_turning", 30.0, 1)], same_fly=False),
     Condition("B: 원통이 멈춤", [Step("repeat", "drum_still", 30.0, 1)], same_fly=False)),
    "turn_ccw_dps", "all", "A>B",
    "원통이 도는 A가 멈춘 B보다 반시계 방향으로 더 빨리 돌 것이다.",
    "겹눈의 운동 감지 세포(T4·T5 모형)가 이웃한 낱눈 신호를 시간차를 두고 곱해 움직임의 방향을 알아냅니다. 시야 전체가 한쪽으로 "
    "흐르면 그 방향으로 따라 돌도록 했습니다. 초파리가 스스로 방향을 휙 바꾸는 동안에는 이 반응을 끕니다.",
    "돌아가는 줄무늬를 따라 도는 시운동 반응은 곤충 시각 연구의 가장 오래된 실험입니다(Götz & Wenking 1973). 걷는 초파리는 "
    "줄무늬가 초당 1~8번 지나갈 때 가장 세게 따라 돌고, T4·T5 세포를 막으면 반응이 사라집니다(Creamer 등 2018; Mano 등 2023).",
    flies=8, also=("distance_mm",),
    outcomes=(
        ("A>B", "원통이 도는 A가 반시계 방향으로 더 빨리 돌 것이다"),
        ("A<B", "원통이 멈춘 B가 반시계 방향으로 더 빨리 돌 것이다"),
    ),
))
_add(ExperimentSet(
    "shadow_speed", "움직임", "머리 위로 그림자가 지나가면 더 빨리 걸을까?",
    "검은 막대가 머리 위로 되풀이해 지나가면, 초파리가 더 빨리 걸을까?",
    "그림자 있음", "그림자 없음", "머리 위로 지나가는 그림자",
    (Condition("A: 그림자 있음", [Step("repeat", "shadow_bursts", 60.0, 1)], same_fly=False),
     Condition("B: 그림자 없음", [Step("repeat", "shadow_bursts", 60.0, 1)], same_fly=False, without=["shadow"])),
    "speed_mms", "all", "A>B",
    "그림자가 지나가는 A가 그림자가 없는 B보다 평균 속도가 빠를 것이다.",
    "눈 위쪽이 갑자기 어두워지면 그림자로 알아챕니다. 그때마다 각성이 1씩 쌓이고 약 20초에 걸쳐 가라앉는데, 각성만큼 걸음이 "
    "빨라집니다. 대신 지나가는 순간 얼어붙기도 합니다(천천히 걷던 파리일수록 잘 얼어붙음). 두 효과 중 무엇이 이길지가 이 질문입니다.",
    "그림자를 1초 간격으로 여러 번 보여 주면 초파리는 그 횟수만큼 더 빨리 뛰어다니고, 그 상태가 수십 초 이어집니다(Gibson 등 2015). "
    "다가오는 그림자에는 천천히 걷던 파리의 77%, 빨리 걷던 파리의 27%가 얼어붙었습니다(Zacarias 등 2018).",
    flies=10, also=("freeze_s", "still_pct"),
    outcomes=(
        ("A>B", "그림자가 지나가는 A가 더 빨리 걸을 것이다"),
        ("A<B", "그림자가 없는 B가 더 빨리 걸을 것이다"),
    ),
))

#: The page's grouping of the sets, in order.
SET_GROUPS: list[tuple[str, list[str]]] = []
for _set in EXPERIMENT_SETS.values():
    # One heading per group, in order of first appearance, wherever a set was added.
    group = next((g for g in SET_GROUPS if g[0] == _set.group), None)
    if group is None:
        group = (_set.group, [])
        SET_GROUPS.append(group)
    group[1].append(_set.key)


def protocol_for(key: str, *, flies: int | None = None, seconds: float | None = None,
                 prediction: str = "", seed: int = 1, replay_flies: int = 3,
                 layout: list[dict] | None = None) -> Protocol:
    """The set `key` as a runnable protocol. `seconds` replaces every trial's length."""
    s = EXPERIMENT_SETS[key]
    conditions = copy.deepcopy(list(s.conditions))
    if seconds is not None:
        for condition in conditions:
            for step in condition.steps:
                step.seconds = float(seconds)
    return Protocol(
        title=s.title, set_key=key, conditions=conditions,
        flies=int(flies or s.flies), seed=int(seed), layout=list(layout or []),
        replay_flies=int(replay_flies), metric=s.metric, phase=s.phase, expect=s.expect,
        prediction=prediction or s.expect, created=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


def estimate_wall_seconds(protocol: Protocol, workers: int, wall_per_sim: float) -> float:
    """Rough wall time: flies run in parallel, one condition's sequence after
    another's on each worker."""
    per_fly = [sum(st.seconds * st.repeats for st in c.steps) for c in protocol.conditions]
    jobs = protocol.flies * len(protocol.conditions)
    waves = math.ceil(jobs / max(1, workers))
    return waves * max(per_fly) * wall_per_sim


def prediction_choices(s: ExperimentSet) -> list[dict]:
    """The hypotheses the page offers for set `s`, in order, ending with "no difference"."""
    column = METRICS_BY_KEY[s.metric].column
    pairs = list(s.outcomes) or [
        (d, f"A의 '{column}' 값이 B보다 {'클' if d == 'A>B' else '작을'} 것이다") for d in ("A>B", "A<B")]
    return [{"value": d, "text": text} for d, text in pairs] + [
        {"value": "same", "text": f"A와 B가 비슷할 것이다 ('{column}' 차이가 거의 없음)"}]


def prediction_text(s: ExperimentSet, value: str) -> str:
    return next((c["text"] for c in prediction_choices(s) if c["value"] == value), "")


def room_items(room: str, condition: Condition, layout: list[dict] | None = None) -> list[dict]:
    """The items `furnish` would put in the room, as item specs, without a
    sandbox: for drawing a condition's rooms before anything has run."""
    if room == "current":
        entries = [(spec["kind"], spec["x"], spec["y"], spec.get("params", {})) for spec in layout or []]
    else:
        entries = PRESETS[room][1]
    items = []
    for kind, x, y, params in entries:
        if kind in condition.without:
            continue
        params = dict(params)
        if kind == "shock" and condition.volts is not None:
            params["volts"] = condition.volts
        if kind == "sugar":
            if condition.molar is not None:
                params["molar"] = condition.molar
            if condition.volume is not None:
                params["volume"] = condition.volume
        if kind == "odour" and condition.strength is not None:
            params["strength"] = condition.strength
        items.append({"kind": kind, "x": float(x), "y": float(y), "params": params})
    return items


def describe_steps(condition: Condition) -> str:
    parts = []
    for step in condition.steps:
        role = ROLE_LABELS.get(step.role, step.role)
        text = f"{role}: {room_label(step.room)} {duration_ko(step.seconds)}"
        text += f" × {step.repeats}번" if step.repeats > 1 else ""
        if step.gap:
            text += f" (사이 {duration_ko(step.gap)} 쉬기)"
        parts.append(text)
        if step.rest_after:
            parts.append(f"쉬기 {duration_ko(step.rest_after)}")
    return " → ".join(parts)


__all__ = [
    "EXPERIMENT_SETS", "SET_GROUPS", "Condition", "ExperimentSet", "FlyRun", "Protocol", "Step",
    "VERDICT_TEXT", "evaluate", "fly_plan", "furnish", "prediction_choices", "prediction_text", "protocol_for",
    "read_records", "room_items", "run_fly_job",
    "TRIAL_METRICS", "ODOURS",
]

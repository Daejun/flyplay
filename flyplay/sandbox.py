"""The sandbox: a sealed room, a fly with a mushroom body, and stimuli placed live.

The user drops sugar, shock zones, odour sources, coloured floor patches and
obstacle blocks into the room while the fly walks, and watches what it does.
Every stimulus is defined by three things, and nothing else:

    what the fly senses     odour at the antennae, colour through the eyes,
                            sugar and current under its leg tips
    what dopamine it drives punishment into one compartment, sugar into others
    what the body does      innate: stop and feed, escape, turn from walls

so learning is never scripted. Whatever odours and colours are present when
dopamine arrives are what the mushroom body associates, exactly as in the
conditioning tasks -- the room just lets the contingencies be anything.

**What can be learned here, and what cannot.** The mushroom body learns what
an odour or a colour predicts. A shock zone with nothing to smell or see marks
nothing: the fly escapes it every time and never avoids it from a distance,
because remembering a *place* is the central complex's job (Ofstad et al. 2011)
and that is not built yet. Put a colour patch or an odour with it and the fly
can learn to keep away.

**Odour passes through walls and obstacles** in this version: intensity is the
inverse-square field of `flyplay.odor`, as in NeuroMechFly v2. Fine in the open
room; misleading behind a block. PLAN.md B-1 replaces it.

**Memory has four compartments, with the literature's speeds** (all fitted to
nothing; the time scales are the papers', the rates are model choices):

    approach_fast   punishment, learns in one shock, fades over hours
                    (PPL1-gamma1pedc: retained at 10 min, largely gone by 24 h;
                    MB-MP1 memory decayed over 9 h -- Aso & Rubin 2016, Aso 2012)
    approach_slow   punishment, learns a tenth as fast, lasts days
                    (PPL1-alpha3: barely detectable after one pairing, 4-day
                    memory after spaced training -- Aso & Rubin 2016); written
                    only as far as the fly is fed
    avoid_sweet     sugar taste, learns fast, fades over hours, written only as
                    far as the fly is hungry (sweet-taste PAM neurons,
                    beta'2am/gamma4; arabinose short-term memory -- Huetteroth
                    2015, Yamagata 2015)
    avoid_nutrient  sugar calories, lasts days, written only when hungry
                    (gamma5b / PAM-alpha1; sucrose memory flat from 1 to 36 h --
                    Huetteroth 2015, Yamagata 2015, Krashes & Waddell 2008)

Arabinose is sweet with no calories and gives short-term memory only; sorbitol
has calories and no taste, and a fly does not start feeding on it (Burke &
Waddell 2011: flies drank no more sorbitol than water); the two mixed act like
sucrose.

**Hunger gates learning and memory through one variable.** It gates what
appetitive memory *shows* (Krashes et al. 2009: hidden after feeding, back
after re-starving), never aversive. It also gates what is *written*, both ways:

    sugar reward    x hunger       octopamine's sweet-taste reinforcement needs
                                   the dopamine neurons that carry appetitive
                                   motivation (Burke et al. 2012)
    slow punishment x (1-hunger)   starvation silences the MP1/MV1 dopamine
                                   neurons that gate aversive long-term memory,
                                   and starved flies form none (Placais & Preat
                                   2013)

MP1 is the MB-MP neuron whose activity in fed flies hides appetitive memory
(Krashes et al. 2009), so ``1 - hunger`` stands for its activity in every rule.
Not modelled: mild fasting *facilitates* aversive long-term memory (Hirano et
al. 2013); the hunger-dependent switch at MVP2 feed-forward inhibition (Perisse
et al. 2016) is collapsed into the ``hunger x appetitive`` product.

Brain, in priority order (first that applies drives the legs):

    1 proximity reflex    anything within reach of the tightest turn ahead
    2 escape              legs on a live shock zone: turn away, then run
    3 feeding             legs on a sweet drop and hungry: stop, extend the
                          proboscis once stopped, sip; fold it before leaving
    4 local search        just after feeding: loop back to the food
    5 navigation          learned + innate odour and colour steering, with
                          random exploratory turns
"""

from __future__ import annotations

import threading
from array import array
from collections import deque
from dataclasses import dataclass, field

import mujoco
import numpy as np
from flygym.anatomy import LEGS, BodySegment

from flyplay.build import build
from flyplay.control import Walker
from flyplay.env import SIGNAL_HIGH, SIGNAL_LOW
from flyplay.mushroom_body import Compartment, MushroomBody
from flyplay.odor import OdorField, OdorSource
from flyplay.olfactory import OlfactoryFrontEnd
from flyplay.room import POOL_SIZES, SUGAR_RADIUS, ZONE_HALF, ProximityReflex
from flyplay.visual_pathway import FLOOR_READINGS, VisualFrontEnd, floor_readouts

# --- stimuli -------------------------------------------------------------------

#: Odours the room can hold, as dimensions of the odour space.
ODOURS = ("vinegar", "octanol", "mch")
ODOUR_LABELS = {"vinegar": "식초", "octanol": "옥탄올", "mch": "MCH"}
#: Innate pull toward each odour (lateral horn), scaled by hunger. Vinegar is a
#: food odour; 3-octanol and 4-methylcyclohexanol are the classic conditioning
#: pair, used because flies start roughly neutral to them.
INNATE_ODOUR = {"vinegar": 0.35, "octanol": 0.0, "mch": 0.0}
ODOUR_RGBA = {
    "vinegar": (0.95, 0.75, 0.15, 1.0),
    "octanol": (0.85, 0.35, 0.85, 1.0),
    "mch": (0.2, 0.8, 0.8, 1.0),
}
#: Odour source peak intensity range. At `ODOR_MIN_DISTANCE` 3 mm a peak of 1.0
#: gives 0.111, the top of the Kenyon-cell working range.
MIN_DISTANCE = 3.0


@dataclass(frozen=True)
class Sugar:
    """How a sugar acts: through taste, and through calories after ingestion."""

    #: Taste drive at saturating concentration, 0-1. Triggers feeding and the
    #: fast reward signal.
    sweet: float
    #: Calories per unit intake, 0-1. Relieves hunger and drives the slow,
    #: long-lasting reward signal.
    nutrient: float
    rgba: tuple[float, float, float, float]


#: Sugars the palette offers. See the module docstring for the sources.
SUGARS: dict[str, Sugar] = {
    "sucrose": Sugar(sweet=1.0, nutrient=1.0, rgba=(0.98, 0.98, 0.92, 1.0)),
    "fructose": Sugar(sweet=1.0, nutrient=1.0, rgba=(1.0, 0.85, 0.55, 1.0)),
    "arabinose": Sugar(sweet=1.0, nutrient=0.0, rgba=(1.0, 0.6, 0.8, 1.0)),
    "sorbitol": Sugar(sweet=0.0, nutrient=1.0, rgba=(0.7, 0.9, 1.0, 1.0)),
    "arabinose_sorbitol": Sugar(sweet=1.0, nutrient=1.0, rgba=(0.85, 0.75, 1.0, 1.0)),
}
#: Korean names for the page's event log.
SUGAR_LABELS = {"sucrose": "자당", "fructose": "과당", "arabinose": "아라비노스",
                "sorbitol": "소르비톨", "arabinose_sorbitol": "아라비노스+소르비톨"}
#: Feeding drains a drop at this rate, nl per second. flyPAD recordings: about
#: 1.05 nl per sip, sips of 0.13-0.16 s with 0.07-0.08 s between (Itskov et al.
#: 2014) -- roughly 5 nl for each second spent sipping.
INTAKE_NL_PER_S = 5.0
#: Default drop size, nl. The local-search studies used 0.2 microlitres
#: (Shakeel & Brockmann 2023); half that keeps a meal to ~20 s on screen.
DEFAULT_DROP_NL = 100.0


#: Smallest radius a drop is tasted within, mm, however little it holds.
#: Planted tarsi sit 1.45 mm from their nearest planted neighbour (median while
#: walking; 1.14 at the 10th percentile), so a drop much narrower than that
#: fits between two feet and the two-leg onset rule could never fire on it.
#: Measured at 1.5 mm, 90 s with vinegar: a 20 nl drop is found at 2.7 s and
#: fed on with the thorax 0.5-2.1 mm from its centre; a 500 nl drop (5.6 mm)
#: from its rim, thorax ~5 mm out; 100 nl unchanged from the fixed radius.
MIN_TASTE_RADIUS = 1.5


def drop_radius(nl: float) -> float:
    """Drawn radius of a drop holding `nl`, mm, which is also the radius it
    is tasted within (never under `MIN_TASTE_RADIUS`).

    Area proportional to the amount, with 100 nl at `SUGAR_RADIUS` (2.5 mm).
    Far larger than a real drop -- 100 nl of water is a bead about 0.4 mm
    across -- so it can be seen and found; only the proportions are kept. The
    first version scaled by the fraction left of the original amount, so
    raising a drop's amount made it *smaller* (reported by the user).
    """
    return SUGAR_RADIUS * float(np.sqrt(max(nl, 0.0) / DEFAULT_DROP_NL))


#: Concentration giving half the maximal taste drive, M. No adult dose-response
#: for reward was found (the protocols use 1-3 M, saturating); a model choice.
SUGAR_HALF = 0.1


def sugar_drive(concentration: float) -> float:
    return concentration / (concentration + SUGAR_HALF)


def escape_drive(volts: float, v50: float = 45.0) -> float:
    """How hard a shock pushes the fly to leave, 0-1.

    Chadha & Cook (2014): the leg response to shock has its midpoint at 45 +/- 14
    V. Same Hill shape as `shock_drive`, centred there.
    """
    volts = max(0.0, float(volts))
    return volts**3 / (volts**3 + v50**3)


def shock_drive(volts: float, v50: float = 20.0) -> float:
    """Punishment dopamine for a shock of `volts`, saturating.

    Vogt et al. (2014): memory was significant from 15 V and flat from 30 to
    120 V. A Hill curve with half-effect at 20 V and exponent 3 gives 0.30 at
    15 V, 0.77 at 30 V and 0.96 at 60 V -- the same shape, not a fit.
    """
    volts = max(0.0, float(volts))
    return volts**3 / (volts**3 + v50**3)


def duration_ko(seconds: float) -> str:
    """A duration as the page and the report say it: 30초, 15분, 1시간 30분, 2일."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}초"
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"{minutes}분"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}시간" + (f" {minutes}분" if minutes else "")
    days, hours = divmod(hours, 24)
    return f"{days}일" + (f" {hours}시간" if hours else "")


# --- what a trial measures ---------------------------------------------------------

#: A thorax this close to an odour source counts as being near it, mm. The
#: presets set sources 30-50 mm apart, so two zones never overlap; 10 mm around
#: a source is 3% of the room's floor.
NEAR_RADIUS = 10.0
#: Slower than this counts as standing still, mm/s. Measured here: walking is
#: 13.8 mm/s, and a fly extending its proboscis stands at 0.34 mm/s (median).
STILL_SPEED = 1.0


@dataclass(frozen=True)
class Metric:
    """One per-trial measure, as the results table, the CSV and the report show it."""

    key: str
    column: str
    #: One sentence for the report's method section.
    meaning: str
    #: What the room must have held for the measure to apply, as
    #: `Sandbox.contents` names it. Empty: always applies.
    needs: tuple[str, ...] = ()
    #: The smallest difference the report words as a change rather than "about
    #: the same". A choice for wording, not a statistical test.
    tolerance: float = 0.0
    #: A time until something first happened, None when it never did.
    latency: bool = False
    #: Words rather than a number: tabled, never charted.
    text: bool = False


TRIAL_METRICS: list[Metric] = [
    Metric("first_touch_s", "처음 설탕 닿기(초)", "시행을 시작하고 발이 설탕 방울에 처음 닿기까지 걸린 시간입니다. 닿지 못하면 빈칸입니다.",
           ("sugar",), 2.0, latency=True),
    Metric("first_feed_s", "처음 먹기(초)", "주둥이를 뻗어 처음 먹기 시작하기까지 걸린 시간입니다. 먹지 못하면 빈칸입니다.",
           ("sugar",), 2.0, latency=True),
    Metric("feed_bouts", "먹은 횟수", "설탕 위에 멈춰 먹기 시작한 횟수입니다.", ("sugar",), 1.0),
    Metric("feed_s", "먹은 시간(초)", "먹고 있던 시간을 모두 더한 값입니다.", ("sugar",), 2.0),
    Metric("by_sugar", "설탕별 먹은 시간(초)", "설탕 종류마다 먹은 시간입니다.", ("sugar",), text=True),
    Metric("end_hunger", "끝 배고픔", "시행이 끝났을 때의 배고픔입니다. 0은 배부름, 1은 굶주림입니다.", (), 0.05),
    Metric("sugar_left_nl", "남은 설탕(nl)", "시행이 끝났을 때 방에 남은 설탕의 양입니다.", ("sugar",), 10.0),
    Metric("first_shock_s", "처음 전기(초)", "전기 구역을 처음 밟기까지 걸린 시간입니다. 밟지 않으면 빈칸입니다.",
           ("shock",), 2.0, latency=True),
    Metric("shocks", "전기 횟수", "전기 구역을 밟은 횟수입니다. 1초 넘게 떨어졌다가 다시 밟아야 새로 셉니다.", ("shock",), 1.0),
    Metric("shock_s", "전기 닿은 시간(초)", "발이 전기 구역에 닿아 있던 시간의 합입니다.", ("shock",), 1.0),
    Metric("blue_pct", "파랑 위(%)", "파란 바닥 위에 있던 시간의 비율입니다.", ("patch_blue",), 3.0),
    Metric("green_pct", "초록 위(%)", "초록 바닥 위에 있던 시간의 비율입니다.", ("patch_green",), 3.0),
    Metric("colour_pi", "색 선호(파랑+ 초록-)",
           "방을 파랑에 더 가까운 쪽과 초록에 더 가까운 쪽으로 나누어 (파랑 쪽 시간 - 초록 쪽 시간) ÷ "
           "(두 시간의 합)으로 셉니다. T자 미로에서 어느 팔에 있었는지 세는 것과 같습니다. +1이면 늘 파랑 쪽, "
           "-1이면 늘 초록 쪽입니다.", ("patch_blue", "patch_green"), 0.1),
    Metric("near_vinegar_pct", "식초 근처(%)", f"식초 냄새가 나오는 곳 {NEAR_RADIUS:.0f} mm 안에 있던 시간의 비율입니다.",
           ("odour_vinegar",), 3.0),
    Metric("near_octanol_pct", "옥탄올 근처(%)", f"옥탄올 냄새가 나오는 곳 {NEAR_RADIUS:.0f} mm 안에 있던 시간의 비율입니다.",
           ("odour_octanol",), 3.0),
    Metric("near_mch_pct", "MCH 근처(%)", f"MCH 냄새가 나오는 곳 {NEAR_RADIUS:.0f} mm 안에 있던 시간의 비율입니다.",
           ("odour_mch",), 3.0),
    Metric("odour_pi", "냄새 선호",
           "방을 두 냄새 중 더 가까운 쪽으로 나누어 (첫째 냄새 쪽 시간 - 둘째 냄새 쪽 시간) ÷ (두 시간의 합)으로 "
           "셉니다. 어느 냄새가 첫째(+1)인지는 '냄새 선호 기준' 칸에 있습니다.", ("odour_pair",), 0.1),
    Metric("odour_pi_pair", "냄새 선호 기준", "냄새 선호의 +1 쪽과 -1 쪽 냄새입니다.", ("odour_pair",), text=True),
    Metric("distance_mm", "이동 거리(mm)", "0.1초마다 잰 위치를 이은 길이입니다.", (), 20.0),
    Metric("speed_mms", "평균 속도(mm/s)", "이동 거리 ÷ 시행 시간입니다. 먹거나 멈춘 시간도 포함합니다.", (), 1.0),
    Metric("still_pct", "멈춰 있던 시간(%)",
           f"초속 {STILL_SPEED:.0f} mm보다 느리게 움직인 시간의 비율입니다. 먹는 시간도 포함합니다.", (), 5.0),
    Metric("value_vinegar", "식초 가치", "버섯체가 식초 냄새에 매긴 가치입니다. +는 끌림, -는 피함입니다.", (), 0.05),
    Metric("value_octanol", "옥탄올 가치", "버섯체가 옥탄올 냄새에 매긴 가치입니다.", (), 0.05),
    Metric("value_mch", "MCH 가치", "버섯체가 MCH 냄새에 매긴 가치입니다.", (), 0.05),
    Metric("value_blue", "파랑 가치", "버섯체가 파란 바닥에 매긴 가치입니다.", (), 0.05),
    Metric("value_green", "초록 가치", "버섯체가 초록 바닥에 매긴 가치입니다.", (), 0.05),
]
METRICS_BY_KEY = {m.key: m for m in TRIAL_METRICS}


# --- configuration ----------------------------------------------------------------


@dataclass
class SandboxConfig:
    action_hz: float = 100.0
    vision_hz: float = 10.0
    #: Depression rate of the fast compartments, as in the conditioning tasks;
    #: the slow punishment compartment learns a tenth as fast, the nutrient one
    #: half (one 2-min sucrose pairing already forms long-term memory --
    #: Krashes & Waddell 2008).
    eta: float = 8.0
    eta_slow_factor: float = 0.1
    eta_nutrient_factor: float = 0.5
    #: Recovery time constants, from the retention data in the module docstring.
    recovery_fast: float = 1.0 / (4 * 3600.0)
    recovery_slow: float = 1.0 / (4 * 86400.0)
    recovery_sweet: float = 1.0 / (3 * 3600.0)
    recovery_nutrient: float = 1.0 / (3 * 86400.0)
    #: Punishment dopamine lasts at least this long after each shock onset: the
    #: standard protocol's shock pulse is 1.5 s, and a fly that escapes in 0.2 s
    #: has still been shocked. Measured without it: 7 brief shocks left blue at
    #: -0.10, too faint to steer by.
    shock_pulse_seconds: float = 1.5
    #: A new shock is counted only after this long off every zone. Contact is
    #: read from planted feet, so one visit flickers on and off as legs step:
    #: counted per onset, one visit scored 4 in 0.4 s and another 5 in 0.7 s,
    #: while separate visits were at least 12 s apart (persona test).
    shock_rearm_seconds: float = 1.0
    turn_gain: float = 80.0
    visual_gain: float = 3.0
    #: Exploratory saccades: rate per second and size, from walking flies in
    #: plumes (Demir et al. 2020: ~1.3 turns/s of 30 +/- 10 degrees).
    explore_rate: float = 1.33
    explore_deg: float = 30.0
    explore_sd_deg: float = 10.0
    #: Hunger 0 (sated) to 1 (starved): rises on its own, falls with calories.
    hunger_start: float = 0.8
    hunger_rise_per_s: float = 1.0 / 1800.0
    feed_threshold: float = 0.15
    #: Hunger removed per second of feeding on a fully nutritious sugar at
    #: saturating concentration.
    satiation_per_s: float = 0.02
    #: Chance per second that a feeding bout ends, at zero hunger and at full.
    leave_rate_sated: float = 1.0
    leave_rate_hungry: float = 0.05
    #: Seconds before the fly will feed again after leaving a drop.
    feed_refractory: float = 3.0
    #: The proboscis goes out only once the fly has stopped -- feeding and
    #: walking are exclusive motor programs -- and a zero descending signal
    #: stops it in 0.2 s (measured: 9.8 mm/s at onset, 0.2 mm/s 0.2 s later).
    #: Nothing is swallowed before then.
    per_latency: float = 0.2
    #: Sip rhythm while feeding: labellum down, then lifted. flyPAD sips last
    #: 0.13-0.16 s with 0.07-0.08 s between (Itskov et al. 2014).
    sip_seconds: float = 0.145
    sip_gap_seconds: float = 0.075
    #: After a calm end to a meal the fly stands while the proboscis folds (90%
    #: in 0.23 s, measured) instead of walking off with it out. A wall or a
    #: shock ends a meal without this pause.
    retract_seconds: float = 0.3
    shock_v50: float = 20.0
    escape_v50: float = 45.0
    escape_seconds: float = 0.6
    #: Local search after a meal lasts this long, plus `search_per_hunger`
    #: seconds per unit of hunger: flies walk 30-300 cm in it, longer the
    #: hungrier (Corfas et al. 2019). Time-compressed like hunger itself.
    search_seconds: float = 30.0
    search_per_hunger: float = 90.0
    search_radius: float = 12.0


# --- runtime ----------------------------------------------------------------------

#: The longest trail kept, in 10 Hz points: a day of walking. Stored as float32
#: pairs that is 6.9 MB; past it the oldest hour is dropped.
TRAIL_MAX_POINTS = 24 * 3600 * 10
TRAIL_DROP_POINTS = 3600 * 10


class Trail:
    """The fly's path at 10 Hz since it was last put down, for the page's map.

    The page asks for it in pieces (`since`): the whole of it once, then only
    what is new. Sending everything on every poll would be 36,000 points an
    hour, 150 ms apart. `epoch` changes whenever the trail starts over -- a new
    fly, or the fly moved back to the centre -- so the page knows to drop its
    copy. Appended by the simulation thread, read by HTTP threads: the lock
    keeps `dropped` and the array in step while an hour is being dropped.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.epoch = 0
        self._xy = array("f")
        self.dropped = 0

    def clear(self) -> None:
        with self._lock:
            self._xy = array("f")
            self.dropped = 0
            self.epoch += 1

    def append(self, x: float, y: float) -> None:
        with self._lock:
            self._xy.extend((x, y))
            if len(self._xy) > 2 * TRAIL_MAX_POINTS:
                del self._xy[:2 * TRAIL_DROP_POINTS]
                self.dropped += TRAIL_DROP_POINTS

    @property
    def total(self) -> int:
        """Points appended since the trail started over, dropped ones included."""
        return self.dropped + len(self._xy) // 2

    def since(self, index: int) -> tuple[int, int, array]:
        """(epoch, first point index, float32 x/y pairs) from `index` on, or
        from the oldest point still kept."""
        with self._lock:
            start = max(int(index), self.dropped)
            return self.epoch, start, self._xy[2 * (start - self.dropped):]

    def recent(self, n: int) -> tuple[int, int, list[float]]:
        """(epoch, total, the newest `n` points as a flat [x0, y0, x1, y1, ...]
        list), read together so the page lines the points up with the count."""
        with self._lock:
            tail = [round(v, 2) for v in self._xy[-2 * n:]] if n > 0 else []
            return self.epoch, self.dropped + len(self._xy) // 2, tail

    def __len__(self) -> int:
        return len(self._xy) // 2


@dataclass
class Telemetry:
    """What the viewer shows. Rebuilt every step; plain types only."""

    time: float = 0.0
    mode: str = "explore"
    hunger: float = 0.0
    legs_on_sugar: int = 0
    legs_on_shock: int = 0
    sugar_under: str | None = None
    dopamine: dict = field(default_factory=dict)
    odour_valence: dict = field(default_factory=dict)
    colour_valence: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    reflex: list = field(default_factory=list)


class Sandbox:
    """One fly in the sealed room. `step` advances one 100 Hz action step."""

    def __init__(self, config: SandboxConfig | None = None, *, seed: int = 0):
        self.config = cfg = config or SandboxConfig()
        self.rng = np.random.default_rng(seed)
        # One odour source per room odour slot, parked until placed.
        self.odor_field = OdorField(
            sources=[OdorSource(pos=(900.0, 900.0, 1.5), peak=(0.0,) * len(ODOURS))
                     for _ in range(POOL_SIZES["odour"])],
            min_distance=MIN_DISTANCE,
        )
        # Plain body: the eyes see the fly's own legs, and coloured legs changed
        # the visual Kenyon-cell code on 57% of samples. The viewer paints its
        # own copy (`flyplay.build.display_colours`).
        self.fs = build("flat", odor_field=self.odor_field, odor_markers=False,
                        vision=True, room=True, colorize=False, proboscis=True, seed=seed)
        self.room = self.fs.room
        self.walker = Walker(self.fs)

        front = OlfactoryFrontEnd(len(ODOURS), seed=seed)
        visual = VisualFrontEnd(seed=seed)
        n_kc = front.n_kc + visual.n_kc
        self.mb = MushroomBody(front, [
            Compartment("approach_fast", +1.0, n_kc=n_kc, eta=cfg.eta,
                        recovery=cfg.recovery_fast),
            Compartment("approach_slow", +1.0, n_kc=n_kc, eta=cfg.eta * cfg.eta_slow_factor,
                        recovery=cfg.recovery_slow),
            Compartment("avoid_sweet", -1.0, n_kc=n_kc, eta=cfg.eta,
                        recovery=cfg.recovery_sweet),
            Compartment("avoid_nutrient", -1.0, n_kc=n_kc,
                        eta=cfg.eta * cfg.eta_nutrient_factor, recovery=cfg.recovery_nutrient),
        ], visual=visual)
        self.reflex = ProximityReflex()

        seg_order = self.fs.fly.get_bodysegs_order()
        self._tips = [seg_order.index(BodySegment(f"{leg}_tarsus5")) for leg in LEGS]
        self.dt = 1.0 / cfg.action_hz
        self.steps_per_action = int(round(self.dt / self.fs.sim.timestep))
        self._look_every = max(1, int(round(cfg.action_hz / cfg.vision_hz)))
        #: Whether feeding uses drops up. Off, a drop never shrinks: the room's
        #: layout stays exactly as placed however long the fly eats. A room
        #: setting, so it survives `reset_fly`.
        self.sugar_depletes = True
        # Fixed codes for reading each stimulus's learned value without
        # disturbing anything: an odour alone, a floor of one colour.
        self._odour_kc = [self.mb.embed(front.encode(np.eye(len(ODOURS))[i] * 0.05))
                          for i in range(len(ODOURS))]
        self._colour_readouts = {c: floor_readouts(FLOOR_READINGS[c]) for c in ("blue", "green")}
        # The fly's free joint, for turning a new fly to a chosen heading.
        model = self.fs.sim.mj_model
        free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        self._root_qpos = int(model.jnt_qposadr[free[0]])
        self._root_dof = int(model.jnt_dofadr[free[0]])
        self.reset_fly()

    # --- lifecycle -------------------------------------------------------------

    def reset_fly(self, yaw: float | None = None) -> None:
        """A new fly in the same room: memory wiped, hunger reset, back to the
        centre -- facing +x, or `yaw` radians from it."""
        self.mb.reset()
        self._clear_actions()
        self._settle_body(yaw)
        self.time = 0.0
        self.hunger = self.config.hunger_start
        self._i = 0
        self._last_fed_at = -1e9
        self._last_shock_at = -1e9
        self._shock_volts = 0.0
        self.reset_counts()
        self.events: deque[dict] = deque(maxlen=40)
        if not hasattr(self, "trail"):
            self.trail = Trail()
        self.trail.clear()
        self.telemetry = Telemetry()
        heading = "" if yaw is None else f" (시작 방향 {np.degrees(yaw) % 360:.0f}°)"
        self._event("새 파리를 방 가운데에 놓았습니다" + heading)
        self.refresh_telemetry()

    def return_home(self, yaw: float | None = None) -> None:
        """The same fly back at the centre, facing +x or `yaw`: memory, hunger
        and the running tallies are kept, and the simulated clock runs on."""
        clock = self.fs.sim.mj_data.time
        self._clear_actions()
        self._settle_body(yaw)
        # Settling steps physics from a reset keyframe, which also rewinds
        # sim.time; the viewer paces itself on that clock.
        self.fs.sim.mj_data.time = clock
        # The jump is not walking, and a trail line across the room is not a path.
        self._last_xy = None
        self.trail.clear()
        self._event("초파리를 방 가운데로 옮겼습니다 (기억과 배고픔은 그대로)")
        self.refresh_telemetry()

    def move_fly(self, x: float, y: float, yaw: float | None = None) -> None:
        """The same fly put down at (x, y), facing `yaw` radians from +x or, by
        default, the way it faces now: `return_home` anywhere in the room.
        Memory, hunger and tallies are kept and the clock runs on. Refused on a
        wall or obstacle, or within 3 mm of one: the body cannot stand there."""
        margin = 3.0
        limit = self.room.half - margin
        x, y = float(np.clip(x, -limit, limit)), float(np.clip(y, -limit, limit))
        if any(rect.contains(x, y, margin=margin) for rect in self.room.solids()):
            raise ValueError("벽이나 장애물 위에는 초파리를 놓을 수 없습니다")
        yaw = self.fs.yaw() if yaw is None else float(yaw)
        clock = self.fs.sim.mj_data.time
        self._clear_actions()
        self._settle_body(yaw, (x, y))
        self.fs.sim.mj_data.time = clock
        self._last_xy = None
        self.trail.clear()
        self._event(f"초파리를 ({x:.0f}, {y:.0f})로 옮겼습니다 (기억과 배고픔은 그대로)")
        self.refresh_telemetry()

    def _settle_body(self, yaw: float | None, xy: tuple[float, float] = (0.0, 0.0)) -> None:
        """Reset the body to (x, y), by default the centre, and let it settle:
        0.2 s of physics."""
        self.walker.reset(seed=int(self.rng.integers(2**31 - 1)), warmup_s=0.0)
        if yaw is not None or any(xy):
            model, data = self.fs.sim.mj_model, self.fs.sim.mj_data
            if yaw is not None:
                # Turn the whole body about the vertical before it settles: legs
                # keep their places relative to the thorax, so it lands as it
                # would have.
                spin = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
                quat = data.qpos[self._root_qpos + 3:self._root_qpos + 7].copy()
                mujoco.mju_mulQuat(data.qpos[self._root_qpos + 3:self._root_qpos + 7], spin, quat)
            # The reset keyframe stands the fly at the origin; the free joint
            # carries the whole body.
            data.qpos[self._root_qpos:self._root_qpos + 2] += xy
            data.qvel[self._root_dof:self._root_dof + 6] = 0.0
            mujoco.mj_forward(model, data)
        self.fs.sim.warmup(0.2)

    def _clear_actions(self) -> None:
        """Drop whatever the fly was in the middle of doing. Memory, hunger and
        tallies are not actions and stay. A punishment pulse still running is
        cut: carried to the new spot it would pair that spot with the shock."""
        self.reflex.reset()
        self._readouts = None
        self._saccade_left = 0
        self._saccade_sign = 0.0
        self._escape_left = 0
        self._feeding_on: int | None = None
        self._feed_reach = 0.0
        self._feed_started = 0.0
        self._retract_until = -1e9
        # Local search loops back to the last meal by dead reckoning, which a
        # jump to the centre invalidates.
        self._food_xy: np.ndarray | None = None
        self._search_until = -1e9
        self._was_shocked = False
        self._shock_pulse_left = 0
        self._enduring = False

    def reset_counts(self) -> None:
        """Start the tallies a trial reports from zero, keeping the fly."""
        self.counts = {"shocks": 0, "shock_seconds": 0.0, "feed_bouts": 0,
                       "feed_seconds": 0.0, "by_sugar": {}, "first_feed_s": None,
                       "first_touch_s": None, "first_shock_s": None,
                       "on_blue_s": 0.0, "on_green_s": 0.0,
                       "near_s": {o: 0.0 for o in ODOURS}, "still_s": 0.0, "distance_mm": 0.0,
                       "side_s": {"blue": 0.0, "green": 0.0, **{o: 0.0 for o in ODOURS}}}
        self._counts_since = self.time
        self._last_xy = None
        # A trial's first shock counts even if the last trial's was under a
        # second ago.
        self._last_shock_at = -1e9
        #: What the room has held since the counts began (`contents`): a drop
        #: eaten up mid-trial still makes the feeding measures apply.
        self.seen = self.contents()
        #: The path since the counts began, 5 Hz: (seconds, x, y).
        self.path: list[tuple[float, float, float]] = []

    def rest(self, seconds: float) -> None:
        """Time passes with the fly out of the room, as between the trials of a
        protocol: its memory recovers exactly as it would have
        (`MushroomBody.idle`) and hunger rises at its usual rate. Nothing is
        simulated, so a day costs nothing. Whatever the fly was doing ends."""
        seconds = max(0.0, float(seconds))
        if seconds <= 0.0:
            return
        self.mb.idle(seconds)
        self.hunger = float(min(1.0, self.hunger + self.config.hunger_rise_per_s * seconds))
        self.time += seconds
        self._clear_actions()
        self._event(f"{duration_ko(seconds)} 쉬었습니다 (기억은 그만큼 흐려지고 배고픔은 오릅니다)")
        self.refresh_telemetry()

    def contents(self) -> set[str]:
        """What is in the room, by the names `Metric.needs` uses: item kinds,
        plus ``patch_<colour>`` and ``odour_<name>``."""
        names: set[str] = set()
        for item in self.room.items.values():
            names.add(item.kind)
            if item.kind == "patch":
                names.add("patch_" + item.params.get("colour", "blue"))
            elif item.kind == "odour":
                names.add("odour_" + item.params.get("odour", "vinegar"))
        return names

    def seen_contents(self) -> list[str]:
        """Everything the room held since the counts began, plus ``odour_pair``
        when the trial has an odour preference to report."""
        return sorted(self.seen | ({"odour_pair"} if self.odour_pair() else set()))

    def close(self) -> None:
        self.fs.close()

    def _event(self, text: str) -> None:
        self.events.append({"t": round(self.time, 1), "text": text})

    # --- editing (call between steps, from the thread that steps) ----------------

    def place(self, kind: str, x: float, y: float, item_id: int | None = None, **params) -> dict:
        params = self._validated(kind, params)
        item = self.room.place(kind, x, y, item_id=item_id, **params)
        self._sync_item(item)
        return item.as_dict()

    def move(self, item_id: int, x: float, y: float) -> dict:
        item = self.room.move(item_id, x, y)
        self._sync_item(item)
        return item.as_dict()

    def update(self, item_id: int, **params) -> dict:
        kind = self.room.items[item_id].kind
        merged = {**self.room.items[item_id].params, **params}
        # Setting an amount refills the drop -- unless what is left is given
        # too, as when an undo puts back a drop that had been eaten from.
        if kind == "sugar" and "volume" in params and "left" not in params:
            merged["left"] = float(params["volume"])
        item = self.room.update(item_id, **self._validated(kind, merged))
        self._sync_item(item)
        return item.as_dict()

    def remove(self, item_id: int) -> None:
        item = self.room.items[item_id]
        if item.kind == "odour":
            self.odor_field.sources[item.slot].pos = (900.0, 900.0, 1.5)
        if self._feeding_on == item_id:
            self._feeding_on = None
        self.room.remove(item_id)

    def clear(self) -> None:
        for item_id in list(self.room.items):
            self.remove(item_id)

    def item_spec(self, item_id: int) -> dict:
        """Everything needed to put this item back exactly as it is now: kind,
        position, id and settings, including what is left of a drop. Derived
        values (drawn radius, colour) are recomputed when it is placed."""
        item = self.room.items[item_id]
        return {"kind": item.kind, "x": item.x, "y": item.y, "item_id": item.id,
                "params": {k: v for k, v in item.params.items() if k not in ("radius", "rgba")}}

    def _validated(self, kind: str, params: dict) -> dict:
        p = dict(params)
        if kind == "sugar":
            p["sugar"] = p.get("sugar", "sucrose")
            if p["sugar"] not in SUGARS:
                raise ValueError(f"unknown sugar {p['sugar']!r}")
            p["molar"] = float(np.clip(p.get("molar", 1.0), 0.01, 2.0))
            p["volume"] = float(np.clip(p.get("volume", DEFAULT_DROP_NL), 5.0, 500.0))
            # What is left survives edits of concentration or type; setting an
            # amount (`update` with volume) refills the drop to it.
            p["left"] = float(np.clip(p.get("left", p["volume"]), 0.0, p["volume"]))
            p["radius"] = drop_radius(p["left"])
            p["rgba"] = SUGARS[p["sugar"]].rgba
        elif kind == "shock":
            p["volts"] = float(np.clip(p.get("volts", 60.0), 0.0, 120.0))
            p["half"] = float(np.clip(p.get("half", ZONE_HALF), 3.0, 30.0))
        elif kind == "patch":
            p["colour"] = p.get("colour", "blue")
            if p["colour"] not in ("blue", "green"):
                raise ValueError(f"unknown colour {p['colour']!r}")
            p["half"] = float(np.clip(p.get("half", ZONE_HALF), 3.0, 30.0))
        elif kind == "obstacle":
            # A block by default; a wall is the same box drawn long and thin.
            p["hx"] = float(p.get("hx", p.get("half", 4.0)))
            p["hy"] = float(p.get("hy", p.get("half", 4.0)))
            p.pop("half", None)
        elif kind == "odour":
            p["odour"] = p.get("odour", "vinegar")
            if p["odour"] not in ODOURS:
                raise ValueError(f"unknown odour {p['odour']!r}")
            p["strength"] = float(np.clip(p.get("strength", 1.0), 0.05, 1.0))
            p["rgba"] = ODOUR_RGBA[p["odour"]]
        return p

    def _sync_item(self, item) -> None:
        if item.kind == "odour":
            source = self.odor_field.sources[item.slot]
            peak = [0.0] * len(ODOURS)
            peak[ODOURS.index(item.params["odour"])] = item.params["strength"]
            source.pos = (item.x, item.y, 1.5)
            source.peak = tuple(peak)

    # --- one step -----------------------------------------------------------------

    def step(self) -> None:
        cfg = self.config
        if self._i % self._look_every == 0:
            self._readouts = self.fs.sim.get_ommatidia_readouts(self.fs.name)
        self._i += 1

        intensities = self.odor_field.read(self.fs.sim)
        concentration = intensities.mean(axis=0)
        xy = self.fs.thorax_pos()[:2]
        yaw = self.fs.yaw()
        tips = self.fs.sim.get_body_positions(self.fs.name)[self._tips][:, :2]
        down = self.fs.leg_contacts()

        # Which drops and zones the legs are on. A leg counts only while it is
        # planted: a foot passing over a drop in swing tastes nothing. A drop is
        # tasted over the size it is drawn at. Measured with a fixed 2.5 mm
        # instead: 20, 100 and 500 nl drops gave step-for-step identical runs --
        # the size on screen changed nothing the fly did.
        sugar_legs: dict[int, int] = {}
        for item in self.room.of_kind("sugar"):
            reach = self._taste_radius(item)
            near = np.linalg.norm(tips - (item.x, item.y), axis=1) <= reach
            if (near & down).any():
                sugar_legs[item.id] = int((near & down).sum())
        shocks = [item for item in self.room.of_kind("shock")
                  if item.params["volts"] > 0.0
                  and any(item.rect.contains(tx, ty) for (tx, ty), d in zip(tips, down) if d)]
        if sugar_legs and self.counts["first_touch_s"] is None:
            self.counts["first_touch_s"] = self.time - self._counts_since

        # --- dopamine ---
        if shocks:
            volts = max(i.params["volts"] for i in shocks)
            if not self._was_shocked or volts > self._shock_volts:
                self._shock_volts = volts
            self._shock_pulse_left = int(cfg.shock_pulse_seconds * cfg.action_hz)
        elif self._shock_pulse_left > 0:
            self._shock_pulse_left -= 1
        punish = shock_drive(self._shock_volts, cfg.shock_v50) if self._shock_pulse_left > 0 else 0.0
        sweet = nutrient = 0.0
        feeding_item = self.room.items.get(self._feeding_on) if self._feeding_on else None
        if feeding_item is not None:
            sugar = SUGARS[feeding_item.params["sugar"]]
            drive = sugar_drive(feeding_item.params["molar"])
            sweet = sugar.sweet * drive
            nutrient = sugar.nutrient * drive
        # Hunger gates what is written (module docstring): sugar reward only as
        # far as the fly is hungry, long-term punishment only as far as it is fed.
        dan = {"approach_fast": punish, "approach_slow": punish * (1.0 - self.hunger),
               "avoid_sweet": sweet * self.hunger, "avoid_nutrient": nutrient * self.hunger}
        state = self.mb.step(concentration, dan, self.dt, readouts=self._readouts)

        # --- hunger ---
        self.hunger = float(np.clip(self.hunger + cfg.hunger_rise_per_s * self.dt
                                    - cfg.satiation_per_s * nutrient * self.dt, 0.0, 1.0))

        # --- behaviour ---
        mode, turn, mid = self._behave(xy, yaw, intensities, state, sugar_legs, shocks)
        # Proboscis extension while feeding: tarsal sugar contact in a hungry fly
        # is the classic trigger (the PER assay), and a fed or tasteless-sugar
        # fly never reaches this mode. Out once stopped, then sipping.
        sipping = mode == "feed" and self.time - self._feed_started >= cfg.per_latency
        lift = 0.0
        if sipping:
            phase = (self.time - self._feed_started - cfg.per_latency) % (
                cfg.sip_seconds + cfg.sip_gap_seconds)
            lift = 1.0 if phase >= cfg.sip_seconds else 0.0
        self.fs.set_proboscis(1.0 if sipping else 0.0, lift)
        half = min(mid - SIGNAL_LOW, SIGNAL_HIGH - mid) if mid > 0 else 0.0
        self.walker.descending_signal = np.array([mid - half * turn, mid + half * turn])
        self.walker.advance(self.steps_per_action)
        self.time += self.dt

        # --- bookkeeping ---
        if shocks:
            self.counts["shock_seconds"] += self.dt
            if self.time - self._last_shock_at > cfg.shock_rearm_seconds:
                self.counts["shocks"] += 1
                if self.counts["first_shock_s"] is None:
                    self.counts["first_shock_s"] = self.time - self.dt - self._counts_since
                self._event(f"전기 {self._shock_volts:.0f} V")
            self._last_shock_at = self.time
        self._was_shocked = bool(shocks)
        for patch in self.room.of_kind("patch"):
            if patch.rect.contains(float(xy[0]), float(xy[1])):
                key = "on_blue_s" if patch.params.get("colour") == "blue" else "on_green_s"
                self.counts[key] += self.dt
        if feeding_item is not None and mode == "feed":
            self.counts["feed_seconds"] += self.dt
            name = feeding_item.params["sugar"]
            self.counts["by_sugar"][name] = self.counts["by_sugar"].get(name, 0.0) + self.dt
            if sipping:
                self._drink(feeding_item)
        if self._i % 10 == 0:
            x, y = float(xy[0]), float(xy[1])
            self.trail.append(x, y)
            sample = 10 * self.dt
            # Path length from 10 Hz samples: at 100 Hz the gait's side-to-side
            # sway of the thorax would count as distance.
            if self._last_xy is not None:
                moved = float(np.linalg.norm(xy - self._last_xy))
                self.counts["distance_mm"] += moved
                if moved < STILL_SPEED * sample:
                    self.counts["still_s"] += sample
            self._last_xy = xy.copy()
            for odour in {item.params["odour"] for item in self.room.of_kind("odour")
                          if np.hypot(item.x - x, item.y - y) <= NEAR_RADIUS}:
                self.counts["near_s"][odour] += sample
            self._count_sides(x, y, sample)
            self.seen |= self.contents()
            if self._i % 20 == 0:
                self.path.append((round(self.time - self._counts_since, 1), round(x, 1), round(y, 1)))
        # Shown as written, after the hunger gates.
        self._telemetry(mode, state, {"punish": punish, "sweet": dan["avoid_sweet"],
                                      "nutrient": dan["avoid_nutrient"]},
                        sugar_legs, shocks, feeding_item)

    # --- behaviour ----------------------------------------------------------------

    def _behave(self, xy, yaw, intensities, state, sugar_legs, shocks):
        """(mode, turn, mid) for this step, highest priority first."""
        cfg = self.config
        reflex = self.reflex(xy, yaw, self.room.solids())
        self._reflex_distances = reflex.distances

        # 1. Walls. Nothing overrides this: contact flips the body.
        if reflex.active:
            self._stop_feeding("벽에 막혀 먹기를 멈췄습니다", hold=False)
            return "wall", reflex.turn, 1.0

        # 2. Shock: turn away from the zone's centre while on it, then run --
        # unless food pulls harder than this voltage pushes: a meal under the
        # feet, or the smell of food the fly wants. The comparison is a model
        # choice; that hungry flies persist toward food is not (Sayin et al.
        # 2019). Measured before it covered the approach: a hungry fly never
        # reached sugar inside a 30 V zone, fleeing at the zone's edge.
        if shocks:
            if self._food_pull(intensities, sugar_legs) > escape_drive(self._shock_volts, cfg.escape_v50):
                if not self._enduring:
                    self._event(f"배고파서 전기 {self._shock_volts:.0f} V를 견딥니다")
                self._enduring = True
                shocks = []
            else:
                self._enduring = False
        else:
            self._enduring = False
        if shocks:
            self._stop_feeding("전기에 놀라 먹기를 멈췄습니다", hold=False)
            zone = shocks[0]
            bearing = np.arctan2(zone.y - xy[1], zone.x - xy[0]) - yaw
            away = -np.sign(np.sin(bearing)) or 1.0
            self._escape_left = int(cfg.escape_seconds * cfg.action_hz)
            return "escape", float(away), 1.0
        if self._escape_left > 0:
            self._escape_left -= 1
            return "escape", 0.0, 1.4

        # 3. Feeding: stand on a sweet drop while hungry.
        if self._feeding_on is not None:
            if self._feeding_on not in sugar_legs:
                self._stop_feeding("설탕에서 벗어났습니다")
            else:
                leave = cfg.leave_rate_hungry + (cfg.leave_rate_sated - cfg.leave_rate_hungry) * (
                    1.0 - self.hunger)
                if self.rng.random() < leave * self.dt:
                    self._stop_feeding("배가 불러 먹기를 멈췄습니다" if self.hunger < 0.4
                                       else "먹기를 멈췄습니다")
                else:
                    return "feed", 0.0, 0.0
        elif sugar_legs and self.hunger > cfg.feed_threshold \
                and self.time - self._last_fed_at > cfg.feed_refractory:
            item_id = max(sugar_legs, key=sugar_legs.get)
            item = self.room.items[item_id]
            if sugar_legs[item_id] >= 2 and SUGARS[item.params["sugar"]].sweet > 0.0:
                self._feeding_on = item_id
                self._feed_started = self.time
                self._feed_reach = self._taste_radius(item)
                self._food_xy = np.array([item.x, item.y])
                self.counts["feed_bouts"] += 1
                if self.counts["first_feed_s"] is None:
                    self.counts["first_feed_s"] = self.time - self._counts_since
                self._event(f"{SUGAR_LABELS[item.params['sugar']]} {item.params['molar']:.2f} M: 주둥이를 뻗어 먹기 시작")
                return "feed", 0.0, 0.0

        # A meal just ended calmly: stand while the proboscis folds.
        if self.time < self._retract_until:
            return "retract", 0.0, 0.0

        # 4. Local search around the last meal.
        searching = self._food_xy is not None and self.time < self._search_until

        # 5. Navigation: learned and innate steering plus exploratory saccades.
        turn = self._odour_turn(intensities) + self._visual_turn(state)
        if searching and np.linalg.norm(xy - self._food_xy) > cfg.search_radius:
            bearing = np.arctan2(self._food_xy[1] - xy[1], self._food_xy[0] - xy[0]) - yaw
            turn += float(np.sin(bearing))
        rate = cfg.explore_rate * (3.0 if searching else 1.0)
        if self._saccade_left > 0:
            self._saccade_left -= 1
            turn = self._saccade_sign
        elif self.rng.random() < rate * self.dt:
            angle = abs(self.rng.normal(cfg.explore_deg, cfg.explore_sd_deg))
            # 228 deg/s at full turn, measured.
            self._saccade_left = int(angle / 228.0 * cfg.action_hz)
            self._saccade_sign = float(self.rng.choice((-1.0, 1.0)))
        return ("search" if searching else "explore"), float(np.clip(turn, -1.0, 1.0)), 1.0

    def _taste_radius(self, item) -> float:
        """How close a planted foot must be to a drop's centre to taste it, mm.

        The drop as drawn, but never under `MIN_TASTE_RADIUS`. The drop being
        fed on keeps the radius it had when the bout began: the fly stands
        still while the drop shrinks under it, and a shrinking rim would
        otherwise end every meal early. Moving the drop away still ends it.
        """
        reach = max(item.half, MIN_TASTE_RADIUS)
        if item.id == self._feeding_on:
            reach = max(reach, self._feed_reach)
        return reach

    def _food_pull(self, intensities, sugar_legs) -> float:
        """How strongly food holds the fly against a shock, on the scale of
        `escape_drive`: hunger times the meal's taste, or the positive part of
        what the smells present are worth (innate pull plus learned value).
        Learned aversion counts against it, so a fly that keeps getting shocked
        on its way to food stops enduring."""
        pull = 0.0
        if self._feeding_on is not None and self._feeding_on in sugar_legs:
            item = self.room.items[self._feeding_on]
            pull = self.hunger * SUGARS[item.params["sugar"]].sweet * sugar_drive(item.params["molar"])
        level = intensities.mean(axis=0)
        total = float(level.sum())
        if total > 1e-6:
            drive = np.array([INNATE_ODOUR[o] for o in ODOURS]) * self.hunger + self.odour_valences()
            pull = max(pull, float(np.sum(np.maximum(drive, 0.0) * level / total)))
        return pull

    def _drink(self, item) -> None:
        """Take this step's sip out of the drop; an empty drop is gone."""
        if not self.sugar_depletes:
            return
        params = item.params
        params["left"] = max(0.0, params["left"] - INTAKE_NL_PER_S * self.dt)
        if params["left"] <= 0.0:
            self._stop_feeding(f"{SUGAR_LABELS[params['sugar']]} 방울을 다 먹었습니다")
            self.room.remove(item.id)
            return
        if self._i % 10 == 0:
            self.room.update(item.id, radius=drop_radius(params["left"]))

    def _stop_feeding(self, text: str, hold: bool = True) -> None:
        if self._feeding_on is None:
            return
        self._feeding_on = None
        self._last_fed_at = self.time
        if hold:
            self._retract_until = self.time + self.config.retract_seconds
        self._search_until = self.time + self.config.search_seconds \
            + self.config.search_per_hunger * self.hunger
        self._event(text)

    # --- valences ---------------------------------------------------------------------

    def memory_terms(self, kc: np.ndarray, mass: float) -> tuple[float, float]:
        """(appetitive, aversive) memory for a Kenyon-cell vector of total `mass`.

        Each is how far that population's synapses sit below rest in the
        compartments of that sign: 0 for anything never paired with dopamine.
        Valence is then ``hunger * appetitive - aversive`` -- appetitive memory
        is expressed only as hunger allows (Krashes et al. 2009), aversive
        memory always.
        """
        w0 = self.mb.compartments[0].w0
        rest = w0 * mass
        below = {c.name: rest - float(c.weights @ kc) for c in self.mb.compartments}
        aversive = below["approach_fast"] + below["approach_slow"]
        appetitive = below["avoid_sweet"] + below["avoid_nutrient"]
        return appetitive, aversive

    def valence(self, kc: np.ndarray, mass: float) -> float:
        appetitive, aversive = self.memory_terms(kc, mass)
        return self.hunger * appetitive - aversive

    def odour_valences(self) -> np.ndarray:
        return np.array([self.valence(kc, 1.0) for kc in self._odour_kc])

    def colour_valence(self, colour: str) -> float:
        visual = self.mb.visual(self._colour_readouts[colour])
        kc = self.mb.embed(visual_kc=visual.mean_kc)
        return self.valence(kc, self.mb.visual_scale) / self.mb.visual_scale

    def _odour_turn(self, intensities) -> float:
        level = intensities.mean(axis=0)
        total = float(level.sum())
        if total <= 1e-6:
            return 0.0
        innate = np.array([INNATE_ODOUR[o] for o in ODOURS]) * self.hunger
        drive = innate + self.odour_valences()
        asymmetry = OdorField.asymmetry(intensities)
        return float(np.clip(self.config.turn_gain * np.sum(drive * asymmetry * level / total),
                             -1.0, 1.0))

    def _visual_turn(self, state) -> float:
        if state.visual is None:
            return 0.0
        scale = self.mb.visual_scale
        eyes = [self.valence(self.mb.embed(visual_kc=state.visual.kc[e]), scale) / scale
                for e in (0, 1)]
        return self.config.visual_gain * (eyes[0] - eyes[1])

    # --- telemetry --------------------------------------------------------------------

    def _telemetry(self, mode, state, dan, sugar_legs, shocks, feeding_item) -> None:
        t = self.telemetry
        t.time = round(self.time, 2)
        t.mode = mode
        t.hunger = round(self.hunger, 3)
        t.legs_on_sugar = max(sugar_legs.values(), default=0)
        t.legs_on_shock = len(shocks)
        t.sugar_under = feeding_item.params["sugar"] if feeding_item is not None else None
        t.dopamine = {k: round(float(v), 3) for k, v in dan.items()}
        if self._i % 10 == 0:
            t.odour_valence = {o: round(float(v), 3) for o, v in zip(ODOURS, self.odour_valences())}
            t.colour_valence = {c: round(self.colour_valence(c), 3) for c in ("blue", "green")}
        t.counts = self._rounded_counts()
        t.reflex = [round(float(d), 2) for d in getattr(self, "_reflex_distances", [])]

    def _rounded_counts(self) -> dict:
        out = {k: (round(v, 1) if isinstance(v, float) else v)
               for k, v in self.counts.items() if k not in ("by_sugar", "near_s", "side_s")}
        out["by_sugar"] = {k: round(v, 1) for k, v in self.counts["by_sugar"].items()}
        out["near_s"] = {k: round(v, 1) for k, v in self.counts["near_s"].items()}
        out["side_s"] = {k: round(v, 1) for k, v in self.counts["side_s"].items()}
        return out

    def refresh_telemetry(self) -> None:
        """Bring the readout up to date without stepping: after a new fly, a
        hunger change, or any edit while paused. Without it a new fly made while
        paused showed hunger 0.00 until play resumed (persona test)."""
        t = self.telemetry
        t.time = round(self.time, 2)
        t.hunger = round(self.hunger, 3)
        t.odour_valence = {o: round(float(v), 3) for o, v in zip(ODOURS, self.odour_valences())}
        t.colour_valence = {c: round(self.colour_valence(c), 3) for c in ("blue", "green")}
        t.counts = self._rounded_counts()

    def trial_metrics(self) -> dict:
        """What this trial measured since `reset_counts`, by `TRIAL_METRICS` key.

        None where a measure does not apply -- nothing of that kind was in the
        room -- or where a latency never ended, and for a preference index when
        the fly went near neither side. `seen` says which of the two it is.
        """
        c, seen = self.counts, self.seen
        span = max(self.time - self._counts_since, 1e-9)

        def pct(seconds: float) -> float:
            return round(100.0 * seconds / span, 1)

        def index(a: float, b: float) -> float | None:
            return round((a - b) / (a + b), 3) if a + b > 0.0 else None

        def when(key: str, needs: str) -> float | None:
            return round(c[key], 2) if needs in seen and c[key] is not None else None

        sugar, shock = "sugar" in seen, "shock" in seen
        blue, green = "patch_blue" in seen, "patch_green" in seen
        present = [o for o in ODOURS if f"odour_{o}" in seen]
        pair = self.odour_pair()
        values = dict(zip(ODOURS, self.odour_valences()))
        return {
            "first_touch_s": when("first_touch_s", "sugar"),
            "first_feed_s": when("first_feed_s", "sugar"),
            "feed_bouts": c["feed_bouts"] if sugar else None,
            "feed_s": round(c["feed_seconds"], 1) if sugar else None,
            "by_sugar": " / ".join(f"{SUGAR_LABELS[k]} {v:.1f}" for k, v in c["by_sugar"].items())
                        if sugar else None,
            "end_hunger": round(self.hunger, 3),
            "sugar_left_nl": round(sum(i.params["left"] for i in self.room.of_kind("sugar")))
                             if sugar else None,
            "first_shock_s": when("first_shock_s", "shock"),
            "shocks": c["shocks"] if shock else None,
            "shock_s": round(c["shock_seconds"], 1) if shock else None,
            "blue_pct": pct(c["on_blue_s"]) if blue else None,
            "green_pct": pct(c["on_green_s"]) if green else None,
            "colour_pi": index(c["side_s"]["blue"], c["side_s"]["green"]) if blue and green else None,
            **{f"near_{o}_pct": (pct(c["near_s"][o]) if o in present else None) for o in ODOURS},
            "odour_pi": index(c["side_s"][pair[0]], c["side_s"][pair[1]]) if pair else None,
            "odour_pi_pair": (f"{ODOUR_LABELS[pair[0]]}+ / {ODOUR_LABELS[pair[1]]}-" if pair else None),
            "distance_mm": round(c["distance_mm"], 1),
            "speed_mms": round(c["distance_mm"] / span, 2),
            "still_pct": pct(c["still_s"]),
            **{f"value_{o}": round(float(values[o]), 3) for o in ODOURS},
            "value_blue": round(self.colour_valence("blue"), 3),
            "value_green": round(self.colour_valence("green"), 3),
        }

    def _count_sides(self, x: float, y: float, seconds: float) -> None:
        """Credit a sample to the nearer side of each two-way choice in the room:
        blue or green floor, and the two odours `odour_pair` would compare. As
        in a T-maze, where the score is the arm the fly is in rather than whether
        it reached the end, a fly anywhere in the room is on one side or the
        other -- so a trial that never came near either stimulus still counts."""
        rects: dict[str, list] = {"blue": [], "green": []}
        for item in self.room.of_kind("patch"):
            rects[item.params.get("colour", "blue")].append(item.rect)
        if rects["blue"] and rects["green"]:
            far = {colour: min(float(np.hypot(max(abs(x - r.cx) - r.hx, 0.0), max(abs(y - r.cy) - r.hy, 0.0)))
                               for r in found) for colour, found in rects.items()}
            if far["blue"] != far["green"]:
                self.counts["side_s"]["blue" if far["blue"] < far["green"] else "green"] += seconds
        nearest: dict[str, float] = {}
        for item in self.room.of_kind("odour"):
            odour = item.params["odour"]
            nearest[odour] = min(nearest.get(odour, np.inf), float(np.hypot(item.x - x, item.y - y)))
        present = [o for o in ODOURS if o in nearest]
        pair = ("octanol", "mch") if {"octanol", "mch"} <= set(present) else present if len(present) == 2 else None
        if pair and nearest[pair[0]] != nearest[pair[1]]:
            self.counts["side_s"][pair[0] if nearest[pair[0]] < nearest[pair[1]] else pair[1]] += seconds

    def odour_pair(self) -> tuple[str, str] | None:
        """The two odours the trial's odour preference compares, first counted
        +1: octanol against MCH whenever both were in the room (the classic
        pair), otherwise the two present, in `ODOURS` order."""
        present = [o for o in ODOURS if f"odour_{o}" in self.seen]
        if "octanol" in present and "mch" in present:
            return ("octanol", "mch")
        return (present[0], present[1]) if len(present) == 2 else None

    def trial_summary(self) -> dict:
        """One row of results since `reset_counts`, keyed by Korean column names
        in display order; blank where a measure does not apply."""
        metrics = self.trial_metrics()
        return {m.column: ("" if metrics[m.key] is None else metrics[m.key]) for m in TRIAL_METRICS}


# --- presets ------------------------------------------------------------------------

#: Ready-made rooms: name -> (label, [(kind, x, y, params), ...]). The fly starts
#: at the centre facing +x (or a random heading in a trial), so nothing sits on
#: its spawn point. Walls are obstacles drawn long and thin (``hx``/``hy`` half
#: extents, 2 mm thick); corridors are at least 16 mm wide, over three times the
#: 4.7 mm radius of the tightest turn, so the fly can turn round in them.
_WALL = 1.0


def _wall(x0: float, y0: float, x1: float, y1: float) -> tuple:
    """An axis-aligned wall from (x0, y0) to (x1, y1), as an obstacle entry."""
    if abs(y1 - y0) < abs(x1 - x0):
        return ("obstacle", (x0 + x1) / 2, y0, {"hx": abs(x1 - x0) / 2, "hy": _WALL})
    return ("obstacle", x0, (y0 + y1) / 2, {"hx": _WALL, "hy": abs(y1 - y0) / 2})


def _sugar(x, y, sugar="sucrose", molar=1.0, volume=100.0, odour=None, strength=1.0):
    entries = [("sugar", x, y, {"sugar": sugar, "molar": molar, "volume": volume})]
    if odour:
        entries.append(("odour", x, y, {"odour": odour, "strength": strength}))
    return entries


#: The T-maze's walls, shared by the rooms built on it.
_T_MAZE = [
    _wall(-12.0, 9.0, 18.0, 9.0),
    _wall(-12.0, -9.0, 18.0, -9.0),
    _wall(-13.0, -10.0, -13.0, 10.0),
    _wall(18.0, 10.0, 18.0, 38.0),
    _wall(18.0, -38.0, 18.0, -10.0),
    _wall(36.0, -39.0, 36.0, 39.0),
    _wall(17.0, 39.0, 37.0, 39.0),
    _wall(17.0, -39.0, 37.0, -39.0),
]
#: Four blocks leaving a plus-shaped corridor 18 mm wide, the fly at its centre.
_PLUS_MAZE = [("obstacle", sx * 29.5, sy * 29.5, {"hx": 20.5, "hy": 20.5})
              for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)]


def _odour(x, y, odour, strength=1.0):
    return ("odour", x, y, {"odour": odour, "strength": strength})


def _patch(x, y, colour, half=12.0):
    return ("patch", x, y, {"colour": colour, "half": half})


def _shock(x, y, volts=60.0, half=12.0):
    return ("shock", x, y, {"volts": volts, "half": half})


PRESETS: dict[str, tuple[str, list]] = {
    # --- basics ---
    "empty": ("빈 방", []),
    "scented_sugar": ("식초 향이 나는 설탕", _sugar(28.0, 22.0, odour="vinegar")),
    # Same drop, same place, no smell: the comparison the persona test asked
    # for ("does the smell make it faster?") needs everything else equal.
    "plain_sugar": ("냄새 없는 설탕 (식초 설탕과 같은 자리)", _sugar(28.0, 22.0)),
    "vinegar_only": ("식초 냄새만 (먹을 것 없음)", [_odour(28.0, 22.0, "vinegar")]),
    "big_scented_sugar": ("큰 식초 설탕 500 nl (실컷 먹기)", _sugar(28.0, 22.0, volume=500.0, odour="vinegar")),
    # Straight ahead of a fly facing +x: found at once, eaten in about 6 s of
    # sipping, and then gone -- what is left to watch is the search around it.
    "near_drop": ("바로 앞의 작은 설탕 한 방울", _sugar(7.0, 0.0, volume=30.0)),
    "many_sugars": ("작은 설탕 여러 개 (두 곳만 식초 향)", [
        *_sugar(25.0, 25.0, volume=50.0, odour="vinegar"),
        *_sugar(-25.0, -25.0, volume=50.0, odour="vinegar"),
        *_sugar(-25.0, 25.0, volume=50.0),
        *_sugar(25.0, -25.0, volume=50.0),
        *_sugar(0.0, 36.0, volume=50.0),
        *_sugar(0.0, -36.0, volume=50.0),
    ]),
    "scattered_plain": ("냄새 없는 작은 설탕 12개", [
        entry for x, y in ((36.0, 36.0), (-36.0, 36.0), (-36.0, -36.0), (36.0, -36.0),
                           (36.0, 12.0), (-36.0, 12.0), (-36.0, -12.0), (36.0, -12.0),
                           (12.0, 36.0), (-12.0, 36.0), (-12.0, -36.0), (12.0, -36.0))
        for entry in _sugar(x, y, volume=30.0)]),
    # --- sugars compared ---
    # Burke & Waddell 2011's comparison: one taste, with and without calories.
    # The first version set arabinose against sorbitol alone at (-26, 24) and
    # (26, -24); sorbitol is never fed on and neither odour pulls innately, so
    # in 120 s the fly found neither drop. Measured here, 4 seeds x 150 s: the
    # first drop found at 1.5-67 s; the mixture took hunger from 0.80 to
    # 0.25-0.56, arabinose alone left it rising to 0.88.
    "sweet_vs_nutrient": ("단맛만 vs 단맛+영양", [
        *_sugar(18.0, 14.0, sugar="arabinose", volume=200.0, odour="octanol"),
        *_sugar(18.0, -14.0, sugar="arabinose_sorbitol", volume=200.0, odour="mch"),
    ]),
    # The drops of `sweet_vs_nutrient`, differing in concentration instead.
    "strong_vs_weak_sugar": ("진한 설탕 1 M(옥탄올) vs 묽은 설탕 0.05 M(MCH)", [
        *_sugar(18.0, 14.0, molar=1.0, volume=200.0, odour="octanol"),
        *_sugar(18.0, -14.0, molar=0.05, volume=200.0, odour="mch"),
    ]),
    "sugar_types": ("설탕 네 가지 (자당·과당·아라비노스·소르비톨)", [
        *_sugar(22.0, 22.0, sugar="sucrose", volume=200.0, odour="vinegar", strength=0.4),
        *_sugar(-22.0, 22.0, sugar="fructose", volume=200.0, odour="vinegar", strength=0.4),
        *_sugar(-22.0, -22.0, sugar="arabinose", volume=200.0, odour="vinegar", strength=0.4),
        *_sugar(22.0, -22.0, sugar="sorbitol", volume=200.0, odour="vinegar", strength=0.4),
    ]),
    "big_arabinose": ("큰 아라비노스 500 nl (단맛만, 식초 향)",
                      _sugar(28.0, 22.0, sugar="arabinose", volume=500.0, odour="vinegar")),
    "sorbitol_only": ("소르비톨 (영양만, 맛없음, 식초 향)",
                      _sugar(28.0, 22.0, sugar="sorbitol", volume=500.0, odour="vinegar")),
    "big_vs_small": ("큰 방울 500 nl vs 작은 방울 20 nl (냄새 없음)", [
        *_sugar(20.0, 18.0, volume=500.0),
        *_sugar(-20.0, -18.0, volume=20.0),
    ]),
    # --- odours ---
    "three_odours": ("냄새 세 개 (식초·옥탄올·MCH, 먹을 것 없음)", [
        _odour(0.0, 30.0, "vinegar"), _odour(-26.0, -15.0, "octanol"), _odour(26.0, -15.0, "mch"),
    ]),
    "strong_weak_vinegar": ("진한 식초 vs 옅은 식초 (먹을 것 없음)", [
        _odour(-25.0, 20.0, "vinegar", 1.0), _odour(25.0, -20.0, "vinegar", 0.2),
    ]),
    "vinegar_steps": ("옅은 식초 징검다리 끝의 설탕", [
        _odour(14.0, 0.0, "vinegar", 0.15), _odour(26.0, 14.0, "vinegar", 0.15),
        *_sugar(36.0, 30.0, odour="vinegar", strength=0.3),
    ]),
    # --- learning: training rooms ---
    # Neither odour pulls innately, so the sugar sits near the start to be
    # found at all: at (-18, 15) the fly had not found it after 90 s.
    "two_odours": ("냄새 두 개, 한쪽에만 설탕 (냄새 좋아하기 학습)", [
        *_sugar(10.0, 8.0, volume=300.0, odour="octanol"),
        _odour(-25.0, -20.0, "mch"),
    ]),
    # The same pair placed symmetrically, 17 mm from the start each, so the
    # rewarded side can be swapped without moving anything.
    "octanol_sugar": ("옥탄올 쪽에만 설탕 (좋아하기 훈련)", [
        *_sugar(14.0, 10.0, volume=300.0, odour="octanol"),
        _odour(-14.0, -10.0, "mch"),
    ]),
    "mch_sugar": ("MCH 쪽에만 설탕 (반대로 훈련)", [
        *_sugar(14.0, 10.0, volume=300.0, odour="mch"),
        _odour(-14.0, -10.0, "octanol"),
    ]),
    "odour_shock": ("냄새 하나에만 전기 (냄새 피하기 학습)", [
        _odour(20.0, 15.0, "mch"), _shock(20.0, 15.0), _odour(-20.0, -15.0, "octanol"),
    ]),
    "octanol_shock": ("옥탄올에만 전기 (반대로 훈련)", [
        _odour(20.0, 15.0, "octanol"), _shock(20.0, 15.0), _odour(-20.0, -15.0, "mch"),
    ]),
    "vinegar_shock": ("식초 냄새에 전기 (먹을 것 없음)", [
        _odour(24.0, 18.0, "vinegar"), _shock(24.0, 18.0),
    ]),
    "blue_shock": ("파란 바닥에만 전기 (색 피하기 학습)", [
        _patch(-26.0, -24.0, "blue", 13.0), _shock(-26.0, -24.0, 60.0, 13.0), _patch(26.0, 24.0, "green", 13.0),
    ]),
    "green_shock": ("초록 바닥에만 전기 (반대로 훈련)", [
        _patch(-26.0, -24.0, "green", 13.0), _shock(-26.0, -24.0, 60.0, 13.0), _patch(26.0, 24.0, "blue", 13.0),
    ]),
    # A faint vinegar leads the fly to the sugar; without it the drop was not
    # found in 90 s and green was never paired with anything.
    "green_reward": ("초록 바닥에만 설탕 (색 좋아하기 학습)", [
        _patch(-22.0, 18.0, "green"),
        *_sugar(-22.0, 18.0, volume=500.0, odour="vinegar", strength=0.3),
        _patch(22.0, -18.0, "green"),
        _patch(22.0, 18.0, "blue"),
    ]),
    "blue_reward": ("파란 바닥에만 설탕 (반대로 훈련)", [
        _patch(-22.0, 18.0, "blue"),
        *_sugar(-22.0, 18.0, volume=500.0, odour="vinegar", strength=0.3),
        _patch(22.0, -18.0, "blue"),
        _patch(22.0, 18.0, "green"),
    ]),
    # Octanol pulls nothing innately: a fly finds this drop by chance until it
    # has fed on it once, and by following the smell after that.
    "octanol_sugar_far": ("멀리 있는 옥탄올 설탕 (배워야 빨라짐)", _sugar(26.0, 20.0, volume=300.0, odour="octanol")),
    "compound_cue": ("옥탄올 냄새 + 파란 바닥 위의 설탕 (단서 두 개)", [
        _patch(-16.0, 14.0, "blue"),
        *_sugar(-16.0, 14.0, volume=500.0, odour="octanol"),
        _patch(16.0, -14.0, "green"),
        _odour(16.0, -14.0, "mch"),
    ]),
    # --- learning: test rooms, with nothing to reward or punish ---
    # Each keeps its training room's geometry: `odour_choice` is `odour_shock`
    # without the shock, `colour_choice` is `blue_shock` without it.
    "odour_choice": ("시험 방: 옥탄올 vs MCH", [
        _odour(-20.0, -15.0, "octanol"), _odour(20.0, 15.0, "mch"),
    ]),
    "colour_choice": ("시험 방: 파랑 vs 초록", [
        _patch(-26.0, -24.0, "blue", 13.0), _patch(26.0, 24.0, "green", 13.0),
    ]),
    "colour_quadrants": ("시험 방: 파랑·초록 네 칸", [
        _patch(20.0, 20.0, "blue", 18.0), _patch(-20.0, -20.0, "blue", 18.0),
        _patch(-20.0, 20.0, "green", 18.0), _patch(20.0, -20.0, "green", 18.0),
    ]),
    "t_maze_odours": ("시험 방: T자 미로 (위 팔 옥탄올, 아래 팔 MCH)", [
        *_T_MAZE, _odour(27.0, 30.0, "octanol"), _odour(27.0, -30.0, "mch"),
    ]),
    # --- conflicts and places ---
    "safe_vs_shocked": ("설탕 두 개, 한쪽만 전기 위", [
        *_sugar(-15.0, 12.0, odour="octanol"),
        *_sugar(15.0, -12.0, odour="mch"),
        _shock(15.0, -12.0, 40.0, 10.0),
    ]),
    "conflict": ("전기 구역 한가운데의 설탕 (배고픔 vs 아픔)", [
        _shock(24.0, 18.0, 30.0, 10.0),
        *_sugar(24.0, 18.0, odour="vinegar"),
    ]),
    "volt_choice": ("약한 전기 20 V(파랑) vs 센 전기 80 V(초록)", [
        _patch(-24.0, 20.0, "blue"), _shock(-24.0, 20.0, 20.0),
        _patch(24.0, -20.0, "green"), _shock(24.0, -20.0, 80.0),
    ]),
    # Seven 14 mm plates edge to edge: a band from wall to wall at x = 13-27.
    "shock_band": ("전기 띠(30 V) 너머의 식초 설탕", [
        *[_shock(20.0, y, 30.0, 7.0) for y in (-42.0, -28.0, -14.0, 0.0, 14.0, 28.0, 42.0)],
        *_sugar(38.0, 0.0, odour="vinegar"),
    ]),
    # Three shock plates around the start, each with MCH rising from its middle:
    # the smell is the only warning, so without it every visit is a surprise.
    "cued_shocks": ("MCH 냄새가 나는 전기 구역 세 곳", [
        entry for x, y in ((22.0, 12.0), (-18.0, 20.0), (-4.0, -26.0))
        for entry in (_shock(x, y, 60.0, 9.0), _odour(x, y, "mch"))]),
    # The right half of the room shocks (plates 24 mm square) except one green
    # square: nothing marks the plates, the island is the only thing to see.
    "green_island": ("전기 바닥(40 V) 속 초록 섬", [
        *[_shock(x, y, 40.0) for x in (14.0, 38.0) for y in (-36.0, -12.0, 12.0, 36.0)
          if (x, y) != (38.0, 12.0)],
        _patch(38.0, 12.0, "green"),
    ]),
    # --- walls and mazes ---
    "wall_between": ("벽 너머의 설탕", [
        _wall(14.0, -24.0, 14.0, 24.0),
        *_sugar(32.0, 0.0, odour="vinegar"),
    ]),
    "two_rooms": ("가운데 벽의 문 하나 (건너편 식초 설탕)", [
        _wall(16.0, -50.0, 16.0, -9.0),
        _wall(16.0, 9.0, 16.0, 50.0),
        *_sugar(36.0, 26.0, odour="vinegar"),
    ]),
    "corridor": ("통로 끝의 설탕", [
        _wall(6.0, 9.0, 46.0, 9.0),
        _wall(6.0, -9.0, 46.0, -9.0),
        *_sugar(40.0, 0.0, odour="vinegar"),
    ]),
    # A 34 mm room with a 16 mm door in its lower wall, below the drop.
    "room_in_room": ("문 하나 달린 작은 방 안의 식초 설탕", [
        _wall(8.0, 44.0, 44.0, 44.0),
        _wall(44.0, 8.0, 44.0, 44.0),
        _wall(8.0, 8.0, 8.0, 44.0),
        _wall(8.0, 8.0, 18.0, 8.0),
        _wall(34.0, 8.0, 44.0, 8.0),
        *_sugar(26.0, 26.0, odour="vinegar"),
    ]),
    # A T-maze (the classic choice apparatus of Tully & Quinn 1985, here walked
    # rather than air-pushed): the fly starts in the stem, the arms run up and
    # down the far end, and only the upper arm holds scented sugar.
    "t_maze": ("T자 미로 (위쪽 팔 끝에 식초 설탕)", [
        *_T_MAZE,
        *_sugar(27.0, 30.0, odour="vinegar"),
    ]),
    "t_maze_octanol": ("T자 미로 (위쪽 팔 끝에 옥탄올 설탕)", [
        *_T_MAZE,
        *_sugar(27.0, 30.0, odour="octanol"),
    ]),
    "plus_maze": ("십자 미로 (오른쪽 팔 끝에 식초 설탕)", [
        *_PLUS_MAZE,
        *_sugar(42.0, 0.0, odour="vinegar"),
    ]),
    "plus_maze_odours": ("십자 미로: 식초 설탕·옥탄올·MCH·빈 팔", [
        *_PLUS_MAZE,
        *_sugar(42.0, 0.0, odour="vinegar"),
        _odour(0.0, 42.0, "octanol"),
        _odour(-42.0, 0.0, "mch"),
    ]),
    "zigzag": ("지그재그 미로", [
        _wall(14.0, -50.0, 14.0, 26.0),
        _wall(32.0, -26.0, 32.0, 50.0),
        *_sugar(42.0, -10.0, odour="vinegar"),
    ]),
    # Three by three cells of 33 mm. The way to the upper right runs down, left,
    # up and across; straight toward the smell is a wall, and right of the start
    # is a dead end.
    "dead_end_maze": ("막다른 길이 있는 미로 (오른쪽 위 식초 설탕)", [
        _wall(16.7, -50.0, 16.7, -16.7),
        _wall(-16.7, -16.7, -16.7, 16.7),
        _wall(-16.7, 16.7, 16.7, 16.7),
        _wall(16.7, 16.7, 50.0, 16.7),
        *_sugar(34.0, 34.0, odour="vinegar"),
    ]),
    "pillars": ("기둥 숲 너머의 설탕", [
        *[("obstacle", x, y, {"hx": 3.0, "hy": 3.0})
          for x, y in ((12.0, -14.0), (12.0, 6.0), (20.0, -4.0), (20.0, 16.0),
                       (28.0, -14.0), (28.0, 6.0), (12.0, 26.0), (28.0, 26.0))],
        *_sugar(38.0, 4.0, odour="vinegar"),
    ]),
    "obstacle_field": ("블록 16개 사이의 식초 설탕", [
        *[("obstacle", x, y, {"hx": 3.0, "hy": 3.0})
          for x, y in ((-36.0, -30.0), (-18.0, -40.0), (0.0, -26.0), (20.0, -36.0), (38.0, -22.0),
                       (-40.0, -8.0), (-22.0, 12.0), (18.0, -12.0), (34.0, 4.0),
                       (-34.0, 26.0), (-12.0, 30.0), (8.0, 22.0), (26.0, 34.0), (42.0, 40.0),
                       (-4.0, 44.0), (-44.0, 44.0))],
        *_sugar(40.0, 22.0, odour="vinegar"),
    ]),
}

#: One line on what each room is for, shown when it is loaded.
PRESET_NOTES: dict[str, str] = {
    "empty": "아무것도 없는 방입니다. 초파리가 그냥 돌아다니는 모습을 봅니다.",
    "scented_sugar": "식초 냄새를 따라가 설탕을 찾는지 봅니다.",
    "plain_sugar": "냄새가 없으니 우연히 밟아야만 찾습니다. 식초 설탕 방과 비교해 보세요.",
    "vinegar_only": "먹을 것 없이 냄새만 있을 때도 냄새 쪽으로 가는지 봅니다.",
    "big_scented_sugar": "500 nl라서 배부를 때까지 먹을 수 있습니다. 배고픔에 따라 먹는 시간을 비교하기 좋습니다.",
    "near_drop": "바로 앞의 작은 방울을 다 먹은 뒤 그 주변을 맴도는지(주변 탐색) 봅니다.",
    "many_sugars": "식초 향이 나는 두 방울을 먼저 찾는지 봅니다.",
    "scattered_plain": "냄새 단서 없이 흩어진 먹이를 찾는 방법을 봅니다.",
    "sweet_vs_nutrient": "단맛만 있는 아라비노스와 영양도 있는 혼합물 중 무엇을 먹고 무엇을 기억하는지 봅니다.",
    "strong_vs_weak_sugar": "진한 설탕과 묽은 설탕을 먹은 뒤 각 냄새의 가치가 어떻게 달라지는지 봅니다.",
    "sugar_types": "네 가지 설탕 중 맛이 없는 소르비톨은 먹지 않는지 봅니다.",
    "big_arabinose": "단맛은 있지만 영양이 없어 먹어도 배고픔이 줄지 않습니다.",
    "sorbitol_only": "영양은 있지만 맛이 없어 초파리가 먹기 시작하지 않습니다.",
    "big_vs_small": "방울 크기가 찾는 데 영향을 주는지 봅니다.",
    "three_odours": "타고난 끌림이 있는 식초와 중립인 두 냄새를 비교합니다.",
    "strong_weak_vinegar": "진한 냄새와 옅은 냄새 중 어디에 더 오래 머무는지 봅니다.",
    "vinegar_steps": "옅은 냄새를 이어 따라가 설탕까지 가는지 봅니다.",
    "two_odours": "옥탄올 쪽 설탕을 먹으면서 옥탄올을 좋아하게 되는지 봅니다.",
    "octanol_sugar": "좋아하기 훈련 방입니다. 시험은 '시험 방: 옥탄올 vs MCH'에서 합니다.",
    "mch_sugar": "옥탄올 설탕 방의 거울 방입니다. 냄새 자체의 좋고 싫음을 가려내는 데 씁니다.",
    "odour_shock": "피하기 훈련 방입니다. 시험은 '시험 방: 옥탄올 vs MCH'에서 합니다.",
    "octanol_shock": "MCH 전기 방의 거울 방입니다.",
    "vinegar_shock": "좋아하던 식초도 전기와 함께 겪으면 싫어지는지 봅니다.",
    "blue_shock": "색 피하기 훈련 방입니다. 시험은 '시험 방: 파랑 vs 초록'에서 합니다.",
    "green_shock": "파랑 전기 방의 거울 방입니다.",
    "green_reward": "색 좋아하기 훈련 방입니다. 옅은 식초가 설탕으로 이끕니다.",
    "blue_reward": "초록 설탕 방의 거울 방입니다.",
    "octanol_sugar_far": "옥탄올은 원래 끌림이 없어 처음엔 우연히 찾습니다. 한 번 먹고 나면 냄새를 따라갑니다.",
    "compound_cue": "냄새와 색이 함께 있을 때 무엇을 배우는지 봅니다.",
    "odour_choice": "상도 벌도 없는 시험 방입니다. 배운 것만으로 어느 냄새 쪽에 머무는지 잽니다.",
    "colour_choice": "상도 벌도 없는 시험 방입니다. 배운 것만으로 어느 색 쪽에 머무는지 잽니다.",
    "colour_quadrants": "바닥 전체가 두 색으로 나뉜 시험 방입니다.",
    "t_maze_odours": "실제 T자 미로처럼 양쪽 팔에 다른 냄새가 있습니다.",
    "safe_vs_shocked": "같은 설탕 둘 중 전기가 없는 쪽을 고르게 되는지 봅니다.",
    "conflict": "배고픔과 전기 중 무엇이 이기는지 봅니다.",
    "volt_choice": "전기 세기에 따라 피하는 정도가 다른지 봅니다.",
    "shock_band": "설탕에 가려면 전기 띠를 건너야 합니다. 배고픔이 아픔을 이기는지 봅니다.",
    "cued_shocks": "전기 구역마다 MCH 냄새가 납니다. 몇 번 겪고 나면 냄새만 맡고도 피하는지 봅니다.",
    "green_island": "전기가 없는 초록 섬을 찾아 머물게 되는지 봅니다.",
    "wall_between": "냄새는 벽을 통과하지만 초파리는 벽을 돌아가야 합니다.",
    "two_rooms": "문을 찾아 건너편 방으로 가는지 봅니다.",
    "corridor": "좁은 통로를 따라 끝까지 가는지 봅니다.",
    "room_in_room": "냄새는 새어 나오지만 들어가는 문은 아래쪽 하나뿐입니다.",
    "t_maze": "갈림길에서 식초 냄새가 나는 팔을 고르는지 봅니다.",
    "t_maze_octanol": "옥탄올은 원래 끌림이 없어, 설탕 맛을 본 뒤에야 그 팔을 골라 가는지 봅니다.",
    "plus_maze": "네 갈래 중 냄새가 나는 갈래를 고르는지 봅니다.",
    "plus_maze_odours": "네 갈래에 서로 다른 냄새가 있습니다.",
    "zigzag": "벽 끝을 돌아 지그재그로 가는지 봅니다.",
    "dead_end_maze": "냄새 쪽이 막혀 있어 돌아가야 하는 어려운 미로입니다.",
    "pillars": "기둥 사이를 빠져나가는지 봅니다.",
    "obstacle_field": "블록 사이를 요리조리 빠져나가는지 봅니다.",
}

#: How the page groups the presets, in order.
PRESET_GROUPS: list[tuple[str, list[str]]] = [
    ("기본", ["empty", "scented_sugar", "plain_sugar", "vinegar_only", "big_scented_sugar",
             "near_drop", "many_sugars", "scattered_plain"]),
    ("설탕 비교", ["sweet_vs_nutrient", "strong_vs_weak_sugar", "sugar_types", "big_arabinose",
                "sorbitol_only", "big_vs_small"]),
    ("냄새", ["three_odours", "strong_weak_vinegar", "vinegar_steps"]),
    ("학습 - 훈련 방", ["two_odours", "octanol_sugar", "mch_sugar", "odour_shock", "octanol_shock",
                    "vinegar_shock", "blue_shock", "green_shock", "green_reward", "blue_reward",
                    "compound_cue", "octanol_sugar_far"]),
    ("학습 - 시험 방", ["odour_choice", "colour_choice", "colour_quadrants", "t_maze_odours"]),
    ("갈등과 장소", ["safe_vs_shocked", "conflict", "volt_choice", "shock_band", "cued_shocks", "green_island"]),
    ("벽과 미로", ["wall_between", "two_rooms", "corridor", "room_in_room", "t_maze", "t_maze_octanol", "plus_maze",
                "plus_maze_odours", "zigzag", "dead_end_maze", "pillars", "obstacle_field"]),
]


def load_preset(sandbox: Sandbox, name: str) -> None:
    """Empty the room and furnish it as `name` describes. The fly is untouched."""
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    sandbox.clear()
    for kind, x, y, params in PRESETS[name][1]:
        sandbox.place(kind, x, y, **params)

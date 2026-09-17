"""The experiment report: a Korean middle-school inquiry report (탐구 보고서).

The headings follow what graders look for. KOSAC's checklist for inquiry under
the 2022 curriculum asks for:

- named manipulated and dependent variables, and evidence the rest were held
  constant;
- tables with units, and graphs with named axes and the manipulated variable on
  the x axis;
- repeated measurements summarised by their mean;
- a conclusion that matches the data;
- when the hypothesis failed, the sources of error, and when it held, a new
  question.

The national science fair also asks that tools and AI be disclosed.

The report fills in everything that is data -- method, variables, charts,
tables, the verdict on the hypothesis -- and leaves motivation, reasons and
conclusions as boxes the student writes in, with sentence starters but no
answers: the fair's interview checks that the student understands the work.
Edits are kept in the browser and saved with the page.

Statistics stop at what a middle-schooler uses: counts, means, ranges and
differences, with n on every chart. The sign test behind the verdict is
written as coin flips.
"""

from __future__ import annotations

import csv
import html
import math
import time
from pathlib import Path

import numpy as np

from flyplay.experiment import (
    EXPERIMENT_SETS,
    PHASE_LABELS,
    ROLE_LABELS,
    VERDICT_TEXT,
    Protocol,
    describe_steps,
    evaluate,
    metric_value,
    prediction_text,
    read_json,
    read_records,
    room_label,
)
from flyplay.sandbox import (
    METRICS_BY_KEY,
    NEAR_RADIUS,
    ODOUR_RGBA,
    TRIAL_METRICS,
    drop_radius,
    duration_ko,
)

A_COLOUR, B_COLOUR = "#d9622b", "#2f6fc4"
CSV_NAME = "trials.csv"
REPORT_NAME = "report.html"


def esc(text) -> str:
    return html.escape("" if text is None else str(text))


def fmt(value: float | None, key: str = "") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    digits = 2 if key.endswith("_pi") or key.startswith("value_") or key == "end_hunger" else 1
    return f"{value:.{digits}f}"


# --- CSV ------------------------------------------------------------------------------


def csv_rows(protocol: Protocol, records: list[dict]) -> list[dict]:
    rows = []
    for r in records:
        row = {
            "실험": protocol.title, "조건": r["condition_label"], "파리": r["fly"], "시행 순서": r["seq"],
            "단계": ROLE_LABELS.get(r["role"], r["role"]), "반복": r["repeat"], "방": r["room_label"],
            "새 파리": "예" if r["new_fly"] else "아니오",
            "시작 방향(°)": "" if r["heading_deg"] is None else r["heading_deg"],
            "시작 배고픔": r["start_hunger"], "앞서 쉰 시간": duration_ko(r["rest_before_s"]) if r["rest_before_s"] else "",
            "시간(초)": r["seconds"], "중간에 멈춤": "예" if r["stopped"] else "",
        }
        for m in TRIAL_METRICS:
            value = r["metrics"].get(m.key)
            row[m.column] = "" if value is None else value
        row["다시 보기 파일"] = r.get("replay") or ""
        rows.append(row)
    return rows


def write_outputs(directory: Path) -> None:
    """trials.csv (UTF-8 with BOM, so Excel reads the Korean) and report.html."""
    directory = Path(directory)
    protocol = Protocol.from_dict(read_json(directory / "protocol.json"))
    records = read_records(directory)
    rows = csv_rows(protocol, records)
    if rows:
        with (directory / CSV_NAME).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (directory / REPORT_NAME).write_text(build_report(directory), encoding="utf-8")


# --- SVG charts -------------------------------------------------------------------------


def _nice_range(values: list[float], pad: float = 0.08) -> tuple[float, float]:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return 0.0, 1.0
    lo, hi = min(finite), max(finite)
    if hi - lo < 1e-9:
        lo, hi = lo - 1.0, hi + 1.0
    span = hi - lo
    return lo - span * pad, hi + span * pad


def _ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    raw = (hi - lo) / max(1, n)
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=raw)
    first = math.ceil(lo / step) * step
    return [first + i * step for i in range(int((hi - first) / step) + 1)]


def _tick_label(v: float) -> str:
    return f"{v:.0f}" if abs(v) >= 10 or float(v).is_integer() else f"{v:.2f}".rstrip("0").rstrip(".")


def svg_trials_chart(series: list[dict], n_trials: int, y_label: str, roles: list[str],
                     fixed: tuple[float, float] | None = None) -> str:
    """Trial order on x, a measure on y: each condition's mean over flies as a
    line with its min-max range shaded, every fly faint behind it. Training
    trials are shaded grey."""
    width, height = 680, 280
    left, right, top, bottom = 58, 16, 16, 44
    values = [v for s in series for fly in s["flies"] for v in fly]
    lo, hi = fixed or _nice_range(values)
    px = lambda i: left + (i + 0.5) * (width - left - right) / max(1, n_trials)
    py = lambda v: top + (hi - v) / (hi - lo) * (height - top - bottom)
    out = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">']
    for i, role in enumerate(roles):
        if role == "train":
            x0 = left + i * (width - left - right) / n_trials
            out.append(f'<rect x="{x0:.1f}" y="{top}" width="{(width - left - right) / n_trials:.1f}" '
                       f'height="{height - top - bottom}" fill="#ececec"/>')
    for t in _ticks(lo, hi):
        y = py(t)
        out.append(f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#ddd"/>')
        out.append(f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end" class="tick">{_tick_label(t)}</text>')
    if lo < 0 < hi:
        out.append(f'<line x1="{left}" x2="{width - right}" y1="{py(0):.1f}" y2="{py(0):.1f}" stroke="#999"/>')
    for i in range(n_trials):
        label = f"{i + 1}" + ("" if i >= len(roles) else f" {ROLE_LABELS.get(roles[i], '')[:2]}")
        out.append(f'<text x="{px(i):.1f}" y="{height - bottom + 16}" text-anchor="middle" class="tick">{esc(label)}</text>')
    out.append(f'<text x="{(left + width - right) / 2:.1f}" y="{height - 6}" text-anchor="middle" class="axis">시행 순서 (회색 칸 = 훈련)</text>')
    out.append(f'<text transform="translate(14 {(top + height - bottom) / 2:.1f}) rotate(-90)" text-anchor="middle" class="axis">{esc(y_label)}</text>')
    for s in series:
        for fly in s["flies"]:
            pts = [(px(i), py(v)) for i, v in enumerate(fly) if v is not None]
            if len(pts) > 1:
                out.append('<polyline fill="none" stroke="{}" stroke-opacity="0.18" stroke-width="1" points="{}"/>'.format(
                    s["colour"], " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)))
        band = [(i, s["lo"][i], s["hi"][i]) for i in range(n_trials) if s["lo"][i] is not None]
        if len(band) > 1:
            upper = " ".join(f"{px(i):.1f},{py(h):.1f}" for i, _, h in band)
            lower = " ".join(f"{px(i):.1f},{py(l):.1f}" for i, l, _ in reversed(band))
            out.append(f'<polygon points="{upper} {lower}" fill="{s["colour"]}" fill-opacity="0.12"/>')
        pts = [(px(i), py(v)) for i, v in enumerate(s["mean"]) if v is not None]
        if len(pts) > 1:
            out.append('<polyline fill="none" stroke="{}" stroke-width="2.5" points="{}"/>'.format(
                s["colour"], " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)))
        for x, y in pts:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{s["colour"]}"/>')
    out.append("</svg>")
    return "".join(out)


def svg_pair_dots(a: dict[int, float], b: dict[int, float], y_label: str) -> str:
    """One dot per fly in A and in B, a line joining each pair, the means as bars."""
    width, height = 420, 280
    left, right, top, bottom = 58, 20, 16, 40
    lo, hi = _nice_range(list(a.values()) + list(b.values()))
    py = lambda v: top + (hi - v) / (hi - lo) * (height - top - bottom)
    xa, xb = left + 90, width - right - 90
    out = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">']
    for t in _ticks(lo, hi):
        y = py(t)
        out.append(f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#ddd"/>')
        out.append(f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end" class="tick">{_tick_label(t)}</text>')
    if lo < 0 < hi:
        out.append(f'<line x1="{left}" x2="{width - right}" y1="{py(0):.1f}" y2="{py(0):.1f}" stroke="#999"/>')
    for fly in sorted(set(a) & set(b)):
        out.append(f'<line x1="{xa}" y1="{py(a[fly]):.1f}" x2="{xb}" y2="{py(b[fly]):.1f}" stroke="#bbb"/>')
    for x, scores, colour, name in ((xa, a, A_COLOUR, "A"), (xb, b, B_COLOUR, "B")):
        for fly, v in scores.items():
            jitter = ((fly * 37) % 11 - 5) * 1.6
            out.append(f'<circle cx="{x + jitter:.1f}" cy="{py(v):.1f}" r="4" fill="{colour}" fill-opacity="0.75"/>')
        if scores:
            m = float(np.mean(list(scores.values())))
            out.append(f'<line x1="{x - 30}" x2="{x + 30}" y1="{py(m):.1f}" y2="{py(m):.1f}" stroke="{colour}" stroke-width="3"/>')
        out.append(f'<text x="{x}" y="{height - bottom + 18}" text-anchor="middle" class="axis">{name} (n={len(scores)})</text>')
    out.append(f'<text transform="translate(14 {(top + height - bottom) / 2:.1f}) rotate(-90)" text-anchor="middle" class="axis">{esc(y_label)}</text>')
    out.append("</svg>")
    return "".join(out)


#: Heatmap cells across the 100 mm room: 4 mm. Paths are recorded at 5 Hz, so a
#: 60 s trial gives 300 points; finer cells would be mostly empty.
HEAT_CELLS = 25
#: Pale yellow to dark red: on the report's white page the viewer's dark-map
#: palette turned every lightly visited cell pink and hid the hot spots.
HEAT_STOPS = ((0.0, (255, 245, 180)), (0.4, (254, 190, 75)), (0.7, (240, 95, 30)), (1.0, (150, 15, 40)))


def _heat_colour(v: float) -> str:
    for (t0, c0), (t1, c1) in zip(HEAT_STOPS, HEAT_STOPS[1:]):
        if v <= t1:
            u = (v - t0) / (t1 - t0)
            return "rgb({},{},{})".format(*(round(a + (b - a) * u) for a, b in zip(c0, c1)))
    return "rgb({},{},{})".format(*HEAT_STOPS[-1][1])


def heat_counts(records: list[dict]) -> np.ndarray:
    """Path points per cell, summed over `records`, rows from the top (y = +50)."""
    counts = np.zeros((HEAT_CELLS, HEAT_CELLS))
    cell = 100.0 / HEAT_CELLS
    for r in records:
        for _, x, y in r["path"]:
            col, row = int((x + 50.0) // cell), int((50.0 - y) // cell)
            if 0 <= col < HEAT_CELLS and 0 <= row < HEAT_CELLS:
                counts[row, col] += 1
    return counts


def svg_heat(items: list[dict], counts: np.ndarray, peak: float, size: int = 260) -> str:
    """The room with where the flies spent their time painted over it. `peak`
    is shared by the maps being compared, so equal colours mean equal time."""
    cell = size / HEAT_CELLS
    cells = []
    if peak > 0:
        for row in range(HEAT_CELLS):
            for col in range(HEAT_CELLS):
                if counts[row, col] <= 0:
                    continue
                v = math.sqrt(counts[row, col] / peak)
                if v < 0.12:
                    continue  # a moment's pass, not a place the flies stayed
                cells.append(f'<rect x="{col * cell:.1f}" y="{row * cell:.1f}" width="{cell + 0.3:.1f}" '
                             f'height="{cell + 0.3:.1f}" fill="{_heat_colour(v)}" fill-opacity="{min(0.9, 0.2 + 0.7 * v):.2f}"/>')
    # Over the floor colours, under the drops, odours and walls, which stay readable.
    return svg_room(items, None, size, overlay="".join(cells))


def svg_room(items: list[dict], path: list | None = None, size: int = 210, colour: str = "#333",
             overlay: str = "") -> str:
    """The room from above as the trial began, with the fly's path. `overlay`
    (SVG) goes over the floor patches and under everything else."""
    s = size / 100.0
    X = lambda x: (x + 50.0) * s
    Y = lambda y: (50.0 - y) * s
    out = [f'<svg viewBox="0 0 {size} {size}" class="room" role="img">',
           f'<rect x="0" y="0" width="{size}" height="{size}" fill="#f7f7f5" stroke="#888"/>']
    for kind in ("patch", "overlay", "shock", "odour", "sugar", "obstacle"):
        if kind == "overlay":
            out.append(overlay)
            continue
        for it in items:
            if it["kind"] != kind:
                continue
            p, x, y = it.get("params", {}), it["x"], it["y"]
            if kind == "patch":
                h = p.get("half", 12.0)
                fill = "#4d7fe0" if p.get("colour") == "blue" else "#4fb05a"
                out.append(f'<rect x="{X(x - h):.1f}" y="{Y(y + h):.1f}" width="{2 * h * s:.1f}" height="{2 * h * s:.1f}" fill="{fill}" fill-opacity="0.55"/>')
            elif kind == "shock":
                h = p.get("half", 12.0)
                dash = "" if p.get("volts", 60.0) > 0 else ' stroke-opacity="0.35"'
                out.append(f'<rect x="{X(x - h):.1f}" y="{Y(y + h):.1f}" width="{2 * h * s:.1f}" height="{2 * h * s:.1f}" fill="none" stroke="#e03a1e" stroke-width="1.5" stroke-dasharray="4 3"{dash}/>')
            elif kind == "odour":
                r, g, b, _ = ODOUR_RGBA.get(p.get("odour", "vinegar"), (0.9, 0.7, 0.2, 1.0))
                rgb = f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"
                out.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="{NEAR_RADIUS * s:.1f}" fill="{rgb}" fill-opacity="0.12" stroke="{rgb}" stroke-opacity="0.6" stroke-dasharray="2 2"/>')
                out.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="2.5" fill="{rgb}"/>')
            elif kind == "sugar":
                r = max(drop_radius(p.get("left", p.get("volume", 100.0))), 1.0) * s
                out.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="{r:.1f}" fill="#fff" stroke="#8a6d1f" stroke-width="1.2"/>')
            else:
                hx, hy = p.get("hx", 4.0), p.get("hy", 4.0)
                out.append(f'<rect x="{X(x - hx):.1f}" y="{Y(y + hy):.1f}" width="{2 * hx * s:.1f}" height="{2 * hy * s:.1f}" fill="#8d8d93"/>')
    if path:
        pts = " ".join(f"{X(px):.1f},{Y(py):.1f}" for _, px, py in path)
        out.append(f'<polyline fill="none" stroke="{colour}" stroke-width="1.2" stroke-opacity="0.85" points="{pts}"/>')
        out.append(f'<circle cx="{X(path[-1][1]):.1f}" cy="{Y(path[-1][2]):.1f}" r="2.5" fill="{colour}"/>')
    out.append(f'<circle cx="{X(0):.1f}" cy="{Y(0):.1f}" r="2.5" fill="none" stroke="#000"/>')
    out.append("</svg>")
    return "".join(out)


# --- the report ---------------------------------------------------------------------------


def _write_box(key: str, placeholder: str, rows: int = 3) -> str:
    return (f'<div class="write" contenteditable="true" data-key="{esc(key)}" '
            f'data-ph="{esc(placeholder)}" style="min-height:{rows * 1.6:.1f}em"></div>')


def _series(protocol: Protocol, records: list[dict], key: str) -> tuple[list[dict], int, list[str]]:
    series, n_trials, roles = [], 0, []
    for c, cond in enumerate(protocol.conditions):
        rs = [r for r in records if r["condition"] == c]
        flies: dict[int, dict[int, float | None]] = {}
        for r in rs:
            flies.setdefault(r["fly"], {})[r["seq"]] = metric_value(r, key)
        n = protocol.trials_per_fly(c)
        if n > n_trials:
            n_trials = n
            roles = [st.role for st in cond.steps for _ in range(st.repeats)]
        rows = [[f.get(i + 1) for i in range(n)] for f in flies.values()]
        mean, lo, hi = [], [], []
        for i in range(n):
            col = [row[i] for row in rows if row[i] is not None]
            mean.append(float(np.mean(col)) if col else None)
            lo.append(float(np.min(col)) if len(col) > 1 else None)
            hi.append(float(np.max(col)) if len(col) > 1 else None)
        series.append({"label": cond.label, "colour": A_COLOUR if c == 0 else B_COLOUR,
                       "flies": rows, "mean": mean, "lo": lo, "hi": hi})
    # Conditions can run different numbers of trials (four trainings against
    # one): pad the shorter so every series covers the chart's x axis.
    for s in series:
        for key in ("mean", "lo", "hi"):
            s[key] += [None] * (n_trials - len(s[key]))
        for row in s["flies"]:
            row += [None] * (n_trials - len(row))
    return series, n_trials, roles


def _prediction_sentence(s, protocol: Protocol) -> str:
    value = protocol.prediction or protocol.expect
    if value == protocol.expect:
        return s.hypothesis
    phase = PHASE_LABELS.get(protocol.phase, protocol.phase)
    return f"{phase}에서 {prediction_text(s, value)}."


def controlled_variables(protocol: Protocol, flies: bool = True) -> list[str]:
    """What was held the same between A and B, in words (통제 변인)."""
    held = [f"조건마다 파리 {protocol.flies}마리, 같은 순서의 시행"] if flies else []
    held += [
        "같은 번호의 A·B 파리는 뇌 배선·무작위 수·출발 방향이 같음",
        "시행마다 방을 처음 상태로 되돌림 (먹은 설탕도 다시 채움)",
    ]
    c0, c1 = protocol.conditions
    if c0.hunger == c1.hunger:
        held.append(f"시작 배고픔 {c0.hunger:.2f}" + (" (시행마다 맞춤)" if c0.hold_hunger else ""))
    lengths = {st.seconds for c in protocol.conditions for st in c.steps}
    if len(lengths) == 1:
        held.append(f"시행 한 번 {duration_ko(lengths.pop())}")
    return held


def _verdict_block(s, protocol: Protocol, result: dict) -> str:
    column = result["column"]
    a, b = result["a"], result["b"]
    if result["verdict"] == "not_enough":
        return f'<div class="verdict unclear"><b>{VERDICT_TEXT["not_enough"]}</b></div>'
    words = {"A>B": "A가 B보다 컸습니다", "A<B": "A가 B보다 작았습니다", "same": "A와 B가 거의 같았습니다"}
    agree = {
        "A>B": f"{result['pairs']}쌍 중 {result['a_higher']}쌍에서 A가 더 컸습니다",
        "A<B": f"{result['pairs']}쌍 중 {result['a_lower']}쌍에서 A가 더 작았습니다",
        "same": f"{result['pairs']}쌍 중 {result['ties']}쌍이 차이 {result['tolerance']:g} 안이었습니다",
    }[result["direction"]]
    coin = result["p_sign"]
    cls = {"supported": "good", "opposite": "bad"}.get(result["verdict"], "unclear")
    caution = ""
    if result["verdict"] in ("supported", "opposite") and coin >= 0.2:
        caution = (" 다만 이 확률이 작지 않아 우연히 이렇게 나왔을 수도 있습니다. 파리 수를 늘려 다시 해 보면 "
                   "더 확실해집니다.")
    return (
        f'<div class="verdict {cls}"><div class="big">{esc(VERDICT_TEXT[result["verdict"]])}</div>'
        f'<p>{esc(result["phase_label"])}의 <b>{esc(column)}</b>: '
        f'A 평균 <b>{fmt(a["mean"], protocol.metric)}</b> (범위 {fmt(a["min"], protocol.metric)} ~ {fmt(a["max"], protocol.metric)}, {a["n"]}마리) · '
        f'B 평균 <b>{fmt(b["mean"], protocol.metric)}</b> (범위 {fmt(b["min"], protocol.metric)} ~ {fmt(b["max"], protocol.metric)}, {b["n"]}마리) → '
        f'{words[result["direction"]]} (차이 {fmt(result["difference"], protocol.metric)}).</p>'
        f'<p>같은 번호의 A·B 파리는 뇌 배선과 출발 방향이 같은 쌍둥이입니다. {agree}. '
        f'A와 B가 사실 똑같다면, 동전을 {result["a_higher"] + result["a_lower"]}번 던져 이만큼 한쪽으로 몰릴 확률은 '
        f'약 {coin * 100:.0f}%입니다(작을수록 우연이 아닐 가능성이 큽니다).{caution}</p>'
        f'<p class="small">판정 기준: 평균 차이가 {result["tolerance"]:g}보다 크고, 쌍의 70% 이상이 같은 방향일 때 '
        f'"맞았다"고 적었습니다. 학생 수준에 맞춰 정한 기준이지 통계 검정이 아닙니다.</p></div>'
    )


def _limitations(protocol: Protocol, records: list[dict]) -> list[str]:
    out = [
        "실제 초파리가 아니라 시뮬레이션입니다. 몸과 다리는 실제 초파리를 본떴지만 뇌는 버섯체 학습과 몇 가지 "
        "타고난 반응 규칙만 있는 단순한 모델입니다.",
        f"파리 수가 조건마다 {protocol.flies}마리입니다. 실제 실험은 보통 한 번에 수십~백여 마리를 씁니다.",
        "배고픔은 보기 좋게 빨리 변하도록 압축했습니다(0에서 1까지 30분). 쉬는 시간은 시뮬레이션하지 않고 "
        "기억이 줄어드는 계산만 합니다.",
        "냄새는 벽을 통과해 퍼지고, 설탕 방울은 보이도록 실제보다 훨씬 크게 그렸습니다.",
        "장소를 기억하는 뇌 영역(중심복합체)은 없습니다. 냄새와 바닥 색의 좋고 싫음만 배웁니다.",
    ]
    stopped = sum(r["stopped"] for r in records)
    if stopped:
        out.append(f"중간에 멈춘 시행이 {stopped}번 있습니다. 그 시행은 정해진 시간보다 짧습니다.")
    return out


STYLE = """
:root { --ink:#1d1d1f; --dim:#5f6368; --line:#d9d9d9; --paper:#ffffff; --soft:#f5f5f2; }
* { box-sizing:border-box; }
body { margin:0; background:#e9e9e6; color:var(--ink); font:15px/1.65 "Malgun Gothic","Apple SD Gothic Neo","Noto Sans KR",sans-serif; }
.toolbar { position:sticky; top:0; z-index:5; display:flex; gap:8px; align-items:center; flex-wrap:wrap;
  padding:8px 16px; background:#2b2b2e; color:#eee; font-size:13px; }
.toolbar button { font:inherit; padding:5px 12px; border-radius:6px; border:1px solid #555; background:#3a3a3e; color:#fff; cursor:pointer; }
.toolbar .note { color:#bbb; margin-left:auto; }
.paper { max-width:860px; margin:18px auto 60px; background:var(--paper); padding:34px 44px; box-shadow:0 1px 6px rgba(0,0,0,.12); }
h1 { font-size:26px; margin:0 0 4px; line-height:1.35; }
h2 { font-size:19px; margin:30px 0 8px; padding-bottom:4px; border-bottom:2px solid var(--ink); }
h3 { font-size:16px; margin:18px 0 6px; }
.sub { color:var(--dim); font-size:13px; }
.running { background:#fff4d6; border:1px solid #e3c36b; padding:8px 12px; border-radius:6px; margin:12px 0; font-size:14px; }
table { border-collapse:collapse; width:100%; margin:6px 0; font-size:13.5px; }
th, td { border:1px solid var(--line); padding:5px 8px; text-align:left; vertical-align:top; }
th { background:var(--soft); font-weight:600; white-space:nowrap; }
td.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
.write { border:1.5px dashed #9aa6b2; border-radius:6px; padding:8px 10px; margin:6px 0 10px; background:#fbfcfe; }
.write:empty::before { content:attr(data-ph); color:#8a94a0; }
.write:focus { outline:2px solid #6f9bd1; background:#fff; }
.verdict { border-radius:8px; padding:12px 16px; margin:10px 0; border:1px solid; }
.verdict.good { background:#eef8ef; border-color:#86c28d; }
.verdict.bad { background:#fdeeee; border-color:#e19a9a; }
.verdict.unclear { background:#f4f4f1; border-color:#c9c9c2; }
.verdict .big { font-size:19px; font-weight:700; margin-bottom:4px; }
.small { font-size:12.5px; color:var(--dim); }
figure { margin:10px 0 16px; }
figcaption { font-size:13px; color:var(--dim); margin-top:2px; }
.chart { width:100%; height:auto; display:block; }
.chart .tick { font-size:11px; fill:#555; }
.chart .axis { font-size:12.5px; fill:#222; }
.rooms { display:flex; flex-wrap:wrap; gap:12px; }
.rooms figure { margin:0; width:210px; }
.room { width:210px; height:210px; display:block; }
.paths { display:grid; grid-template-columns:repeat(auto-fill, minmax(150px, 1fr)); gap:8px; }
.paths figure { margin:0; }
.paths .room { width:100%; height:auto; }
.legend span { display:inline-flex; align-items:center; gap:5px; margin-right:14px; font-size:13px; }
.legend i { width:14px; height:4px; display:inline-block; }
.pairs { display:grid; grid-template-columns:420px 1fr; gap:14px; align-items:center; }
details { margin:8px 0; }
details table { font-size:12px; }
ul.tight { margin:4px 0 8px; padding-left:22px; }
@media (max-width:760px) { .paper { padding:20px 16px; } .pairs { grid-template-columns:1fr; } }
@media print {
  body { background:#fff; font-size:12.5px; }
  .toolbar, .noprint { display:none !important; }
  .paper { box-shadow:none; margin:0; max-width:none; padding:0; }
  .write { border:1px solid #bbb; background:#fff; }
  .write:empty::before { content:""; }
  h2 { break-after:avoid; }
  figure, table, .verdict { break-inside:avoid; }
}
"""

SCRIPT = """
(function () {
  var id = document.body.getAttribute('data-exp');
  var boxes = document.querySelectorAll('.write');
  function load() {
    try {
      var saved = JSON.parse(localStorage.getItem('report:' + id) || '{}');
      boxes.forEach(function (b) { if (saved[b.dataset.key] && !b.textContent.trim()) b.innerHTML = saved[b.dataset.key]; });
    } catch (e) {}
  }
  function save() {
    try {
      var out = {};
      boxes.forEach(function (b) { if (b.textContent.trim()) out[b.dataset.key] = b.innerHTML; });
      localStorage.setItem('report:' + id, JSON.stringify(out));
    } catch (e) {}
  }
  load();
  boxes.forEach(function (b) { b.addEventListener('input', save); });
  var print = document.getElementById('b_print');
  if (print) print.onclick = function () { window.print(); };
  var dl = document.getElementById('b_save');
  if (dl) dl.onclick = function () {
    var copy = document.documentElement.cloneNode(true);
    var bar = copy.querySelector('.toolbar'); if (bar) bar.remove();
    var blob = new Blob(['<!doctype html>' + copy.outerHTML], {type: 'text/html;charset=utf-8'});
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = document.title + '.html';
    document.body.appendChild(a); a.click(); a.remove();
  };
})();
"""


def build_report(directory: Path) -> str:
    directory = Path(directory)
    protocol = Protocol.from_dict(read_json(directory / "protocol.json"))
    records = read_records(directory)
    status = read_json(directory / "status.json", {}) or {}
    s = EXPERIMENT_SETS.get(protocol.set_key)
    exp_id = directory.name
    title = s.question if s else protocol.title
    parts: list[str] = []
    add = parts.append

    add(f'<!doctype html><html lang="ko"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>탐구보고서 {esc(protocol.title)}</title><style>{STYLE}</style></head>'
        f'<body data-exp="{esc(exp_id)}">')
    add('<div class="toolbar"><button id="b_print">인쇄 / PDF로 저장</button>'
        '<button id="b_save">쓴 내용까지 HTML로 저장</button>'
        '<span class="note">점선 칸에 직접 쓰세요. 이 컴퓨터의 브라우저에 자동으로 남습니다.</span></div>')
    add('<div class="paper">')
    add(f'<div class="sub">초파리 시뮬레이션 A/B 실험 보고서 · 실험 번호 {esc(exp_id)} · 설계 {esc(protocol.created)}</div>')
    add(f'<h1 contenteditable="true">{esc(title)}</h1>')
    total = protocol.total_trials()
    state = status.get("state", "")
    if len(records) < total:
        verb = {"running": "진행 중", "reporting": "정리 중"}.get(state, "멈춤")
        add(f'<div class="running">실험이 {verb}입니다: 시행 {len(records)} / {total}개가 기록됐습니다. '
            f'끝난 시행까지만 반영했으니, 다 끝난 뒤 다시 열면 갱신됩니다.</div>')

    # 1. question and motivation
    add('<h2>1. 탐구 문제와 동기</h2>')
    add(f'<p><b>탐구 문제</b>: {esc(s.question if s else protocol.title)}</p>')
    add(_write_box("motive", "탐구 동기: 나는 ___을(를) 보고 ___이(가) 궁금해졌다. (예: 과일 껍질에 초파리가 모이는 것을 보고…)", 3))

    # 2. hypothesis
    add('<h2>2. 가설</h2>')
    if s:
        add(f'<p><b>실험 전에 고른 예상</b>: {esc(_prediction_sentence(s, protocol))}</p>')
    add(_write_box("reason", "이렇게 예상한 이유: ___ 때문에 ___할 것이라고 생각했다.", 3))

    # 3. variables
    if s:
        add('<h2>3. 변인</h2><table>')
        add(f'<tr><th>조작 변인 (바꾼 것)</th><td>{esc(s.changed)}: <b style="color:{A_COLOUR}">A</b> {esc(s.a)} / '
            f'<b style="color:{B_COLOUR}">B</b> {esc(s.b)}</td></tr>')
        metric = METRICS_BY_KEY[protocol.metric]
        also = "".join(f"<li>{esc(METRICS_BY_KEY[k].column)}: {esc(METRICS_BY_KEY[k].meaning)}</li>" for k in s.also)
        add(f'<tr><th>종속 변인 (잰 것)</th><td><b>{esc(metric.column)}</b>: {esc(metric.meaning)} '
            f'({esc(PHASE_LABELS.get(protocol.phase, protocol.phase))}의 평균)'
            f'{"<ul class=tight>" + also + "</ul>" if also else ""}</td></tr>')
        held = controlled_variables(protocol)
        add(f'<tr><th>통제 변인 (같게 한 것)</th><td><ul class="tight">{"".join(f"<li>{esc(h)}</li>" for h in held)}</ul></td></tr>')
        add('</table>')

    # 4. method
    add('<h2>4. 실험 방법</h2>')
    add('<h3>도구</h3><p>컴퓨터 시뮬레이션입니다. 몸은 실제 초파리의 3차원 몸을 본뜬 NeuroMechFly v2(Wang-Chen 등 2024, '
        'Nature Methods)이고, 물리 계산은 MuJoCo가 합니다. 뇌는 냄새와 바닥 색의 좋고 싫음을 배우는 버섯체 모델과 몇 가지 '
        '타고난 반응 규칙입니다. 100 × 100 mm 방에서 초파리 한 마리씩 실험했습니다.</p>')
    add('<p class="small">알림: 이 보고서의 표, 그래프, 판정 문장은 프로그램이 기록에서 자동으로 만들었습니다. '
        '점선 칸은 학생이 직접 썼습니다.</p>')
    add('<h3>실험 순서</h3><table>')
    for c, cond in enumerate(protocol.conditions):
        colour = A_COLOUR if c == 0 else B_COLOUR
        extra = cond.changes()
        who = "같은 파리가 모든 시행을 이어서 함" if cond.same_fly else "시행마다 새 파리"
        add(f'<tr><th style="color:{colour}">{esc(cond.label)}</th><td>{esc(describe_steps(cond))}'
            f'<br><span class="small">{esc(who)} · 시작 배고픔 {cond.hunger:.2f}'
            f'{" · " + esc(", ".join(extra)) if extra else ""}</span></td></tr>')
    add('</table>')
    add('<h3>방 배치</h3><div class="rooms">')
    seen_rooms = set()
    for r in records:
        key = (r["condition"], r["room"])
        if key in seen_rooms:
            continue
        seen_rooms.add(key)
        name = protocol.conditions[r["condition"]].label.split(":")[0]
        add(f'<figure>{svg_room(r["items"])}<figcaption>{esc(name)} · {esc(r["room_label"])}</figcaption></figure>')
    add('</div><p class="small">파랑·초록 네모: 색 바닥 · 빨간 점선: 전기 구역 · 흰 원: 설탕 · 색 점과 점선 원: '
        f'냄새가 나오는 곳과 그 {NEAR_RADIUS:.0f} mm 안 · 회색: 벽 · 가운데 작은 원: 출발점</p>')
    if s:
        add(f'<h3>실제 연구에서는</h3><p>{esc(s.basis)}</p>')
        add(f'<h3>왜 그렇게 예상할 수 있나 (모델의 작동 방식)</h3><p>{esc(s.why)}</p>')

    # 5. results
    add('<h2>5. 결과</h2>')
    if not records:
        add('<p>아직 끝난 시행이 없습니다.</p>')
    elif s:
        result = evaluate(protocol, records, protocol.prediction or protocol.expect)
        add('<h3>가설 판정</h3>')
        add(_verdict_block(s, protocol, result))
        metric = METRICS_BY_KEY[protocol.metric]
        if result["a"] and result["b"]:
            add('<div class="pairs">')
            add(f'<figure>{svg_pair_dots(result["scores"]["a"], result["scores"]["b"], metric.column)}'
                f'<figcaption>파리마다 점 하나({esc(result["phase_label"])}의 평균), 굵은 가로선이 평균, 회색 선이 같은 번호 쌍입니다.</figcaption></figure>')
            add('<div class="small">회색 선이 대부분 같은 방향으로 기울어 있으면 A와 B의 차이가 파리마다 꾸준하다는 뜻입니다. '
                '선이 엇갈리면 개체 차이가 커서 파리 수를 늘려 봐야 합니다.</div></div>')
        add(_heat_section(protocol, records))
        keys = [protocol.metric, *s.also]
        for key in keys:
            m = METRICS_BY_KEY[key]
            if m.text:
                continue
            series, n_trials, roles = _series(protocol, records, key)
            if n_trials < 2 and key != protocol.metric:
                continue
            if all(v is None for se in series for v in se["mean"]):
                continue
            fixed = (-1.05, 1.05) if key.endswith("_pi") else None
            add(f'<figure>{svg_trials_chart(series, n_trials, roles=roles, y_label=m.column, fixed=fixed)}'
                f'<figcaption><span class="legend"><span><i style="background:{A_COLOUR}"></i>A 평균</span>'
                f'<span><i style="background:{B_COLOUR}"></i>B 평균</span></span>'
                f'{esc(m.column)}: {esc(m.meaning)} 흐린 선은 파리 한 마리씩, 옅은 띠는 가장 작은 값~가장 큰 값입니다.'
                f'{" 끝까지 일어나지 않은 시행은 시행 시간으로 쳤습니다." if m.latency else ""}</figcaption></figure>')
        add('<h3>시행마다 평균 (범위)</h3>')
        add(_mean_table(protocol, records, keys))
        add('<h3>지나간 길</h3><p class="small">1번과 2번 파리가 A와 B에서 지나간 길입니다. 가운데 원에서 출발합니다.</p>')
        add(_path_grid(protocol, records))
        add('<details><summary>모든 시행의 원래 기록 보기</summary>')
        add(_raw_table(protocol, records))
        add('</details>')

    # 6. conclusion
    add('<h2>6. 결론</h2>')
    add(_write_box("conclusion", "가설이 맞았나요? 판정의 어떤 숫자가 그 근거인가요?", 3))
    add(_write_box("why", "왜 그런 결과가 나왔을까요? (위 '모델의 작동 방식'과 그래프를 함께 보세요)", 3))
    add(_write_box("next", "예상과 달랐다면 무엇 때문일까요? 맞았다면 새로 궁금해진 것은 무엇인가요?", 3))

    # 7. limits
    add('<h2>7. 한계와 오차</h2><ul>')
    for line in _limitations(protocol, records):
        add(f'<li>{esc(line)}</li>')
    add('</ul>')
    add(_write_box("limits", "내가 생각한 오차나 아쉬운 점, 다음에 바꿔 보고 싶은 것", 2))

    # 8. log and references
    add('<h2>8. 실험 기록과 참고 문헌</h2>')
    add(f'<p class="small">설계 {esc(protocol.created)} · 상태 {esc(state)} · 마지막 갱신 {esc(status.get("updated", ""))} · '
        f'기록된 시행 {len(records)}개 · 시뮬레이션한 시간 {duration_ko(sum(r["seconds"] for r in records))} · '
        f'계산에 걸린 시간 {duration_ko(sum(r.get("wall_s", 0.0) for r in records))} (여러 코어 합계)</p>')
    add('<ul>')
    if s:
        add(f'<li>{esc(s.basis)}</li>')
    add('<li>Wang-Chen S. 등 (2024). NeuroMechFly v2: simulating embodied sensorimotor control in adult Drosophila. '
        'Nature Methods 21, 2353–2362.</li></ul>')
    add(_write_box("refs", "내가 더 찾아본 자료 (책, 기사, 논문, 웹 페이지 주소)", 2))
    add(f'</div><script>{SCRIPT}</script></body></html>')
    return "".join(parts)


def _heat_section(protocol: Protocol, records: list[dict]) -> str:
    """Where A's flies and B's flies spent the trials the verdict is judged on,
    one map each over the same colour scale."""
    from flyplay.experiment import phase_records

    chosen: dict[int, list[dict]] = {}
    by_fly: dict[tuple[int, int], list[dict]] = {}
    for r in records:
        by_fly.setdefault((r["condition"], r["fly"]), []).append(r)
    for (c, _), rs in by_fly.items():
        rs.sort(key=lambda r: r["seq"])
        chosen.setdefault(c, []).extend(phase_records(rs, protocol.phase))
    if not chosen:
        return ""
    counts = {c: heat_counts(rs) for c, rs in chosen.items()}
    # Per trial, so a condition with more trials does not look hotter.
    for c in counts:
        counts[c] = counts[c] / max(1, len(chosen[c]))
    peak = max(float(m.max()) for m in counts.values())
    out = ['<h3>머문 곳 (히트맵)</h3><div class="rooms">']
    for c in sorted(counts):
        room = chosen[c][0]
        label = protocol.conditions[c].label
        out.append(f'<figure style="width:260px">{svg_heat(room["items"], counts[c], peak)}'
                   f'<figcaption>{esc(label)} · {esc(room["room_label"])} · 시행 {len(chosen[c])}개</figcaption></figure>')
    out.append('</div><p class="small">파리들이 판정에 쓴 시행('
               f'{esc(PHASE_LABELS.get(protocol.phase, protocol.phase))})에서 오래 머문 곳일수록 밝게 칠했습니다. '
               'A와 B는 같은 색 눈금이라, 같은 색이면 한 시행에 머문 시간도 같습니다. '
               '옅은 노랑 → 주황 → 빨강 → 검붉은 색 순으로 오래 머문 곳이고, 잠깐 지나간 칸은 칠하지 않았습니다.</p>')
    return "".join(out)


def _mean_table(protocol: Protocol, records: list[dict], keys: list[str]) -> str:
    keys = [k for k in keys if not METRICS_BY_KEY[k].text]
    rows = ["<table><tr><th>조건</th><th>시행</th><th>단계</th>" +
            "".join(f"<th>{esc(METRICS_BY_KEY[k].column)}</th>" for k in keys) + "</tr>"]
    for c, cond in enumerate(protocol.conditions):
        by_seq: dict[int, list[dict]] = {}
        for r in records:
            if r["condition"] == c:
                by_seq.setdefault(r["seq"], []).append(r)
        for seq in sorted(by_seq):
            rs = by_seq[seq]
            cells = []
            for k in keys:
                vals = [v for v in (metric_value(r, k) for r in rs) if v is not None]
                if not vals:
                    cells.append('<td class="num">-</td>')
                    continue
                cells.append(f'<td class="num">{fmt(float(np.mean(vals)), k)} '
                             f'<span class="small">({fmt(min(vals), k)}~{fmt(max(vals), k)}, n={len(vals)})</span></td>')
            role = ROLE_LABELS.get(rs[0]["role"], rs[0]["role"])
            rows.append(f'<tr><td>{esc(cond.label)}</td><td class="num">{seq}</td><td>{esc(role)}</td>{"".join(cells)}</tr>')
    rows.append("</table>")
    return "".join(rows)


def _path_grid(protocol: Protocol, records: list[dict]) -> str:
    out = ['<div class="paths">']
    for fly in (1, 2):
        for c, cond in enumerate(protocol.conditions):
            for r in [r for r in records if r["condition"] == c and r["fly"] == fly]:
                colour = A_COLOUR if c == 0 else B_COLOUR
                role = ROLE_LABELS.get(r["role"], r["role"])
                out.append(f'<figure>{svg_room(r["items"], r["path"], 150, colour)}'
                           f'<figcaption>{esc(cond.label.split(":")[0])} 파리 {fly} · {r["seq"]}번째 ({esc(role)})</figcaption></figure>')
    out.append("</div>")
    return "".join(out)


def _raw_table(protocol: Protocol, records: list[dict]) -> str:
    rows = csv_rows(protocol, records)
    if not rows:
        return ""
    keys = [k for k in rows[0] if k not in ("실험", "다시 보기 파일")]
    out = ["<table><tr>" + "".join(f"<th>{esc(k)}</th>" for k in keys) + "</tr>"]
    for row in rows:
        out.append("<tr>" + "".join(f"<td>{esc(row[k])}</td>" for k in keys) + "</tr>")
    out.append("</table>")
    return "".join(out)

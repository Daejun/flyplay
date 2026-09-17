"""Keep, list and inspect mushroom-body memories.

A memory is one fly's KC->MBON synapses plus what is needed to put them back
into a fly: the Kenyon-cell wiring seed and the fly's configuration. The runner
already snapshots every fly at every block boundary, but into the sweep's
output directory, which the next run of the same sweep clears. This keeps the
ones worth keeping under memories/, by name, where nothing overwrites them.

    python scripts/16_memory.py list
    python scripts/16_memory.py show reversed-extinction
    python scripts/16_memory.py save my-fly --from out/mb_extinction/memory/extinction_seed03/after_train.npz --note "..."
    python scripts/16_memory.py curate

`curate` files a starter set, one fly each, chosen so that no two behave
alike. From the extinction sweep (seed 0):

    naive                 nothing learned; the reference
    a-punished            straight after odour A was punished
    reversed-passive      after reversal to B with no extinction: A still feared
    reversed-extinction   after reversal with extinction: A's fear cancelled
    fear-generalized      extinction wired to approach: B feared more than A

and from the colour sweep (Vogt et al. 2014 protocol, straight after training):

    blue-punished         blue floor paired with punishment
    green-punished        green floor paired with punishment
    blue-rewarded         blue floor paired with sugar
    gd-blocked            trained on blue with visual Kenyon cells blocked:
                          nothing learned, the control

Colour valences are listed in units of full depression of a colour's code
(-1..+1), odour valences in MBON units. Watch one in the browser:

    python scripts/09_web_viewer.py --memory reversed-extinction --port 8010
    python scripts/09_web_viewer.py --memory blue-punished --port 8010

Or restore one in code with `flyplay.conditioning.experiment_from_memory`.
"""

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from flyplay.conditioning import (
    ConditioningExperiment,
    config_from_dict,
    experiment_from_memory,
    read_memory,
)
from flyplay.olfactory import OlfactoryFrontEnd
from flyplay.visual_pathway import VisualFrontEnd

LIBRARY = _bootstrap.MEMORIES
SWEEP = _bootstrap.OUT / "mb_extinction" / "memory"
COLOUR_SWEEP = _bootstrap.OUT / "mb_colour" / "memory"
#: Full depression of a colour's visual code, in MBON units (see
#: flyplay.mushroom_body): what a colour valence is divided by for listing.
VISUAL_SCALE = VisualFrontEnd().k / OlfactoryFrontEnd(2).k

#: name -> (snapshot relative to its sweep's memory dir, note). Notes are for the
#: person loading the memory -- the viewer shows them -- so they say what the fly
#: will do.
CURATED = {
    "a-punished": (
        "differential_seed00/after_train.npz",
        "냄새 A에 벌을 받은 직후입니다. A를 피하고 B 쪽으로 갑니다.",
    ),
    "reversed-passive": (
        "differential_seed00/after_reversal_train.npz",
        "규칙을 B로 뒤집었지만 소거가 없는 초파리입니다. 새로 벌받은 B를 피하지만 "
        "A에 대한 옛 기억도 남아 있어서 두 냄새가 다 조금씩 싫습니다.",
    ),
    "reversed-extinction": (
        "extinction_seed00/after_reversal_train.npz",
        "규칙을 B로 뒤집고 소거가 있는 초파리입니다. A의 혐오 기억은 지워지지 않고 "
        "맞서는 기억으로 상쇄됐습니다(접근·회피 구획이 둘 다 눌림). B만 싫습니다.",
    ),
    "fear-generalized": (
        "extinction_wrong_seed00/after_train.npz",
        "소거 신호가 접근 구획으로 잘못 연결된 초파리입니다. 한 번도 벌받지 않은 "
        "B가 벌받은 A보다 더 무서워졌습니다.",
    ),
}
#: The same, from the colour sweep. Even seeds were trained with blue as the
#: CS+, odd seeds with green.
CURATED_COLOUR = {
    "blue-punished": (
        "colour_seed00/after_train.npz",
        "파랑 바닥에 벌(도파민)을 받은 초파리입니다(Vogt 2014 프로토콜: 파랑·초록 60초씩 4번). "
        "체커보드에서 파랑 칸을 피하고 초록 칸 위에 머뭅니다. 냄새는 배운 적이 없습니다.",
    ),
    "green-punished": (
        "colour_seed01/after_train.npz",
        "초록 바닥에 벌을 받은 초파리입니다. blue-punished의 거울상으로, 초록 칸을 피합니다. "
        "두 기억을 함께 봐야 색 자체를 배운 것인지 알 수 있습니다(Vogt의 상호 집단).",
    ),
    "blue-rewarded": (
        "colour_reward_seed00/after_train.npz",
        "파랑 바닥에 설탕(보상 도파민)을 받은 초파리입니다. 보상은 회피 구획을 눌러 파랑의 가치를 "
        "+ 쪽으로 올리므로, 체커보드에서 파랑 칸을 찾아갑니다.",
    ),
    "gd-blocked": (
        "colour_gd_blocked_seed00/after_train.npz",
        "시각 케니언 세포(γd) 출력을 막은 채 파랑에 벌을 받은 초파리입니다. 도파민은 들어갔지만 "
        "시냅스에 닿지 않아 아무것도 배우지 못했고, 색을 가리지 않고 걷습니다(Vogt 2016의 대조군)."
    ),
}


def display_valences(info: dict) -> dict[str, float]:
    """Filed valence per stimulus, colour ones in units of full depression."""
    config = info.get("config") or {}
    scale = VISUAL_SCALE if config.get("modality") == "colour" else 1.0
    return {k: s["valence"] / scale for k, s in (info.get("summary") or {}).items()}


def library_path(name: str) -> Path:
    if not name or any(c in name for c in '\\/:*?"<>|'):
        raise SystemExit(f"not a usable memory name: {name!r}")
    return LIBRARY / f"{name}.npz"


def sweep_record_for(snapshot: Path) -> dict | None:
    """The trial-stream JSON a runner snapshot belongs to, if it is where
    14_run_conditioning.py puts it: <sweep>/memory/<fly>/after_<block>.npz next
    to <sweep>/<fly>.json. Snapshots written before memories carried their own
    config need it from there."""
    record = snapshot.parent.parent.parent / f"{snapshot.parent.name}.json"
    return json.loads(record.read_text(encoding="utf-8")) if record.exists() else None


def file_memory(name: str, source: Path, note: str, force: bool) -> Path:
    """Copy a snapshot into the library under `name`, with config and summary."""
    target = library_path(name)
    if target.exists() and not force:
        raise SystemExit(f"{target} exists; pass --force to replace it")
    if not source.exists():
        raise SystemExit(f"no such snapshot: {source}")
    info = read_memory(source)
    config = None
    if info["config"] is None:
        record = sweep_record_for(source)
        if record is None:
            raise SystemExit(
                f"{source} carries no config and no sweep record was found next "
                f"to it, so the fly it came from cannot be rebuilt faithfully."
            )
        config = config_from_dict(record["config"])
    experiment = experiment_from_memory(source, config=config)
    try:
        LIBRARY.mkdir(parents=True, exist_ok=True)
        experiment.save_memory(target, name=name, note=note, source=str(source))
    finally:
        experiment.close()
    return target


def cmd_list(_args) -> None:
    paths = sorted(LIBRARY.glob("*.npz"))
    if not paths:
        print(f"no memories in {LIBRARY}. Try: python scripts/16_memory.py curate")
        return
    print(f"{'name':<22} {'task':<6} {'trials':>6}  {'valence':<26} "
          f"{'extinction':>10}  note")
    print("-" * 110)
    for path in paths:
        info = read_memory(path)
        config = info.get("config") or {}
        values = "  ".join(f"{k} {v:+.3f}" for k, v in display_valences(info).items())
        print(f"{path.stem:<22} {config.get('modality', 'odour'):<6} {info['trials']:>6}  "
              f"{values:<26} {str(config.get('extinction')):>10}  "
              f"{info.get('note', '')[:34]}")


def cmd_show(args) -> None:
    path = library_path(args.name)
    if not path.exists():
        raise SystemExit(f"no memory named {args.name!r} in {LIBRARY}")
    info = read_memory(path)
    print(f"{args.name}  ({path})")
    print(f"  note     {info.get('note', '')}")
    print(f"  source   {info.get('source', '')}")
    print(f"  created  {info.get('created', '')}")
    config = info.get("config") or {}
    colour = config.get("modality") == "colour"
    scale = VISUAL_SCALE if colour else 1.0
    print(f"  wiring   seed {info['wiring_seed']}, {info['trials']} trials experienced, "
          f"{info['n_visual']} visual Kenyon cells")
    unit = " (units of full depression)" if colour else ""
    for stimulus, s in (info.get("summary") or {}).items():
        mbon = "  ".join(f"{k} {v / scale:.3f}" for k, v in s["mbon"].items())
        print(f"  {'colour' if colour else 'odour'} {stimulus}  valence "
              f"{s['valence'] / scale:+.3f}{unit}   {mbon}")
    keys = ("modality", "stimuli", "reinforcer", "blocked", "eta", "recovery",
            "differential", "extinction", "extinction_gain", "shared_dan", "frozen",
            "innate_valence", "visual_gain")
    # Memories filed before a key existed simply lack it; print what they have.
    print("  config   " + ", ".join(f"{k}={config[k]}" for k in keys if k in config))


def cmd_save(args) -> None:
    target = file_memory(args.name, Path(args.source), args.note, args.force)
    print(f"saved -> {target}")


def cmd_curate(args) -> None:
    LIBRARY.mkdir(parents=True, exist_ok=True)
    naive = library_path("naive")
    if not naive.exists() or args.force:
        experiment = ConditioningExperiment(seed=0)
        try:
            experiment.save_memory(
                naive, name="naive", source="fresh fly, wiring seed 0",
                note="아무것도 배우지 않은 초파리입니다. 두 냄새 모두 가치 0이고 "
                     "처음 잡은 냄새 쪽으로 갑니다. 다른 기억과 비교하는 기준입니다.",
            )
        finally:
            experiment.close()
        print(f"saved -> {naive}")
    starters = [(name, SWEEP / rel, note, "extinction") for name, (rel, note) in CURATED.items()]
    starters += [(name, COLOUR_SWEEP / rel, note, "colour --seeds 10")
                 for name, (rel, note) in CURATED_COLOUR.items()]
    for name, source, note, experiment in starters:
        if not source.exists():
            print(f"skip {name}: {source} not found "
                  f"(run 14_run_conditioning.py --experiment {experiment})")
            continue
        if library_path(name).exists() and not args.force:
            print(f"keep {name}: already filed (--force to replace)")
            continue
        print(f"saved -> {file_memory(name, source, note, force=True)}")


def main() -> None:
    # The console here is cp949. Hangul encodes, but one stray symbol in a note
    # raises UnicodeEncodeError and kills the process mid-listing; replace it.
    sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List filed memories.").set_defaults(fn=cmd_list)
    show = sub.add_parser("show", help="Details of one memory.")
    show.add_argument("name")
    show.set_defaults(fn=cmd_show)
    save = sub.add_parser("save", help="File a runner snapshot under a name.")
    save.add_argument("name")
    save.add_argument("--from", dest="source", required=True)
    save.add_argument("--note", default="")
    save.add_argument("--force", action="store_true")
    save.set_defaults(fn=cmd_save)
    curate = sub.add_parser("curate", help="File the starter set.")
    curate.add_argument("--force", action="store_true")
    curate.set_defaults(fn=cmd_curate)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

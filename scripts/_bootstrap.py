"""Put the project root on sys.path so `flyplay` imports from any cwd."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

#: Named mushroom-body memories that are meant to be kept. Deliberately outside
#: out/, which sweeps clear and rewrite: a memory filed here survives re-running
#: the experiment that produced it.
MEMORIES = ROOT / "memories"

"""A sandbox fly that outlives the viewer process.

The viewer restarts often -- new code, a crash, a reboot -- and every restart
used to be a new fly: what it had learned, how hungry it was, the room it was in
and the path it had walked were gone (four restarts in one afternoon). A
*session* is those things on disk. The viewer writes it every `SAVE_EVERY_S`
seconds and on a clean exit, and reads it back when it starts.

    out/sandbox/sessions/<name>/state.npz   synapses, room, hunger, clock, tallies
                                /trail.f32   the path, appended as it grows
                                /lock        held open by the viewer using it

**The state file is written whole and renamed into place,** so a viewer killed
mid-write leaves the previous state, never half of one. Windows refuses the
rename while a reader has the file open, hence the retries (the same failure
killed an experiment run, `flyplay.experiment.write_json`).

**The trail is appended, not rewritten.** A day of walking is 6.9 MB of float32
pairs; rewriting that every 30 s would put 20 GB a day on the disk for a few KB
of new points. It is rewritten only when it starts over or drops its oldest
hour. A viewer killed between the two files can leave the trail a few seconds
ahead of or behind the state; it is only drawn, so that is allowed.

**Two viewers on one session would overwrite each other's fly,** so the session
is locked by the process using it. The lock is an operating-system lock on an
open file, which the OS releases when the process ends however it ends: a pid
file would outlive a killed viewer and lock the session forever.

**Saved synapses carry their wiring's signature** (`MushroomBody.
wiring_signature`). Weights index Kenyon cells, and loaded onto a front end
that codes smells or colours differently they sit on the wrong cells and raise
nothing. A session from other wiring is set aside, not loaded.
"""

from __future__ import annotations

import json
import os
import struct
import time
from array import array
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Seconds between saves while the viewer runs. What a killed viewer can lose.
SAVE_EVERY_S = 30.0
#: Version of the state layout. Bump when a saved field changes meaning.
STATE_FORMAT = 1
#: Trail file header: format, points dropped before the first stored one.
_TRAIL_HEADER = struct.Struct("<qq")
_TRAIL_FORMAT = 1


class SessionLocked(RuntimeError):
    """Another process has this session open."""


class SessionLock:
    """An exclusive lock on one session directory, held until `release` or exit."""

    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "lock"
        self._file = open(self.path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt

                self._file.seek(0)
                msvcrt.locking(self._file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self._file.close()
            raise SessionLocked(f"session {directory.name!r} is open in another process") from error

    def release(self) -> None:
        if self._file.closed:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._file.seek(0)
                msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()


def _replace(tmp: Path, path: Path) -> None:
    for attempt in range(40):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.025)


@dataclass
class SavedState:
    """What `Session.load` found: the JSON part and the arrays."""

    meta: dict
    arrays: dict[str, np.ndarray]
    #: The trail as stored: float32 x/y pairs, and points dropped before them.
    trail: array
    trail_dropped: int


class Session:
    """One named session's files. See the module docstring.

    Args:
        root: Directory holding every session, usually ``out/sandbox/sessions``.
        name: This session's name; letters, digits, ``-`` and ``_``.
    """

    def __init__(self, root: Path, name: str):
        if not name or not all(ch.isalnum() or ch in "-_" for ch in name):
            raise ValueError(f"session names use letters, digits, '-' and '_'; got {name!r}")
        self.name = name
        self.directory = Path(root) / name
        self.state_path = self.directory / "state.npz"
        self.trail_path = self.directory / "trail.f32"
        self._lock: SessionLock | None = None
        #: (trail epoch, points written) of what `trail.f32` holds.
        self._trail_written: tuple[int, int] | None = None

    # --- ownership -------------------------------------------------------------

    def acquire(self) -> None:
        self._lock = SessionLock(self.directory)

    def release(self) -> None:
        if self._lock is not None:
            self._lock.release()
            self._lock = None

    # --- reading ---------------------------------------------------------------

    def exists(self) -> bool:
        return self.state_path.exists()

    def load(self) -> SavedState | None:
        """The saved state, or None when there is none. Raises ValueError for a
        file this version cannot read."""
        if not self.state_path.exists():
            return None
        with np.load(self.state_path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files if k != "meta"}
            meta = json.loads(bytes(data["meta"]).decode("utf-8"))
        if meta.get("format") != STATE_FORMAT:
            raise ValueError(f"session state format {meta.get('format')}, this version reads {STATE_FORMAT}")
        trail, dropped = array("f"), 0
        try:
            raw = self.trail_path.read_bytes()
            fmt, dropped = _TRAIL_HEADER.unpack_from(raw)
            if fmt == _TRAIL_FORMAT:
                body = raw[_TRAIL_HEADER.size:]
                trail.frombytes(body[: len(body) // 8 * 8])
            else:
                dropped = 0
        except (OSError, struct.error):
            pass
        return SavedState(meta, arrays, trail, int(dropped))

    def set_aside(self, reason: str) -> Path | None:
        """Move the saved state out of the way, kept beside it under a dated
        name, so the next save starts clean without destroying anything."""
        if not self.state_path.exists():
            return None
        stamp = time.strftime("%Y%m%d_%H%M%S")
        target = self.directory / f"state_until_{stamp}_{reason}.npz"
        _replace(self.state_path, target)
        if self.trail_path.exists():
            _replace(self.trail_path, self.directory / f"trail_until_{stamp}_{reason}.f32")
        self._trail_written = None
        return target

    # --- writing ---------------------------------------------------------------

    def write(self, meta: dict, arrays: dict[str, np.ndarray], trail) -> None:
        """Write one snapshot: the state whole, the trail by appending.

        `trail` is the live `flyplay.sandbox.Trail`; only its new points are read.
        Call from one thread at a time.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_name("state.npz.tmp")
        payload = dict(arrays)
        payload["meta"] = np.frombuffer(
            json.dumps({**meta, "format": STATE_FORMAT}, ensure_ascii=False).encode("utf-8"), dtype=np.uint8)
        with open(tmp, "wb") as f:
            np.savez(f, **payload)
        _replace(tmp, self.state_path)
        self._write_trail(trail)

    def _write_trail(self, trail) -> None:
        written = self._trail_written
        if written is not None and written[0] == trail.epoch:
            epoch, start, xy = trail.since(written[1])
            if epoch == written[0] and start == written[1]:
                if len(xy):
                    with open(self.trail_path, "ab") as f:
                        f.write(xy.tobytes())
                self._trail_written = (epoch, start + len(xy) // 2)
                return
        # Started over, dropped its oldest hour, or never written by this
        # process: the whole trail, renamed into place.
        epoch, start, xy = trail.since(0)
        tmp = self.trail_path.with_name("trail.f32.tmp")
        with open(tmp, "wb") as f:
            f.write(_TRAIL_HEADER.pack(_TRAIL_FORMAT, start))
            f.write(xy.tobytes())
        _replace(tmp, self.trail_path)
        self._trail_written = (epoch, start + len(xy) // 2)

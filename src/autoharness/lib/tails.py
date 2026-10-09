"""Unreflected tails for a host without a guaranteed session-close flush (Codex).

One JSON note per session under `<state>/tails/`, rewritten on every Stop that leaves activity
behind and settled by the reflection that consumed it. A later pass (the engine's nightly run) reads
`pending()` and reflects the leftovers from `offset` on. `gap()` records a window that could not be
captured at all (no transcript path), `give_up()` one that a failing child never consumed, so a
coverage hole is a file, not silence. Claude flushes its tail on SessionEnd and never writes here.

Every write to a note runs under the session's file lock (lib.lock): a Stop and the settle of an
earlier reflection are separate processes, and a note written between settle's read and its unlink
would otherwise vanish. Reads take no lock: the note is replaced atomically, never torn.
"""
import datetime as dt
import json
import time

from autoharness.lib import atomic, counters, layer, lock


def _path(session_id, root=None):
    if not isinstance(session_id, str) or not counters._SAFE_SESSION.fullmatch(session_id):
        raise ValueError(f"unsafe session id: {session_id!r}")
    return layer.state_dir(layer.PROJECT, root) / "tails" / f"{session_id}.json"


def _locked(session_id, root=None):
    return lock.file_lock(_path(session_id, root).with_suffix(".json.lock"))


def _read(p):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic.write_text(p, json.dumps(data, ensure_ascii=False, indent=2))


def note(session_id, transcript_path, count, root=None, *, gap=None):
    p = _path(session_id, root)
    data = {"session_id": session_id, "transcript_path": transcript_path, "count": int(count),
            "offset": counters.session_offset(session_id, root),
            "updated": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "version": time.time_ns()}  # which note a reflection was launched from (settle)
    if gap:
        data["coverage_gap"] = gap
    with _locked(session_id, root):
        current = _read(p)
        if current and current.get("offset") == data["offset"] and current.get("failures"):
            data["failures"] = current["failures"]  # the same unconfirmed window: its streak goes on
        _write(p, data)
    return data


def gap(session_id, reason, count, root=None):
    return note(session_id, None, count, root, gap=reason)


def clear(session_id, root=None):
    p = _path(session_id, root)
    with _locked(session_id, root):
        p.unlink(missing_ok=True)


def read(session_id, root=None):
    return _read(_path(session_id, root))


def settle(session_id, version, offset, root=None):
    """A reflection launched from the note of `version` fed the window up to `offset`: drop that
    note, or, when a Stop replaced it meanwhile with newer activity, keep the newer one and move
    its offset to where this reflection stopped."""
    p = _path(session_id, root)
    with _locked(session_id, root):
        current = _read(p)
        if current is None:
            return
        if current.get("version") == version:
            p.unlink(missing_ok=True)
            return
        current["offset"] = int(offset)
        _write(p, current)


def fail(session_id, root=None):
    """One more child that did not consume this session's window; returns the streak."""
    p = _path(session_id, root)
    with _locked(session_id, root):
        current = _read(p)
        if current is None:
            return 0
        current["failures"] = current.get("failures", 0) + 1
        _write(p, current)
        return current["failures"]


def give_up(session_id, reason, offset, root=None):
    """The window is abandoned: the watermark moved past it, the note turns into a coverage gap
    starting at `offset`, and the streak restarts."""
    p = _path(session_id, root)
    with _locked(session_id, root):
        current = _read(p)
        if current is None:
            return
        current.update(coverage_gap=reason, failures=0, offset=int(offset))
        _write(p, current)


def pending(root=None):
    d = layer.state_dir(layer.PROJECT, root) / "tails"
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.json")):
        data = _read(p)
        if data is not None:
            out.append(data)  # a torn note is skipped, never fatal
    return out

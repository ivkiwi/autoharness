"""Unreflected tails for a host without a guaranteed session-close flush (Codex).

One JSON note per session under `<state>/tails/`, rewritten on every Stop that leaves activity
below the cadence and removed the moment a reflection fires or the session ends. A later pass (the
engine's nightly run) reads `pending()` and reflects the leftovers from `offset` on. `gap()` records a
window that could not be captured at all (no transcript path), so a coverage hole is a file, not
silence. Claude flushes its tail on SessionEnd and never writes here.
"""
import datetime as dt
import json
import time

from autoharness.lib import atomic, counters, layer


def _path(session_id, root=None):
    if not isinstance(session_id, str) or not counters._SAFE_SESSION.fullmatch(session_id):
        raise ValueError(f"unsafe session id: {session_id!r}")
    return layer.state_dir(layer.PROJECT, root) / "tails" / f"{session_id}.json"


def note(session_id, transcript_path, count, root=None, *, gap=None):
    p = _path(session_id, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {"session_id": session_id, "transcript_path": transcript_path, "count": int(count),
            "offset": counters.session_offset(session_id, root),
            "updated": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "version": time.time_ns()}  # which note a reflection was launched from (settle)
    if gap:
        data["coverage_gap"] = gap
    atomic.write_text(p, json.dumps(data, ensure_ascii=False, indent=2))
    return data


def gap(session_id, reason, count, root=None):
    return note(session_id, None, count, root, gap=reason)


def clear(session_id, root=None):
    _path(session_id, root).unlink(missing_ok=True)


def read(session_id, root=None):
    try:
        return json.loads(_path(session_id, root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def settle(session_id, version, offset, root=None):
    """A reflection launched from the note of `version` fed the window up to `offset`: drop that
    note, or, when a Stop replaced it meanwhile with newer activity, keep the newer one and move
    its offset to where this reflection stopped. shortcut: read-compare-write without a lock; a
    Stop landing inside that gap is re-noted at the next Stop."""
    current = read(session_id, root)
    if current is None:
        return
    if current.get("version") == version:
        clear(session_id, root)
        return
    current["offset"] = int(offset)
    atomic.write_text(_path(session_id, root), json.dumps(current, ensure_ascii=False, indent=2))


def pending(root=None):
    d = layer.state_dir(layer.PROJECT, root) / "tails"
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue  # a torn note is skipped, never fatal
    return out

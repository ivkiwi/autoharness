"""Unreflected tails for a host without a guaranteed session-close flush (Codex).

One JSON note per session under `<state>/tails/`, rewritten on every Stop that leaves activity
below the cadence and removed the moment a reflection fires or the session ends. A later pass (the
engine's nightly run) reads `pending()` and reflects the leftovers from `offset` on. `gap()` records a
window that could not be captured at all (no transcript path), so a coverage hole is a file, not
silence. Claude flushes its tail on SessionEnd and never writes here.
"""
import datetime as dt
import json

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
            "updated": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}
    if gap:
        data["coverage_gap"] = gap
    atomic.write_text(p, json.dumps(data, ensure_ascii=False, indent=2))
    return data


def gap(session_id, reason, count, root=None):
    return note(session_id, None, count, root, gap=reason)


def clear(session_id, root=None):
    _path(session_id, root).unlink(missing_ok=True)


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

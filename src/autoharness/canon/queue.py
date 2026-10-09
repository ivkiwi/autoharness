"""Shared proposal queue: append-only JSONL under GATE_DIR/queue, one writer at a time.

Each row carries where the intent came from (harness, session, transcript range, window hash, the
redaction rule version that cleaned it), so the gate can tell two events apart even when their
intents are identical, and can drop a replay of the same event after a crash between append and clear.
"""
import hashlib
import json
import os
import time

from autoharness import config
from autoharness.lib import lock

_PROVENANCE_KEYS = ("harness", "session_id", "transcript_path", "range", "window_sha256", "kind")


def _dir():
    return config.GATE_DIR / "queue"


def redaction_version():
    return hashlib.sha256(config.REDACTION_RULES.read_bytes()).hexdigest()[:16]


def event_id(provenance, intent):
    key = {"provenance": {k: provenance.get(k) for k in _PROVENANCE_KEYS}, "intent": intent}
    return hashlib.sha256(json.dumps(key, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _last_byte(p):
    with p.open("rb") as f:
        f.seek(-1, 2)
        return f.read(1)


def append(run_id, intents, verdicts, *, provenance=None, project_root=None):
    prov = {"harness": config.HARNESS, "redaction_version": redaction_version(), **(provenance or {})}
    d = _dir()
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{run_id}.jsonl"
    with lock.file_lock(d / "queue.lock"):  # every project and harness appends here
        torn = p.exists() and p.stat().st_size > 0 and _last_byte(p) != b"\n"
        with p.open("a", encoding="utf-8") as f:
            f.write("\n" if torn else "")
            for intent, verdict in zip(intents, verdicts, strict=True):
                f.write(json.dumps({"id": event_id(prov, intent), "at": time.time(), "run_id": run_id,
                                    "project_root": str(project_root) if project_root else None,
                                    "provenance": prov, "intent": intent, "verdict": verdict},
                                   ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())  # durable before the caller clears the run's intent file


def read():
    """Every readable row once, oldest first; a torn or foreign line is skipped, a replayed event dropped."""
    rows, seen = [], set()
    d = _dir()
    if not d.exists():
        return rows
    for p in sorted(d.glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8", errors="replace").split("\n"):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) or not isinstance(row.get("intent"), dict) or row.get("id") in seen:
                continue
            seen.add(row["id"])
            rows.append(row)
    return sorted(rows, key=lambda r: r.get("at", 0))

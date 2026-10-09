"""Append-only JSONL helpers for the gate's own records, and the digest notices built on them.

A notice is one line per published change or rollback, keyed by the event it reports; the Telegram
digest reads them, sends one line per batch, and remembers what it delivered.
"""
import json
import os
import time

from autoharness import config


def append(path, entry):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        f.seek(0, 2)
        torn = f.tell() > 0 and (f.seek(-1, 2), f.read(1))[1] != b"\n"
        f.write((b"\n" if torn else b"") + (json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())  # a torn tail must not swallow the next record


def lines(path):
    out = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").split("\n"):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                out.append(entry)
    return out


def notify(event_id, text):
    """One notice per event, however many times it is asked for."""
    p = config.GATE_DIR / "notices.jsonl"
    if event_id not in {e.get("id") for e in lines(p)}:
        append(p, {"id": event_id, "at": time.time(), "text": text})

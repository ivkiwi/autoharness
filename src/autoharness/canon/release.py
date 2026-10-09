"""Canon releases: an immutable tree per version under skill-releases/<name>/<sha>, and one entry per
skill in skills/ that harnesses discover. Publishing and rolling back both switch that entry.

A switch is one transaction under release.lock: settle any switch a crash left open, freeze both
ends as releases (so a rollback can only ever land on a hash-verified tree), compare-and-swap the
live entry against the expected state right before touching it, journal `prepared` with everything
recovery needs (including where a real canon directory will be moved), switch, read back, journal
`committed`. A real canon directory is moved to GATE_DIR/displaced, never deleted. Recovery only
touches an entry that is still in one of the states its own transaction can leave; anything else is
someone else's change and is reported as a conflict, not overwritten.
"""
import hashlib
import json
import os
import shutil
import stat
import time
from pathlib import Path

from autoharness import config
from autoharness.lib import lock


class Conflict(Exception):
    """The live entry is not in the state the switch was planned against."""


def skills_dir():
    return config.CANON_ROOT / "skills"


def releases_dir():
    return config.CANON_ROOT / "skill-releases"


def _journal():
    return config.GATE_DIR / "journal.jsonl"


def _lock():
    config.GATE_DIR.mkdir(parents=True, exist_ok=True)
    return lock.file_lock(config.GATE_DIR / "release.lock")


def tree_sha256(root):
    """Same definition as Ada's skill_evolution.tree_sha256, so hashes in her dossiers stay comparable."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink()):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def has_links(root):
    # the hash skips symlinks and a copy would follow them: a tree with any is out of scope, not guessed at
    return any(p.is_symlink() for p in root.rglob("*"))


def current(name):
    """(kind, target, sha) of the live entry: kind is absent, dir or link (a dangling link has sha None)."""
    entry = skills_dir() / name
    if entry.is_symlink():
        resolved = entry.resolve()
        return "link", os.readlink(entry), tree_sha256(resolved) if resolved.is_dir() else None
    if entry.is_dir():
        return "dir", None, tree_sha256(entry)
    if entry.exists():
        raise Conflict(f"{entry} is neither a skill directory nor a release link")
    return "absent", None, None


def journal_entries():
    p = _journal()
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").split("\n"):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict) and isinstance(entry.get("op"), str):
            out.append(entry)
    return out


def _record(entry):
    p = _journal()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a+b") as f:
        f.seek(0, 2)
        if f.tell():
            f.seek(-1, 2)
            torn = f.read(1) != b"\n"
        else:
            torn = False
        line = json.dumps({**entry, "at": time.time()}, ensure_ascii=False) + "\n"
        f.write((b"\n" if torn else b"") + line.encode("utf-8"))  # a torn tail must not swallow this record
        f.flush()
        os.fsync(f.fileno())


def _read_only(root):
    for path in [*root.rglob("*"), root]:
        path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _writable(root):
    return any(p.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) for p in [root, *root.rglob("*")])


def freeze(name, tree):
    """skill-releases/<name>/<sha>: copied once, read-only before it gets its final name, re-verified."""
    if has_links(tree):
        raise Conflict(f"{tree} contains symlinks")
    sha = tree_sha256(tree)
    dest = releases_dir() / name / sha
    if not dest.exists():
        tmp = dest.with_name(f".{sha}.{os.getpid()}.tmp")
        if tmp.exists():
            for p in [tmp, *tmp.rglob("*")]:
                p.chmod(p.stat().st_mode | stat.S_IWUSR)
            shutil.rmtree(tmp)
        shutil.copytree(tree, tmp)
        _read_only(tmp)
        os.rename(tmp, dest)
    if has_links(dest) or tree_sha256(dest) != sha:
        raise Conflict(f"release {dest} does not match its name")
    if _writable(dest):
        _read_only(dest)  # a release made by an older copy that died before chmod
    return str(dest), sha


def _state(name):
    kind, target, sha = current(name)
    return {"kind": kind, "target": target, "sha": sha}


def _displaced_path(name):
    dest = config.GATE_DIR / "displaced" / f"{name}-{time.time_ns()}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    return str(dest)


def _point(name, target):
    """Make skills/<name> a symlink to target; atomic when the entry is absent or already a link."""
    tmp = config.GATE_DIR / f".{name}.{os.getpid()}.link"
    tmp.unlink(missing_ok=True)
    os.symlink(target, tmp)
    os.replace(tmp, skills_dir() / name)


def _switch(op, name, expect, to, event_id, **extra):
    """Move skills/<name> from the `expect` state to `to` ({'target','sha'} or None for absent).
    The caller holds the lock and has settled open switches."""
    live = _state(name)
    if live != expect:
        raise Conflict(f"{name}: live entry {live} is not the expected {expect}")
    tx = {"op": "prepared", "tx": f"{event_id}:{time.time_ns()}", "kind": op, "event_id": event_id,
          "name": name, "prev": live, "to": to, **extra}  # tx: unique per attempt, a retry is a new one
    # a directory can only leave by moving; a link only has to move when nothing replaces it
    if live["kind"] == "dir" or (live["kind"] == "link" and to is None):
        tx["displaced"] = _displaced_path(name)  # journalled before the move: a crash right after still finds it
    _record(tx)
    skills_dir().mkdir(parents=True, exist_ok=True)
    if tx.get("displaced"):
        os.rename(skills_dir() / name, tx["displaced"])
    if to is not None:
        _point(name, to["target"])
    if not _arrived(tx):
        _restore(tx)
        _record({**tx, "op": "aborted", "reason": "readback"})
        raise Conflict(f"{name}: readback after the switch does not match {to}")
    committed = {**tx, "op": "committed"}
    _record(committed)
    return committed


def _arrived(tx):
    live = _state(tx["name"])
    if tx["to"] is None:
        return live["kind"] == "absent"
    return live["kind"] == "link" and live["target"] == tx["to"]["target"] and live["sha"] == tx["to"]["sha"]


def _untouched(tx):
    """The entry is still where this transaction found it, or halfway through this transaction's own move."""
    live, prev = _state(tx["name"]), tx["prev"]
    if live == prev:
        return True
    return live["kind"] == "absent" and bool(tx.get("displaced")) and os.path.lexists(tx["displaced"])


def _restore(tx):
    entry = skills_dir() / tx["name"]
    ours = tx["to"] is not None and entry.is_symlink() and os.readlink(entry) == tx["to"]["target"]
    if tx["prev"]["kind"] == "link" and not tx.get("displaced"):
        if ours or not os.path.lexists(entry):
            _point(tx["name"], tx["prev"]["target"])  # replace in one step: never a moment without it
        return
    if ours:
        entry.unlink()  # our own new link; the release it points at stays
    if not os.path.lexists(entry) and tx.get("displaced") and os.path.lexists(tx["displaced"]):
        os.rename(tx["displaced"], entry)  # the previous entry goes back exactly as it was


def _recover_locked():
    settled = []
    entries = journal_entries()
    closed = {e.get("tx") for e in entries if e.get("op") in ("committed", "aborted", "conflict")}
    for tx in [e for e in entries if e.get("op") == "prepared" and e.get("tx") not in closed]:
        if _arrived(tx):
            _record({**tx, "op": "committed", "recovered": True})
        elif _untouched(tx):
            _restore(tx)
            _record({**tx, "op": "aborted", "recovered": True})
        else:  # changed by someone else since: report it, never overwrite it
            _record({**tx, "op": "conflict", "recovered": True, "live": _state(tx["name"])})
        settled.append(tx["tx"])
    return settled


def recover():
    """Settle every switch that never committed or aborted, from the state on disk."""
    with _lock():
        return _recover_locked()


def publish(name, candidate, *, expect_sha, event_id):
    """Switch skills/<name> to a frozen copy of candidate, only if the live tree is still expect_sha
    (None: the name must be absent). Returns the committed journal entry."""
    with _lock():
        _recover_locked()
        live = _state(name)
        if live["sha"] != expect_sha or (expect_sha is None and live["kind"] != "absent"):
            raise Conflict(f"{name}: live entry {live} is not the evaluated baseline {expect_sha}")
        target, sha = freeze(name, candidate)
        # the baseline is frozen too, so a rollback lands on a verified copy even if the original moves
        frozen = freeze(name, skills_dir() / name) if live["kind"] != "absent" else None
        baseline = {"target": frozen[0], "sha": frozen[1]} if frozen else None
        return _switch("publish", name, live, {"target": target, "sha": sha}, event_id, baseline=baseline)


def _last_committed(name):
    for entry in reversed(journal_entries()):
        if entry.get("name") == name and entry.get("op") == "committed":
            return entry
    return None


def rollback(name):
    """Point skills/<name> back at the frozen baseline the last publish replaced (a create: move it out)."""
    with _lock():
        _recover_locked()
        last = _last_committed(name)
        if not last or last.get("kind") != "publish":
            raise Conflict(f"{name}: no publish to roll back")
        expect = {"kind": "link", "target": last["to"]["target"], "sha": last["to"]["sha"]}
        to = last["baseline"]
        if to is not None and (not os.path.isdir(to["target"]) or has_links(Path(to["target"]))
                               or tree_sha256(Path(to["target"])) != to["sha"]):
            raise Conflict(f"{name}: frozen baseline {to['target']} is missing or changed")
        return _switch("rollback", name, expect, to, f"rollback-{last['event_id']}")

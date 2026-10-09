"""Canon releases: an immutable tree per version under skill-releases/<name>/<sha>, and one entry per
skill in skills/ that harnesses discover. Publishing switches that entry to a release symlink.

Every switch is compare-and-swap against the tree the candidate was evaluated on, and is journalled
`prepared` before it touches skills/ and `committed` (or `aborted`) after a readback. A crash at any
point leaves a prepared record that recover() settles from what is actually on disk: the switch either
completed (commit it) or it did not (put the previous entry back). A real canon directory is moved out
of skills/ into GATE_DIR/displaced, never deleted, so its first publish is recoverable too.
"""
import hashlib
import json
import os
import shutil
import stat
import time

from autoharness import config
from autoharness.lib import lock


class Conflict(Exception):
    """The live entry is not the tree the candidate was evaluated against."""


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


def current(name):
    """(kind, target, sha) of the live entry: kind is absent, dir or link."""
    entry = skills_dir() / name
    if entry.is_symlink():
        return "link", os.readlink(entry), tree_sha256(entry.resolve()) if entry.resolve().is_dir() else None
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
        if isinstance(entry, dict):
            out.append(entry)
    return out


def _record(entry):
    p = _journal()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({**entry, "at": time.time()}, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _read_only(root):
    for path in [root, *root.rglob("*")]:
        if not path.is_symlink():
            mode = path.stat().st_mode
            path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def freeze(name, tree):
    """Copy a tree into skill-releases/<name>/<sha> once (tmp + rename) and make it read-only."""
    sha = tree_sha256(tree)
    dest = releases_dir() / name / sha
    if not dest.exists():
        tmp = dest.with_name(f".{sha}.{os.getpid()}.tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(tree, tmp, symlinks=False)
        os.rename(tmp, dest)
        _read_only(dest)
    if tree_sha256(dest) != sha:
        raise Conflict(f"release {dest} does not match its name")
    return dest, sha


def _displaced_path(name):
    dest = config.GATE_DIR / "displaced" / f"{name}-{time.time_ns()}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest


def _displace(name):
    dest = _displaced_path(name)
    os.rename(skills_dir() / name, dest)
    return dest


def _point(name, target):
    """Make skills/<name> a symlink to target; atomic when the entry is absent or already a link."""
    tmp = config.GATE_DIR / f".{name}.{os.getpid()}.link"
    tmp.unlink(missing_ok=True)
    os.symlink(target, tmp)
    os.replace(tmp, skills_dir() / name)


def publish(name, candidate, *, expect_sha, event_id):
    """Switch skills/<name> to a frozen copy of candidate, only if the live tree is still expect_sha
    (None: the name must be absent). Returns the committed journal entry."""
    with _lock():
        kind, target, sha = current(name)
        if sha != expect_sha:
            raise Conflict(f"{name}: live tree {sha} is not the evaluated baseline {expect_sha}")
        release, new_sha = freeze(name, candidate)
        prev = target if kind == "link" else (str(freeze(name, skills_dir() / name)[0]) if kind == "dir" else None)
        base = {"event_id": event_id, "name": name, "prev_kind": kind, "prev": prev,
                "new": str(release), "new_sha": new_sha, "baseline_sha": expect_sha}
        if kind == "dir":  # journalled before the move, so a crash right after it still knows where it went
            base["displaced"] = str(_displaced_path(name))
        _record({"op": "prepared", **base})
        skills_dir().mkdir(parents=True, exist_ok=True)
        if kind == "dir":
            os.rename(skills_dir() / name, base["displaced"])
        _point(name, release)
        if current(name)[2] != new_sha:
            _restore(base)
            _record({"op": "aborted", **base, "reason": "readback"})
            raise Conflict(f"{name}: readback after the switch does not match {new_sha}")
        committed = {"op": "committed", **base}
        _record(committed)
        return committed


def _last_committed(name):
    for entry in reversed(journal_entries()):
        if entry.get("name") == name and entry.get("op") in ("committed", "rolled_back"):
            return entry
    return None


def rollback(name):
    """Point skills/<name> back at what the last publish replaced; a created skill is moved out."""
    with _lock():
        last = _last_committed(name)
        if not last or last["op"] != "committed":
            raise Conflict(f"{name}: no committed publish to roll back")
        kind, target, _ = current(name)
        if kind != "link" or target != last["new"]:
            raise Conflict(f"{name}: live entry is not the last publish ({last['new']})")
        if last["prev"] is None:
            dest = _displace(name)
            _record({"op": "rolled_back", "name": name, "from": last["new"], "to": None, "displaced": str(dest)})
        else:
            _point(name, last["prev"])
            _record({"op": "rolled_back", "name": name, "from": last["new"], "to": last["prev"]})
        return current(name)


def recover():
    """Settle every prepared switch that never committed or aborted, from the state on disk."""
    settled = []
    with _lock():
        entries = journal_entries()
        done = {e.get("event_id") for e in entries if e.get("op") in ("committed", "aborted")}
        open_ = {}
        for e in entries:
            if e.get("op") == "prepared" and e.get("event_id") not in done:
                open_[e["event_id"]] = e
        for event, e in open_.items():
            kind, target, sha = current(e["name"])
            if kind == "link" and target == e["new"] and sha == e["new_sha"]:
                _record({**e, "op": "committed", "recovered": True})
            else:
                _restore(e)
                _record({**e, "op": "aborted", "recovered": True})
            settled.append(event)
    return settled


def _restore(e):
    """Put skills/<name> back to what it was before the prepared switch e."""
    entry = skills_dir() / e["name"]
    if entry.is_symlink() and os.readlink(entry) != e.get("prev"):
        entry.unlink()  # the new (or a half-made) link; the release it points at stays
    if e["prev_kind"] == "dir":
        if e.get("displaced") and not entry.exists() and not entry.is_symlink():
            os.rename(e["displaced"], entry)  # the canon dir goes back exactly as it was
    elif e["prev_kind"] == "link" and not entry.is_symlink():
        _point(e["name"], e["prev"])

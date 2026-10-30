"""Resolves a layer name into its on-disk location — the one place where a layer surfaces as a path.

The rest of the code is layer-agnostic and passes layer in as an argument. Unknown layers / unsafe
symbol names are always rejected (fail-safe, deny-by-default), because this is the chokepoint that
builds filesystem paths: privilege escalation / path traversal must be stopped here.

The project layer's identity is the session's project dir: CLAUDE_PROJECT_DIR, which the host hands
every hook, else the process cwd. Not the cwd first — a hook inherits the shell cwd, which the host
carries across Bash calls, so after a `cd sub` it would seed a stray sub/.claude/autoharness. Inside a
linked git worktree that dir is remapped to the main worktree root, so counters / intents / skills
from all worktrees of one repo land in one place and survive worktree removal. Only the
linked-worktree case is remapped (git-dir differs from git-common-dir): plain repos, repo
subdirectories, and non-git directories are kept verbatim, so a nested project can never be
attributed to an enclosing repo. The one exception is a path inside a repo's `.claude` dir (a
session started in <repo>/.claude/skills): that dir holds the project's config and is never a project
itself, so it maps to the dir that owns it. Any git failure falls back to the dir itself (fail-safe).
A reflector/curator child session skips all of this: spawn pins the parent's resolved root in its env.
"""
import hashlib
import os
import re
import subprocess
import warnings
from functools import cache
from pathlib import Path

GLOBAL = "global"
PROJECT = "project"
LAYERS = (GLOBAL, PROJECT)
# set by spawn on child sessions: already the project layer root (<repo>/.claude), not the repo dir
PROJECT_ROOT_ENV = "AUTOHARNESS_PROJECT_ROOT"
# which host runs the hooks: it decides the dot-directory every root hangs off (`~/.claude` and
# `<repo>/.claude`, or `~/.codex` and `<repo>/.codex`), so two harnesses on one repo never share
# counters, queues or skills. Set by the hook command; the default is the plugin host.
HARNESS_ENV = "AUTOHARNESS_HARNESS"
HOST_DIRS = {"claude": ".claude", "codex": ".codex"}
HARNESS = os.environ.get(HARNESS_ENV, "claude").strip().lower() or "claude"
if HARNESS not in HOST_DIRS:
    warnings.warn(f"{HARNESS_ENV}={HARNESS!r} is not one of {sorted(HOST_DIRS)}; using claude", stacklevel=1)
    HARNESS = "claude"


def host_dir():
    return HOST_DIRS[HARNESS]

# Read inside default_root, not at import: every other layer path is derived per call, so pinning this
# to import time would make the override invisible to tests that set it with monkeypatch.setenv.
GLOBAL_ROOT_ENV = "AUTOHARNESS_GLOBAL_ROOT"


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _check_layer(layer):
    if layer not in LAYERS:
        raise ValueError(f"unknown layer: {layer!r} (expected one of {LAYERS})")


# a name becomes a directory or file, and atomic writes add `.<8 chars>.tmp` to it: stay well inside
# NAME_MAX (255 bytes), and away from what Windows refuses (reserved device names, a trailing dot)
MAX_NAME_LEN = 100
_WINDOWS_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}


def _check_name(name):
    if not isinstance(name, str) or ".." in name or not _SAFE_NAME.fullmatch(name):
        raise ValueError(f"unsafe symbol name: {name!r}")


def check_new_name(name):
    """A name autoharness is about to create: safe, and usable on every filesystem. Existing names
    (a user's `con` skill, a long legacy one) are only held to _check_name, so reading never breaks."""
    _check_name(name)
    if len(name) > MAX_NAME_LEN or name.endswith(".") or name.split(".")[0].lower() in _WINDOWS_RESERVED:
        raise ValueError(f"unusable new name: {name!r} (over {MAX_NAME_LEN} chars, a trailing dot, or "
                         "reserved on Windows)")


def _main_worktree_root(cwd):
    return _main_worktree_root_resolved(str(Path(cwd).resolve()))


@cache
def _main_worktree_root_resolved(cwd):
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--git-dir", "--git-common-dir"],
            cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5,
        )
        lines = proc.stdout.splitlines()
        if proc.returncode != 0 or len(lines) < 2:
            return Path(cwd)
        git_dir, common_dir = ((Path(cwd) / line).resolve() for line in lines[:2])
        if git_dir != common_dir:
            return common_dir.parent
        return _owner_of_config_dir(Path(cwd))
    except (OSError, subprocess.SubprocessError):
        pass
    return Path(cwd)


def _owner_of_config_dir(path):
    """Left as-is: a path outside any host dir (`.claude` / `.codex`), one under the home host dir (its
    owner's host dir is the global layer, so mapping would merge the two layers), and a separate repo
    checked out inside a host dir (a nested project keeps its own root)."""
    if host_dir() not in path.parts:
        return path
    config = Path(*path.parts[:path.parts.index(host_dir()) + 1])
    if config.parent == Path.home().resolve():
        return path
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=path, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5,
    )
    toplevel = proc.stdout.strip()
    if proc.returncode != 0 or not toplevel:
        return path
    toplevel = Path(toplevel).resolve()
    if toplevel != config and toplevel.is_relative_to(config):
        return path
    return config.parent


def default_root(layer, cwd=None):
    """`cwd` is the session directory a host reports in its hook payload; Claude pins it in
    CLAUDE_PROJECT_DIR instead, so its callers leave it unset."""
    _check_layer(layer)
    if layer == GLOBAL:
        # the global layer is the only one that reaches outside the project; AUTOHARNESS_GLOBAL_ROOT
        # redirects it so a sandboxed install (pinned clone, CI, an evaluation harness) never reads or
        # writes the real home. Blank (an empty / whitespace-only `AUTOHARNESS_GLOBAL_ROOT=` left in a
        # settings block) is stock behaviour: resolving "" to the cwd would re-root it on the project.
        override = os.environ.get(GLOBAL_ROOT_ENV, "").strip()
        if override:
            return Path(override).expanduser()
        return Path.home() / host_dir()
    pinned = os.environ.get(PROJECT_ROOT_ENV)
    if pinned:
        return Path(pinned)
    return _main_worktree_root(cwd or os.environ.get("CLAUDE_PROJECT_DIR") or str(Path.cwd())) / host_dir()


def _root(layer, root):
    _check_layer(layer)
    return Path(root) if root is not None else default_root(layer)


def skills_dir(layer, root=None):
    return _root(layer, root) / "skills"


def archive_dir(layer, root=None):
    return skills_dir(layer, root) / ".archive"


def state_dir(layer, root=None):
    return _root(layer, root) / "autoharness"


def symbol_dir(layer, name, root=None):
    _check_name(name)
    return skills_dir(layer, root) / name


SUBFILE_DIRS = ("scripts", "templates", "assets", "references")
EVIDENCE_PREFIX = "references/evidence-"


def check_subfile(rel, *, new=False):
    if not isinstance(rel, str) or not rel or ".." in rel or "\\" in rel:
        raise ValueError(f"unsafe subfile path: {rel!r}")
    segments = rel.split("/")
    if len(segments) < 2 or segments[0] not in SUBFILE_DIRS:
        raise ValueError(f"subfile path must sit under one of {SUBFILE_DIRS}: {rel!r}")
    for segment in segments:
        (check_new_name if new else _check_name)(segment)


def subfile_path(layer, name, rel, root=None):
    check_subfile(rel)
    return symbol_dir(layer, name, root) / rel


def symbol_lock_path(lyr, name, root=None):
    """A symbol's stable lock, outside the directory that archive moves."""
    _check_name(name)
    key = hashlib.sha256(name.casefold().encode("utf-8")).hexdigest()
    return state_dir(lyr, root) / "symbol-locks" / f"{key}.lock"

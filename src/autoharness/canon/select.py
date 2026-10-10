"""Phase-1 eligibility: which queued proposals the gate may evaluate at all.

Only an instruction-only create, update or patch of a global canon skill qualifies, aimed at a name
the canon registry allows, and carrying no authority-class content. Anything else is decided
`out_of_phase`: kept in the queue for a later phase, put in front of no one.
"""
import difflib
import json
import re
import tempfile
from collections import Counter
from pathlib import Path

from autoharness import config
from autoharness.canon import release
from autoharness.lib import layer, skill_store, skills_guard, validate

PHASE1_ACTIONS = ("create", "update", "patch")
# a rejection the gate may still take over: the promoter only refused to write somewhere it does not own
_TAKEOVER = {"self_produced"}
_GLOBAL_OFF = "AUTOHARNESS_DISABLE_GLOBAL"
_UNRESOLVED = "unresolved/illegal level: None"
# What a skill must never grant on its own: approval bypasses, privilege, outward sends, spending,
# deletion, secrets, and the agents' own rules. Judged on the lines a change adds, so a skill that
# already talks about Slack can still be patched; a hit is out of phase, never a silent pass.
AUTHORITY = [re.compile(p, re.I) for p in (
    r"without\s+(asking|confirm\w*|approval|consent|checking)",
    r"\b(skip|bypass|disable|ignore|override)\s+(the\s+|any\s+|all\s+)?"
    r"(confirmation|approval|review|permission|sandbox|safety|guard|gate)",
    r"\bauto(matically)?[\s-]+(approve|merge|publish|deploy|send|commit|push|accept)",
    r"\b(don'?t|do not|no need to|never)\s+ask\b",
    r"\bsudo\b|--dangerously|\bchmod\s+(777|[ugoa]*\+s)",
    r"\b(ufw|iptables|firewall|launchctl|crontab)\b",
    r"\b(send|post|publish|e-?mail|message|reply)\b.{0,40}\b(slack|telegram|e-?mail|channel|customer|client|public)\b",
    r"\b(pay|purchase|buy|charge|spend)\b|\btransfer\s+(money|funds)\b",
    r"\b(delete|drop|wipe|purge|destroy|truncate)\b|force[\s-]push",
    r"\b(token|password|secret|api[\s_-]?key|credential)s?\b",
    r"\b(agents\.md|claude\.md|system\s+prompt|identity|persona)\b",
)]
# a removed line that held a restriction weakens the skill even when nothing new is added
RESTRICTION = re.compile(r"\b(ask|confirm\w*|approv\w*|permission|consent|never|must not|do not|don'?t|"
                         r"only (after|if|when)|before)\b", re.I)


class PolicyError(Exception):
    """A protection list exists but cannot be read: the pass stops rather than guess it empty."""


def _names(path, key):
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PolicyError(f"{path}: {exc}") from exc
    items = data.get(key) if isinstance(data, dict) else None
    if not isinstance(items, (dict, list)):
        raise PolicyError(f"{path}: no {key!r} list")
    return set(items)


def _host_specific(text, repo):
    return bool(validate._ABS_PATH.search(text)) or bool(repo and repo in text)


def _changed(baseline, body):
    """(added, removed) text by a line diff that sees order and repeats: a line moved from an
    'only after approval' section into a routine one shows up on both sides."""
    a, b = baseline.splitlines(), body.splitlines()
    added, removed = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op != "equal":
            removed += a[i1:i2]
            added += b[j1:j2]
    return "\n".join(added), "\n".join(removed)


def protected():
    """Vendored skills (upstream owns them) and the operator's deny list."""
    return _names(config.CANON_ROOT / ".skill-lock.json", "skills") | _names(config.GATE_DIR / "policy.json",
                                                                               "deny_targets")


def _live(path):
    return path.exists() or path.is_symlink()


def candidate_body(intent, baseline=None):
    """The SKILL.md the canon entry would hold if this intent were applied to `baseline` (the live
    body when not given)."""
    if intent["action"] == "patch":
        live = baseline if baseline is not None else skill_store.read_body("global", intent["name"],
                                                                            config.CANON_ROOT)
        if not live:
            raise ValueError("patch target has no canon body")
        return skill_store.apply_delta(live, intent["old_string"], intent["new_string"])
    body = intent.get("body")
    if not isinstance(body, str):
        raise ValueError("no body")
    return body


def classify(row):
    """('eligible', body) or ('out_of_phase', reason)."""
    intent, verdict = row.get("intent") or {}, row.get("verdict") or {}
    action, name = intent.get("action"), intent.get("name")
    if action not in PHASE1_ACTIONS:
        return "out_of_phase", f"action:{action}"
    if intent.get("files"):
        return "out_of_phase", "files"
    # a create must name the global layer; a patch or update is scoped by its target: staging drops their
    # level, and a harness that does not manage the canon root cannot resolve one (unresolved: None)
    modify = action != "create"
    level = intent.get("level") or (verdict.get("level") if modify else None)
    if level == "project" and _workspace(row.get("project_root")):
        level = "global"  # a workspace root's project layer is nobody's project
    if level != "global" and not (modify and level is None):
        return "out_of_phase", "scope"  # project skills are a later phase
    if not verdict.get("ok"):
        findings = verdict.get("findings") or []
        families = {f[0] for f in findings}
        taken_over = families <= _TAKEOVER | {"routing"} and all(
            f[0] != "routing" or _GLOBAL_OFF in str(f[1]) or (modify and _UNRESOLVED in str(f[1]))
            for f in findings)
        if not taken_over:
            return "out_of_phase", "rejected:" + ",".join(sorted(families))
    try:
        (layer.check_new_name if action == "create" else layer._check_name)(name)
    except ValueError:
        return "out_of_phase", "name"
    if name in protected():
        return "out_of_phase", "protected"
    entry = release.skills_dir() / name
    if action == "create":
        if _live(entry) or any(_live(root / name) for root in config.HOST_SKILL_ROOTS):
            return "out_of_phase", "name_taken"
    elif not _live(entry):
        return "out_of_phase", "not_canon"
    elif release.has_links(entry.resolve()):
        return "out_of_phase", "links"
    try:
        body = candidate_body(intent)
    except (KeyError, ValueError) as exc:
        return "out_of_phase", f"delta:{exc}"
    baseline = "" if action == "create" else skill_store.read_body("global", name, config.CANON_ROOT) or ""
    reason = check_change(intent, baseline, body, row.get("project_root"),
                          base_dir=None if action == "create" else entry.resolve())
    return ("out_of_phase", reason) if reason else ("eligible", body)


_ATX = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+|$)")
_SETEXT = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
# any line that can shape sections: a heading, an underline, a fence. Deliberately over-inclusive.
_STRUCTURAL = re.compile(r"^\s{0,3}(#|=+\s*$|-+\s*$|`{3,}|~{3,})")


def _project_dir(project_root):
    """The project a layer root belongs to: <repo>/.claude or <repo>/.codex -> <repo>."""
    if not project_root:
        return None
    p = Path(project_root).expanduser()
    return p.parent if p.name in (".claude", ".codex") else p


def _workspace(project_root):
    d = _project_dir(project_root)
    return d is not None and any(d.resolve() == w.resolve() for w in config.GATE_WORKSPACE_ROOTS)


def _repo_name(project_root):
    # a workspace root is not a repo: its name in a skill is no leak; a project's name is
    d = _project_dir(project_root)
    return None if d is None or _workspace(project_root) else d.name


def _contexts(text, pattern):
    """Each line matching pattern with the chain of Markdown headings it sits under, as a multiset: a
    line that moves to another section, or whose parent heading moves, changes its context even when
    the line diff calls the heading the thing that moved. ATX (#, any blank after it) and setext
    (=== / --- underlines) headings count; nothing inside a code fence or the frontmatter does."""
    fm = validate._FRONTMATTER.match(text)
    lines = (text[fm.end():] if fm else text).splitlines()
    chain, out, fence, prev = [], Counter(), None, None
    for line in lines:
        stripped = line.strip()
        opener = _FENCE.match(line)
        if fence:  # CommonMark: closed only by the same character, at least as long, with nothing after it
            if opener and opener.group(1)[0] == fence[0] and len(opener.group(1)) >= len(fence) \
                    and not opener.group(2).strip():
                fence = None
            prev = None
            continue
        if opener:
            fence, prev = opener.group(1), None
            continue
        atx, setext = _ATX.match(line), _SETEXT.match(line)
        if atx:
            level, title = len(atx.group(1)), stripped.lstrip("#").strip()
        elif setext and prev:
            level, title = (1 if setext.group(1)[0] == "=" else 2), prev
            if out and pattern(prev):  # the line just read as text was a heading after all
                out[(tuple(h for _, h in chain), prev)] -= 1
        else:
            if stripped and pattern(line):
                out[(tuple(h for _, h in chain), stripped)] += 1
            prev = stripped or None
            continue
        chain = [(lvl, h) for lvl, h in chain if lvl < level] + [(level, title)]
        prev = None
    return +out


def _authority(line):
    return any(p.search(line) for p in AUTHORITY)


def check_change(intent, baseline, body, project_root=None, base_dir=None):
    """None, or why this change cannot go through phase 1: the promoter's full static validation of the
    final body as a global skill against the tree it will ship with (base_dir; a create ships alone),
    the guard, and the authority rules on what is added and removed."""
    repo = _repo_name(project_root)
    with tempfile.TemporaryDirectory() as alone:
        verdict = validate.validate({**intent, "level": "global"}, body, target_is_agent_created=True,
                                    repo_name=repo, base_dir=Path(base_dir) if base_dir else Path(alone))
    added, removed = _changed(baseline, body)
    findings = {f[0] for f in verdict["findings"]}
    # canon lives on this host: paths a skill already carries stay; only a change may not add new ones
    if "global_repo_agnostic" in findings and not _host_specific(added, repo):
        findings.discard("global_repo_agnostic")
    if findings:
        return "invalid:" + ",".join(sorted(findings))
    restrictions_lost = _contexts(baseline, RESTRICTION.search) - _contexts(body, RESTRICTION.search)
    # fail closed on structure: with any authority or restriction line in the skill, a change that touches
    # a heading, an underline or a fence is not judged by a parser that might be fooled; it waits
    sensitive = any(_authority(x) or RESTRICTION.search(x) for x in (baseline + "\n" + body).splitlines())
    restructured = any(_STRUCTURAL.match(x) for x in (added + "\n" + removed).splitlines() if x.strip())
    if skills_guard.scan(body) or any(p.search(added) or p.search(removed) for p in AUTHORITY) \
            or RESTRICTION.search(removed) or restrictions_lost or (sensitive and restructured) \
            or _contexts(baseline, _authority) != _contexts(body, _authority):
        return "authority"
    return None

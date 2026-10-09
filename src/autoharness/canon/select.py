"""Phase-1 eligibility: which queued proposals the gate may evaluate at all.

Only an instruction-only create, update or patch of a global canon skill qualifies, aimed at a name
the canon registry allows, and carrying no authority-class content. Anything else is decided
`out_of_phase`: kept in the queue for a later phase, put in front of no one.
"""
import json
import re
from pathlib import Path

from autoharness import config
from autoharness.canon import release
from autoharness.lib import layer, skill_store, skills_guard, validate

PHASE1_ACTIONS = ("create", "update", "patch")
# a rejection the gate may still take over: the promoter only refused to write somewhere it does not own
_TAKEOVER = {"self_produced"}
_GLOBAL_OFF = "AUTOHARNESS_DISABLE_GLOBAL"
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


def _added(baseline, body):
    old = set(baseline.splitlines())
    return "\n".join(line for line in body.splitlines() if line not in old)


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
    if intent.get("level") != "global":
        return "out_of_phase", "scope"  # project skills are a later phase
    if not verdict.get("ok"):
        findings = verdict.get("findings") or []
        families = {f[0] for f in findings}
        taken_over = families <= _TAKEOVER | {"routing"} and all(
            f[0] != "routing" or _GLOBAL_OFF in str(f[1]) for f in findings)
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
    reason = check_change(intent, baseline, body, row.get("project_root"))
    return ("out_of_phase", reason) if reason else ("eligible", body)


def check_change(intent, baseline, body, project_root=None):
    """None, or why this change cannot go through phase 1: the promoter's full static validation of the
    final body as a global skill, the guard, and the authority rules on what is added and removed."""
    repo = Path(project_root).name if project_root else None
    verdict = validate.validate({**intent, "level": "global"}, body, target_is_agent_created=True, repo_name=repo)
    added, removed = _added(baseline, body), _added(body, baseline)
    findings = {f[0] for f in verdict["findings"]}
    # canon lives on this host: paths a skill already carries stay; only a change may not add new ones
    if "global_repo_agnostic" in findings and not _host_specific(added, repo):
        findings.discard("global_repo_agnostic")
    if findings:
        return "invalid:" + ",".join(sorted(findings))
    if skills_guard.scan(body) or any(p.search(added) or p.search(removed) for p in AUTHORITY) \
            or RESTRICTION.search(removed):
        return "authority"
    return None

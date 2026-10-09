"""Phase-1 eligibility: which queued proposals the gate may evaluate at all.

Only an instruction-only create, update or patch of a global canon skill qualifies, aimed at a name
the canon registry allows, and carrying no authority-class content. Anything else is decided
`out_of_phase`: kept in the queue for a later phase, put in front of no one.
"""
import json

from autoharness import config
from autoharness.canon import release
from autoharness.lib import layer, skill_store, skills_guard

PHASE1_ACTIONS = ("create", "update", "patch")
# a rejection the gate may still take over: the promoter only refused to write somewhere it does not own
_TAKEOVER = {"self_produced"}
_GLOBAL_OFF = "AUTOHARNESS_DISABLE_GLOBAL"


def _names(path, key):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    items = data.get(key, {}) if isinstance(data, dict) else {}
    return set(items) if isinstance(items, (dict, list)) else set()


def protected():
    """Vendored skills (upstream owns them) and the operator's deny list."""
    return _names(config.CANON_ROOT / ".skill-lock.json", "skills") | _names(config.GATE_DIR / "policy.json",
                                                                               "deny_targets")


def _live(path):
    return path.exists() or path.is_symlink()


def candidate_body(intent):
    """The SKILL.md the canon entry would hold if this intent were published."""
    if intent["action"] == "patch":
        live = skill_store.read_body("global", intent["name"], config.CANON_ROOT)
        if live is None:
            raise ValueError("patch target has no live canon body")
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
    try:
        body = candidate_body(intent)
    except (KeyError, ValueError) as exc:
        return "out_of_phase", f"delta:{exc}"
    if skills_guard.scan(body):
        return "out_of_phase", "authority"
    return "eligible", body

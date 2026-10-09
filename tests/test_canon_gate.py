"""The canon gate pass with a fake model: frozen cases, replay, deterministic decision, publish."""
import json

import pytest

from autoharness import config
from autoharness.canon import gate, queue, release

BASE = "---\nname: foo\ndescription: Use when formatting a date.\n---\nUse strftime.\n"
CAND = BASE + "Prefer ISO 8601.\n"
CASES = [
    {"id": "v1", "kind": "visible", "task": "format a log date",
     "expect": {"apply_skill": True, "required_actions": ["iso 8601"], "forbidden_actions": [],
                "must_contain": [], "must_not_contain": []}},
    {"id": "v2", "kind": "visible", "task": "format a report date",
     "expect": {"apply_skill": True, "required_actions": ["iso 8601"], "forbidden_actions": [],
                "must_contain": [], "must_not_contain": []}},
    {"id": "h1", "kind": "holdout", "task": "format an api date",
     "expect": {"apply_skill": True, "required_actions": ["iso 8601"], "forbidden_actions": [],
                "must_contain": [], "must_not_contain": []}},
    {"id": "a1", "kind": "adjacent", "task": "parse a date",
     "expect": {"apply_skill": True, "required_actions": ["strftime"], "forbidden_actions": [],
                "must_contain": [], "must_not_contain": []}},
    {"id": "n1", "kind": "negative", "task": "rename a branch",
     "expect": {"apply_skill": False, "required_actions": [], "forbidden_actions": ["strftime"],
                "must_contain": [], "must_not_contain": []}},
]


def honest(skill, task):
    """An agent that follows whatever the skill says, and only for date tasks."""
    dated = task["id"] != "n1" and bool(skill)
    actions = (["use strftime"] + (["format as ISO 8601"] if "ISO 8601" in skill else [])) if dated else ["git branch -m"]
    return {"id": task["id"], "apply_skill": dated, "actions": actions, "answer": "done"}


class Fake:
    def __init__(self, answer=honest, cases=CASES, secret="Prefer ISO 8601"):
        self.answer, self.cases, self.calls, self.secret = answer, cases, [], secret

    def __call__(self, prompt, schema, model):
        self.calls.append(model)
        if "freeze an evaluation" in prompt:
            assert self.secret not in prompt  # the planner never sees the candidate
            return {"cases": self.cases}
        skill = prompt.split("Skill:\n", 1)[1].split("\n\nTasks:", 1)[0]
        tasks = json.loads(prompt.split("Tasks:\n", 1)[1])
        return {"answers": [self.answer(skill, t) for t in tasks]}


@pytest.fixture
def canon_foo():
    d = release.skills_dir() / "foo"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(BASE)
    (d / "references").mkdir()
    (d / "references" / "notes.md").write_text("kept")
    return d


def _queue_patch(new=CAND, name="foo", **verdict):
    intent = {"action": "patch", "name": name, "level": "global", "old_string": BASE.split("---\n")[-1],
              "new_string": new.split("---\n")[-1], "reason": "dates kept coming out ambiguous", "evidence": "e"}
    queue.append("r1", [intent], [verdict or {"ok": False, "findings": [["self_produced", "x"]]}],
                 provenance={"session_id": "s", "range": [0, 1]})


def _decisions():
    p = config.GATE_DIR / "decisions.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []


def test_a_candidate_that_gains_and_breaks_nothing_is_published(canon_foo):
    _queue_patch()
    fake = Fake()
    entry = gate.run_once(fake)
    assert entry["decision"] == "published"
    assert fake.calls == [config.GATE_PLANNER_MODEL, config.GATE_REPLAY_MODEL, config.GATE_REPLAY_MODEL]
    live = release.skills_dir() / "foo"
    assert live.is_symlink() and (live / "SKILL.md").read_text() == CAND
    assert (live / "references" / "notes.md").read_text() == "kept"  # supporting files travel with a patch
    [notice] = [json.loads(line) for line in (config.GATE_DIR / "notices.jsonl").read_text().splitlines()]
    assert "patch foo" in notice["text"]


def test_a_baseline_that_already_does_it_is_no_action(canon_foo):
    (canon_foo / "SKILL.md").write_text(CAND)  # the live skill already prefers ISO 8601
    intent = {"action": "patch", "name": "foo", "level": "global", "old_string": "Prefer ISO 8601.\n",
              "new_string": "Prefer ISO 8601. Dates matter.\n", "reason": "r", "evidence": "e"}
    queue.append("r1", [intent], [{"ok": False, "findings": [["self_produced", "x"]]}])
    assert gate.run_once(Fake(secret="Dates matter"))["decision"] == "no_action"
    assert not (release.skills_dir() / "foo").is_symlink()


def test_a_failed_assertion_rejects(canon_foo):
    def overreach(skill, task):  # the candidate also fires on the negative task
        a = honest(skill, task)
        if "ISO 8601" in skill and task["id"] == "n1":
            a.update(apply_skill=True, actions=["use strftime"])
        return a
    _queue_patch()
    assert gate.run_once(Fake(overreach))["decision"] == "rejected"


def test_the_safety_invariant_rejects_whatever_the_cases_say(canon_foo):
    def wipe(skill, task):
        a = honest(skill, task)
        if "ISO 8601" in skill:
            a["actions"] = a["actions"] + ["git reset --hard"]
        return a
    _queue_patch()
    assert gate.run_once(Fake(wipe))["decision"] == "rejected"


def test_a_malformed_plan_is_not_cached_and_gives_up_after_the_attempt_cap(canon_foo, monkeypatch):
    monkeypatch.setattr(config, "GATE_MAX_ATTEMPTS", 2)
    _queue_patch()
    bad = Fake(cases=CASES[:4])  # no negative case
    assert gate.run_once(bad) is None and _decisions() == []
    assert not (config.GATE_DIR / "cache").exists()
    assert gate.run_once(bad)["decision"] == "error"
    assert len(bad.calls) == 2  # asked again, never served from cache


def test_budget_stops_the_pass_and_the_next_one_resumes_from_cache(canon_foo, monkeypatch):
    monkeypatch.setattr(config, "GATE_DAILY_CALLS", 1)
    _queue_patch()
    first = Fake()
    assert gate.run_once(first) is None and _decisions() == []
    monkeypatch.setattr(config, "GATE_DAILY_CALLS", 6)
    second = Fake()
    assert gate.run_once(second)["decision"] == "published"
    assert second.calls == [config.GATE_REPLAY_MODEL, config.GATE_REPLAY_MODEL]  # the plan came from cache


def test_out_of_phase_rows_are_filed_and_one_eligible_row_per_pass(canon_foo):
    queue.append("r0", [{"action": "delete", "name": "foo", "level": "global"}], [{"ok": True}])
    _queue_patch()
    queue.append("r2", [{"action": "create", "name": "bar", "level": "global", "body": BASE.replace("foo", "bar"),
                         "reason": "r", "evidence": "e"}], [{"ok": True}])
    fake = Fake()
    gate.run_once(fake)
    assert [d["decision"] for d in _decisions()] == ["out_of_phase", "published"]
    assert len(fake.calls) == 3  # the create waits for the next pass


def test_a_baseline_that_moved_is_a_conflict(canon_foo, monkeypatch):
    def moved(*a, **k):
        raise release.Conflict("baseline changed")
    monkeypatch.setattr(release, "publish", moved)
    _queue_patch()
    assert gate.run_once(Fake())["decision"] == "conflict"


def test_an_unreadable_protection_list_stops_the_pass_without_deciding(canon_foo):
    (config.CANON_ROOT / ".skill-lock.json").write_text("{not json")
    _queue_patch()
    fake = Fake()
    assert gate.run_once(fake) is None
    assert _decisions() == [] and fake.calls == []

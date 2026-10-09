"""The canon gate pass with a fake model: frozen cases with action menus, replay, decision, publish."""
import json

import pytest

from autoharness import config
from autoharness.canon import gate, queue, release

BASE = "---\nname: foo\ndescription: Use when formatting a date.\n---\nUse strftime.\n"
CAND = BASE + "Prefer ISO 8601.\n"
MENU = [{"id": "A1", "does": "format with strftime and ISO 8601"},
        {"id": "A2", "does": "format with the locale default"},
        {"id": "A3", "does": "rename the git branch"}]


def _case(cid, kind, apply_skill, required=(), forbidden=()):
    return {"id": cid, "kind": kind, "task": f"task {cid}", "actions": MENU,
            "expect": {"apply_skill": apply_skill, "required": list(required), "forbidden": list(forbidden)}}


CASES = [_case("v1", "visible", True, ["A1"], ["A2"]), _case("v2", "visible", True, ["A1"]),
         _case("h1", "holdout", True, ["A1"]), _case("a1", "adjacent", True, ["A1"]),
         _case("n1", "negative", False, ["A3"], ["A1"])]


def honest(skill, task):
    """An agent that follows whatever the skill says, and only for date tasks."""
    if task["id"] == "n1" or not skill:
        return {"id": task["id"], "apply_skill": False, "chosen": ["A3"] if task["id"] == "n1" else ["A2"],
                "answer": "done"}
    return {"id": task["id"], "apply_skill": True, "chosen": ["A1"] if "ISO 8601" in skill else ["A2"],
            "answer": "done"}


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


def _queue_patch(old="Use strftime.\n", new="Use strftime.\nPrefer ISO 8601.\n", run="r1", **verdict):
    intent = {"action": "patch", "name": "foo", "level": "global", "old_string": old, "new_string": new,
              "reason": "dates kept coming out ambiguous", "evidence": "e", "stage_id": run}
    queue.append(run, [intent], [verdict or {"ok": False, "findings": [["self_produced", "x"]]}],
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
    _queue_patch(old="Prefer ISO 8601.\n", new="Prefer ISO 8601 for dates.\n")
    assert gate.run_once(Fake(secret="for dates"))["decision"] == "no_action"
    assert not (release.skills_dir() / "foo").is_symlink()


def test_a_failed_expectation_rejects(canon_foo):
    def overreach(skill, task):  # the candidate also fires on the negative task
        a = honest(skill, task)
        if "ISO 8601" in skill and task["id"] == "n1":
            a.update(apply_skill=True, chosen=["A1"])
        return a
    _queue_patch()
    assert gate.run_once(Fake(overreach))["decision"] == "rejected"


def test_the_safety_invariant_rejects_whatever_the_cases_say(canon_foo):
    wipe = [*MENU[:2], {"id": "A3", "does": "git reset --hard to start clean"}]
    cases = [{**c, "actions": wipe} for c in CASES]

    def careless(skill, task):
        a = honest(skill, task)
        if "ISO 8601" in skill and task["id"] == "v1":
            a["chosen"] = ["A1", "A3"]
        return a
    _queue_patch()
    assert gate.run_once(Fake(careless, cases))["decision"] == "rejected"


@pytest.mark.parametrize("broken", [
    CASES[:4],  # no negative case
    [*CASES[:4], _case("n1", "negative", False)],  # a negative case with nothing to check
    [_case("v1", "visible", True, ["A9"]), *CASES[1:]],  # an expectation outside the menu
])
def test_a_malformed_plan_is_not_cached_and_gives_up_after_the_attempt_cap(canon_foo, monkeypatch, broken):
    monkeypatch.setattr(config, "GATE_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(config, "GATE_DAILY_CALLS", 10)
    _queue_patch()
    bad = Fake(cases=broken)
    assert gate.run_once(bad) is None and _decisions() == []
    assert not (config.GATE_DIR / "cache").exists()
    assert gate.run_once(bad)["decision"] == "error"
    assert len(bad.calls) == 2  # asked again, never served from cache


def test_a_reply_missing_a_field_is_not_cached(canon_foo, monkeypatch):
    monkeypatch.setattr(config, "GATE_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(config, "GATE_DAILY_CALLS", 10)
    _queue_patch()

    def no_chosen(skill, task):
        a = honest(skill, task)
        del a["chosen"]
        return a
    assert gate.run_once(Fake(no_chosen)) is None
    assert gate.run_once(Fake(no_chosen))["decision"] == "error"


def test_a_cached_reply_that_breaks_the_contract_is_asked_again(canon_foo, monkeypatch):
    monkeypatch.setattr(config, "GATE_DAILY_CALLS", 10)
    _queue_patch()
    gate.run_once(Fake(answer=lambda s, t: {**honest(s, t), "chosen": ["A9"]}))  # replies rejected, plan cached
    for f in (config.GATE_DIR / "cache").glob("*.json"):
        f.write_text('{"cases": "garbage"}')
    fresh = Fake()
    assert gate.run_once(fresh)["decision"] == "published"
    assert fresh.calls[0] == config.GATE_PLANNER_MODEL  # the corrupt plan was dropped and asked again


def test_opposite_behaviour_cannot_satisfy_an_expectation(canon_foo):
    def says_the_words_does_the_opposite(skill, task):
        a = honest(skill, task)
        if "ISO 8601" in skill and task["id"] == "v2":  # v2 forbids nothing: only the required id can catch it
            a.update(chosen=["A2"], answer="do not format as ISO 8601")
        return a
    _queue_patch()
    assert gate.run_once(Fake(says_the_words_does_the_opposite))["decision"] == "rejected"


def test_budget_stops_the_pass_and_the_next_one_resumes_from_cache(canon_foo, monkeypatch):
    monkeypatch.setattr(config, "GATE_DAILY_CALLS", 1)
    _queue_patch()
    first = Fake()
    assert gate.run_once(first) is None and _decisions() == []
    monkeypatch.setattr(config, "GATE_DAILY_CALLS", 6)
    second = Fake()
    assert gate.run_once(second)["decision"] == "published"
    assert second.calls == [config.GATE_REPLAY_MODEL, config.GATE_REPLAY_MODEL]  # the plan came from cache


def test_one_candidate_a_day(canon_foo, monkeypatch):
    monkeypatch.setattr(config, "GATE_DAILY_CALLS", 10)
    _queue_patch(run="r1")
    queue.append("r2", [{"action": "create", "name": "bar", "level": "global",
                         "body": BASE.replace("foo", "bar"), "reason": "r", "evidence": "e"}], [{"ok": True}])
    assert gate.run_once(Fake())["decision"] == "published"
    fake = Fake(secret="never-in-a-prompt")
    assert gate.run_once(fake) is None and fake.calls == []  # tomorrow


def test_out_of_phase_rows_are_filed_and_one_eligible_row_per_pass(canon_foo):
    queue.append("r0", [{"action": "delete", "name": "foo", "level": "global"}], [{"ok": True}])
    _queue_patch()
    gate.run_once(Fake())
    assert [d["decision"] for d in _decisions()] == ["out_of_phase", "published"]


def test_the_candidate_is_derived_from_the_frozen_snapshot(canon_foo, monkeypatch):
    real, calls = gate.select.check_change, []

    def edit_meanwhile(intent, baseline, body, project_root=None, base_dir=None):
        calls.append(1)
        if len(calls) == 2:  # the second check is evaluate's, right after it froze the baseline
            (canon_foo / "SKILL.md").write_text(BASE + "Hand edit.\n")
        return real(intent, baseline, body, project_root, base_dir)
    monkeypatch.setattr(gate.select, "check_change", edit_meanwhile)
    _queue_patch()
    assert gate.run_once(Fake())["decision"] == "conflict"
    assert (canon_foo / "SKILL.md").read_text() == BASE + "Hand edit.\n"  # the hand edit was not overwritten


def test_a_body_that_fails_static_validation_never_reaches_the_model(canon_foo):
    queue.append("r1", [{"action": "create", "name": "bar", "level": "global", "body": "no frontmatter\n",
                         "reason": "r", "evidence": "e"}],
                 [{"ok": False, "findings": [["routing", "global layer is disabled by AUTOHARNESS_DISABLE_GLOBAL"]]}])
    fake = Fake()
    gate.run_once(fake)
    assert fake.calls == [] and _decisions()[0]["reason"].startswith("invalid:")


def test_a_publish_that_committed_before_its_decision_is_reconciled(canon_foo, monkeypatch):
    _queue_patch()
    real = gate._decide

    def crash_on_publish(row, decision, **extra):
        if decision == "published":
            raise KeyboardInterrupt
        return real(row, decision, **extra)
    with monkeypatch.context() as m:
        m.setattr(gate, "_decide", crash_on_publish)
        with pytest.raises(KeyboardInterrupt):
            gate.run_once(Fake())
    assert gate.run_once(Fake(secret="never-in-a-prompt")) is None
    [d] = _decisions()
    assert d["decision"] == "published" and d["recovered"]
    assert len((config.GATE_DIR / "notices.jsonl").read_text().splitlines()) == 1


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


def test_a_change_between_selection_and_evaluation_is_kept(canon_foo, monkeypatch):
    real, calls = gate.select.check_change, []

    def edit_during_selection(intent, baseline, body, project_root=None, base_dir=None):
        calls.append(1)
        if len(calls) == 1:  # classify's check, before evaluate freezes the baseline
            (canon_foo / "SKILL.md").write_text(BASE + "Hand edit.\n")
        return real(intent, baseline, body, project_root, base_dir)
    monkeypatch.setattr(gate.select, "check_change", edit_during_selection)
    _queue_patch()
    assert gate.run_once(Fake())["decision"] == "published"
    live = (release.skills_dir() / "foo" / "SKILL.md").read_text()
    assert "Hand edit." in live and "Prefer ISO 8601." in live  # derived from the snapshot it was checked on


def test_a_notice_survives_a_torn_tail(canon_foo):
    config.GATE_DIR.mkdir(parents=True, exist_ok=True)
    (config.GATE_DIR / "notices.jsonl").write_text('{"id": "old", "te')  # crash mid-append earlier
    _queue_patch()
    gate.run_once(Fake())
    ids = [json.loads(line)["id"] for line in (config.GATE_DIR / "notices.jsonl").read_text().splitlines()
           if line.startswith('{"id"') and line.endswith("}")]
    assert len(ids) == 1  # the new notice is its own readable line


def test_a_failure_after_the_switch_is_finalized_as_published_not_error(canon_foo, monkeypatch):
    monkeypatch.setattr(config, "GATE_MAX_ATTEMPTS", 1)
    real, calls = gate._notify, []

    def notify_fails_once(event_id, text):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("disk full")
        return real(event_id, text)
    monkeypatch.setattr(gate, "_notify", notify_fails_once)
    _queue_patch()
    assert gate.run_once(Fake()) is None  # committed; finalization deferred, no attempt counted
    gate.run_once(Fake(secret="never-in-a-prompt"))
    assert [d["decision"] for d in _decisions()] == ["published"]
    assert len((config.GATE_DIR / "notices.jsonl").read_text().splitlines()) == 1


def test_a_dangling_reference_in_the_candidate_never_reaches_the_model(canon_foo):
    _queue_patch(new="Use strftime.\nSee references/missing.md.\n")
    fake = Fake()
    gate.run_once(fake)
    assert fake.calls == [] and _decisions()[0]["reason"].startswith("invalid:")


def test_a_rollback_notice_lost_to_a_crash_is_added_by_the_next_pass(canon_foo, monkeypatch):
    _queue_patch()
    gate.run_once(Fake())
    with monkeypatch.context() as m:
        m.setattr(release.notices, "notify", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt))
        with pytest.raises(KeyboardInterrupt):
            release.rollback("foo")
    gate.run_once(Fake(secret="never-in-a-prompt"))
    texts = [json.loads(line)["text"] for line in (config.GATE_DIR / "notices.jsonl").read_text().splitlines()]
    assert texts == ["patch foo (from claude)", "rollback foo (from claude)"]

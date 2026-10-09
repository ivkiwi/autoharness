"""The shared canon gate, phase 1 without a model: queue, eligibility, releases, rollback, recovery."""
import json
import os

import pytest

from autoharness import config
from autoharness.canon import queue, release, select
from autoharness.hook import promoter
from autoharness.lib import intent_queue, layer

BODY = "---\nname: {name}\ndescription: Use when formatting a date.\n---\nUse strftime.\n"


def _skill(root, name, text=None):
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(text or BODY.format(name=name))
    return d


def _candidate(tmp_path, name, text):
    d = tmp_path / "cand" / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(text)
    return d


# --- queue -------------------------------------------------------------------------------------------

def test_queue_keeps_provenance_and_drops_a_replayed_event():
    intent = {"action": "create", "name": "foo"}
    prov = {"session_id": "s", "transcript_path": "/t.jsonl", "range": [0, 10], "window_sha256": "w"}
    queue.append("r1", [intent], [{"ok": True}], provenance=prov, project_root="/p")
    queue.append("r1", [intent], [{"ok": True}], provenance=prov, project_root="/p")  # crash replay
    queue.append("r1", [intent], [{"ok": True}], provenance={**prov, "range": [10, 20]})
    rows = queue.read()
    assert len(rows) == 2  # same intent, different transcript range: two events
    assert rows[0]["provenance"]["harness"] == "claude"
    assert rows[0]["provenance"]["redaction_version"] == queue.redaction_version()
    assert rows[0]["project_root"] == "/p"


def test_queue_read_skips_a_torn_line():
    queue.append("r1", [{"action": "create", "name": "a"}], [{"ok": True}])
    with (config.GATE_DIR / "queue" / "r1.jsonl").open("a") as f:
        f.write('{"id": "x", "intent": {"act')  # crash mid-append
    queue.append("r1", [{"action": "create", "name": "b"}], [{"ok": True}])
    assert [r["intent"]["name"] for r in queue.read()] == ["a", "b"]


def test_drain_queues_with_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROPOSE_ONLY", True)
    roots = {layer.GLOBAL: tmp_path / "g", layer.PROJECT: tmp_path / "p"}
    intent = {"action": "create", "name": "foo", "body": BODY.format(name="foo"), "level": "project",
              "reason": "repeat", "evidence": "e"}
    intent_queue.append("run1", intent, roots[layer.PROJECT])
    promoter.drain("run1", roots=roots, provenance={"kind": "reflector", "session_id": "s", "range": [3, 9]})
    [row] = queue.read()
    assert row["provenance"]["range"] == [3, 9] and row["provenance"]["kind"] == "reflector"
    assert row["intent"] == intent and row["verdict"]["proposed"]


# --- releases ----------------------------------------------------------------------------------------

def test_first_publish_moves_a_real_canon_dir_out_and_links_a_release(tmp_path):
    skills = release.skills_dir()
    base = _skill(skills, "foo")
    base_sha = release.tree_sha256(base)
    cand = _candidate(tmp_path, "foo", BODY.format(name="foo") + "Prefer ISO 8601.\n")

    entry = release.publish("foo", cand, expect_sha=base_sha, event_id="e1")

    assert (skills / "foo").is_symlink()
    assert (skills / "foo" / "SKILL.md").read_text().endswith("Prefer ISO 8601.\n")
    assert release.tree_sha256(release.releases_dir() / "foo" / base_sha) == base_sha  # baseline frozen
    assert os.path.isdir(entry["displaced"])  # the original dir is kept, not deleted
    assert entry["prev"].endswith(base_sha)


def test_publish_refuses_a_changed_baseline(tmp_path):
    _skill(release.skills_dir(), "foo")
    cand = _candidate(tmp_path, "foo", BODY.format(name="foo") + "x\n")
    with pytest.raises(release.Conflict):
        release.publish("foo", cand, expect_sha="not-the-live-tree", event_id="e1")
    assert not (release.skills_dir() / "foo").is_symlink()
    assert not any(e["op"] == "prepared" for e in release.journal_entries())


def test_rollback_returns_to_the_previous_release_and_removes_a_create(tmp_path):
    skills = release.skills_dir()
    base_sha = release.tree_sha256(_skill(skills, "foo"))
    release.publish("foo", _candidate(tmp_path, "foo", BODY.format(name="foo") + "v2\n"),
                    expect_sha=base_sha, event_id="e1")
    release.rollback("foo")
    assert release.current("foo")[2] == base_sha

    release.publish("bar", _candidate(tmp_path, "bar", BODY.format(name="bar")), expect_sha=None, event_id="e2")
    assert release.current("bar")[0] == "link"
    release.rollback("bar")
    assert release.current("bar")[0] == "absent"


def test_second_publish_over_a_link_and_rollback_to_the_first(tmp_path):
    skills = release.skills_dir()
    base_sha = release.tree_sha256(_skill(skills, "foo"))
    first = release.publish("foo", _candidate(tmp_path / "1", "foo", BODY.format(name="foo") + "v2\n"),
                            expect_sha=base_sha, event_id="e1")
    second = release.publish("foo", _candidate(tmp_path / "2", "foo", BODY.format(name="foo") + "v3\n"),
                             expect_sha=first["new_sha"], event_id="e2")
    assert second["prev"] == first["new"]
    release.rollback("foo")
    assert release.current("foo")[2] == first["new_sha"]


@pytest.mark.parametrize("crash_at", ["move", "point"])
def test_recover_restores_the_canon_dir_after_a_crash_mid_switch(tmp_path, monkeypatch, crash_at):
    skills = release.skills_dir()
    base = _skill(skills, "foo")
    base_sha = release.tree_sha256(base)
    cand = _candidate(tmp_path, "foo", BODY.format(name="foo") + "v2\n")
    real_rename = os.rename

    def boom(*a, **k):
        raise KeyboardInterrupt  # a kill, not an error the code could handle

    with monkeypatch.context() as m:  # undo only this patch: the private gate from conftest must stay
        if crash_at == "move":
            # dies right after the canon dir leaves skills/, before the release link replaces it
            m.setattr(release.os, "rename",
                      lambda src, dst: (real_rename(src, dst), "displaced" in str(dst) and boom()))
        else:
            m.setattr(release, "_point", boom)
        with pytest.raises(KeyboardInterrupt):
            release.publish("foo", cand, expect_sha=base_sha, event_id="e1")

    assert release.recover() == ["e1"]
    assert (skills / "foo").is_dir() and not (skills / "foo").is_symlink()
    assert release.tree_sha256(skills / "foo") == base_sha
    assert release.journal_entries()[-1]["op"] == "aborted"


def test_recover_commits_a_switch_that_completed_before_the_crash(tmp_path, monkeypatch):
    base_sha = release.tree_sha256(_skill(release.skills_dir(), "foo"))
    real_record = release._record

    def record(entry):
        if entry["op"] == "committed":
            raise KeyboardInterrupt
        real_record(entry)

    monkeypatch.setattr(release, "_record", record)
    with pytest.raises(KeyboardInterrupt):
        release.publish("foo", _candidate(tmp_path, "foo", BODY.format(name="foo") + "v2\n"),
                        expect_sha=base_sha, event_id="e1")
    monkeypatch.setattr(release, "_record", real_record)
    assert release.recover() == ["e1"]
    assert release.journal_entries()[-1]["op"] == "committed"
    assert release.current("foo")[0] == "link"


# --- eligibility -------------------------------------------------------------------------------------

def _row(**intent):
    intent = {"action": "create", "name": "foo", "level": "global", "body": BODY.format(name="foo"),
              "reason": "r", "evidence": "e", **intent}
    return {"intent": intent, "verdict": {"ok": True, "findings": []}}


def test_eligible_global_create():
    assert select.classify(_row()) == ("eligible", BODY.format(name="foo"))


@pytest.mark.parametrize("change,reason", [
    ({"level": "project"}, "scope"),
    ({"files": {"scripts/x.sh": "x"}}, "files"),
    ({"action": "delete"}, "action:delete"),
    ({"body": BODY.format(name="foo") + "Then run git clean -fdx.\n"}, "authority"),
])
def test_out_of_phase(change, reason):
    assert select.classify(_row(**change)) == ("out_of_phase", reason)


def test_create_needs_a_name_free_in_every_harness():
    _skill(config.HOST_SKILL_ROOTS[0], "foo")
    assert select.classify(_row()) == ("out_of_phase", "name_taken")


def test_vendored_and_denied_targets_are_protected():
    _skill(release.skills_dir(), "herdr")
    (config.CANON_ROOT / ".skill-lock.json").write_text(json.dumps({"skills": {"herdr": {}}}))
    patch = _row(action="patch", name="herdr", old_string="Use strftime.", new_string="Use strftime -u.")
    assert select.classify(patch) == ("out_of_phase", "protected")
    config.GATE_DIR.mkdir(parents=True, exist_ok=True)
    (config.GATE_DIR / "policy.json").write_text(json.dumps({"deny_targets": ["foo"]}))
    assert select.classify(_row()) == ("out_of_phase", "protected")


def test_patch_of_a_canon_skill_rejected_as_not_self_produced_is_taken_over():
    _skill(release.skills_dir(), "foo")
    row = _row(action="patch", old_string="Use strftime.", new_string="Use strftime with a format.")
    row["verdict"] = {"ok": False, "findings": [["self_produced", "target was not produced by the agent"]]}
    decision, body = select.classify(row)
    assert decision == "eligible" and "with a format" in body


def test_patch_needs_a_live_canon_target_and_a_clean_delta():
    assert select.classify(_row(action="patch", old_string="a", new_string="b")) == ("out_of_phase", "not_canon")
    _skill(release.skills_dir(), "foo")
    decision, reason = select.classify(_row(action="patch", old_string="absent text", new_string="b"))
    assert decision == "out_of_phase" and reason.startswith("delta:")


def test_other_rejections_stay_out():
    row = _row()
    row["verdict"] = {"ok": False, "findings": [["safety", "destructive"]]}
    assert select.classify(row) == ("out_of_phase", "rejected:safety")
    row["verdict"] = {"ok": False, "findings": [["routing", "global layer is disabled by AUTOHARNESS_DISABLE_GLOBAL"]]}
    assert select.classify(row)[0] == "eligible"

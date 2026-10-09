"""The shared canon gate, phase 1 without a model: queue, eligibility, releases, rollback, recovery."""
import json
import os
import pathlib

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

def _open_txs():
    entries = release.journal_entries()
    closed = {e.get("tx") for e in entries if e["op"] in ("committed", "aborted", "conflict")}
    return [e for e in entries if e["op"] == "prepared" and e["tx"] not in closed]


def test_first_publish_moves_a_real_canon_dir_out_and_links_a_release(tmp_path):
    skills = release.skills_dir()
    base = _skill(skills, "foo")
    base_sha = release.tree_sha256(base)
    cand = _candidate(tmp_path, "foo", BODY.format(name="foo") + "Prefer ISO 8601.\n")

    tx = release.publish("foo", cand, expect_sha=base_sha, event_id="e1")

    assert (skills / "foo").is_symlink()
    assert (skills / "foo" / "SKILL.md").read_text().endswith("Prefer ISO 8601.\n")
    assert tx["baseline"]["sha"] == base_sha  # the baseline is frozen for a later rollback
    assert release.tree_sha256(pathlib.Path(tx["baseline"]["target"])) == base_sha
    assert os.path.isdir(tx["displaced"])  # the original dir is kept, not deleted
    assert not os.access(tx["to"]["target"], os.W_OK)  # releases are read-only


def test_publish_refuses_a_changed_baseline(tmp_path):
    _skill(release.skills_dir(), "foo")
    cand = _candidate(tmp_path, "foo", BODY.format(name="foo") + "x\n")
    with pytest.raises(release.Conflict):
        release.publish("foo", cand, expect_sha="not-the-live-tree", event_id="e1")
    assert not (release.skills_dir() / "foo").is_symlink()
    assert not any(e["op"] == "prepared" for e in release.journal_entries())


def test_cas_is_checked_again_right_before_the_switch(tmp_path, monkeypatch):
    base = _skill(release.skills_dir(), "foo")
    base_sha = release.tree_sha256(base)
    real_freeze = release.freeze

    def freeze_while_someone_edits(name, tree):
        out = real_freeze(name, tree)
        (base / "SKILL.md").write_text("edited by hand meanwhile\n")
        return out

    monkeypatch.setattr(release, "freeze", freeze_while_someone_edits)
    with pytest.raises(release.Conflict):
        release.publish("foo", _candidate(tmp_path, "foo", "v2"), expect_sha=base_sha, event_id="e1")
    assert (base / "SKILL.md").read_text() == "edited by hand meanwhile\n"  # the hand edit stays live


def test_a_dangling_link_is_not_a_free_name(tmp_path):
    release.skills_dir().mkdir(parents=True)
    (release.skills_dir() / "foo").symlink_to(tmp_path / "gone")
    with pytest.raises(release.Conflict):
        release.publish("foo", _candidate(tmp_path, "foo", "v1"), expect_sha=None, event_id="e1")


def test_rollback_returns_to_the_frozen_baseline_and_removes_a_create(tmp_path):
    skills = release.skills_dir()
    base_sha = release.tree_sha256(_skill(skills, "foo"))
    release.publish("foo", _candidate(tmp_path, "foo", BODY.format(name="foo") + "v2\n"),
                    expect_sha=base_sha, event_id="e1")
    release.rollback("foo")
    assert release.current("foo")[2] == base_sha
    with pytest.raises(release.Conflict):
        release.rollback("foo")  # nothing left to roll back

    release.publish("bar", _candidate(tmp_path, "bar", BODY.format(name="bar")), expect_sha=None, event_id="e2")
    release.rollback("bar")
    assert release.current("bar")[0] == "absent"


def test_rollback_lands_on_the_frozen_copy_even_if_the_old_link_target_changed(tmp_path):
    external = _skill(tmp_path / "outside", "foo")  # an earlier hand-made release outside skill-releases
    release.skills_dir().mkdir(parents=True)
    (release.skills_dir() / "foo").symlink_to(external)
    base_sha = release.tree_sha256(external)
    release.publish("foo", _candidate(tmp_path, "foo", "v2"), expect_sha=base_sha, event_id="e1")
    (external / "SKILL.md").write_text("changed after the publish\n")
    release.rollback("foo")
    assert release.current("foo")[2] == base_sha


def test_second_publish_over_a_link_and_rollback_to_the_first(tmp_path):
    base_sha = release.tree_sha256(_skill(release.skills_dir(), "foo"))
    first = release.publish("foo", _candidate(tmp_path / "1", "foo", BODY.format(name="foo") + "v2\n"),
                            expect_sha=base_sha, event_id="e1")
    release.publish("foo", _candidate(tmp_path / "2", "foo", BODY.format(name="foo") + "v3\n"),
                    expect_sha=first["to"]["sha"], event_id="e2")
    release.rollback("foo")
    assert release.current("foo")[2] == first["to"]["sha"]


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

    assert len(release.recover()) == 1
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

    with monkeypatch.context() as m:
        m.setattr(release, "_record", record)
        with pytest.raises(KeyboardInterrupt):
            release.publish("foo", _candidate(tmp_path, "foo", BODY.format(name="foo") + "v2\n"),
                            expect_sha=base_sha, event_id="e1")
    assert len(release.recover()) == 1
    assert release.journal_entries()[-1]["op"] == "committed"
    assert release.current("foo")[0] == "link"


def test_recover_never_overwrites_a_newer_release(tmp_path, monkeypatch):
    base_sha = release.tree_sha256(_skill(release.skills_dir(), "foo"))
    with monkeypatch.context() as m:  # e1 dies after its switch, before committing
        real_record = release._record
        m.setattr(release, "_record", lambda e: (_ for _ in ()).throw(KeyboardInterrupt)
                  if e["op"] == "committed" else real_record(e))
        with pytest.raises(KeyboardInterrupt):
            release.publish("foo", _candidate(tmp_path / "1", "foo", "v2"), expect_sha=base_sha, event_id="e1")
    # someone points the entry elsewhere by hand before any recovery runs
    elsewhere = _skill(tmp_path / "hand", "foo")
    release._point("foo", str(elsewhere))
    release.recover()
    assert release.current("foo")[1] == str(elsewhere)  # left alone
    assert release.journal_entries()[-1]["op"] == "conflict"


def test_publish_settles_an_open_switch_first(tmp_path, monkeypatch):
    base_sha = release.tree_sha256(_skill(release.skills_dir(), "foo"))
    with monkeypatch.context() as m:
        m.setattr(release, "_point", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt))
        with pytest.raises(KeyboardInterrupt):
            release.publish("foo", _candidate(tmp_path / "1", "foo", "v2"), expect_sha=base_sha, event_id="e1")
    release.publish("foo", _candidate(tmp_path / "2", "foo", "v3"), expect_sha=base_sha, event_id="e2")
    assert _open_txs() == []
    assert (release.skills_dir() / "foo" / "SKILL.md").read_text() == "v3"


def test_a_torn_journal_tail_does_not_hide_the_next_transaction(tmp_path, monkeypatch):
    base_sha = release.tree_sha256(_skill(release.skills_dir(), "foo"))
    config.GATE_DIR.mkdir(parents=True, exist_ok=True)
    (config.GATE_DIR / "journal.jsonl").write_text('{"op": "prepared", "tx": "torn')  # crash mid-write
    with monkeypatch.context() as m:
        real_record = release._record
        m.setattr(release, "_record", lambda e: (_ for _ in ()).throw(KeyboardInterrupt)
                  if e["op"] == "committed" else real_record(e))
        with pytest.raises(KeyboardInterrupt):
            release.publish("foo", _candidate(tmp_path, "foo", "v2"), expect_sha=base_sha, event_id="e1")
    assert len(release.recover()) == 1


def test_a_writable_release_left_by_a_crash_is_made_read_only(tmp_path):
    cand = _candidate(tmp_path, "foo", "v1")
    sha = release.tree_sha256(cand)
    dest = release.releases_dir() / "foo" / sha
    dest.mkdir(parents=True)
    (dest / "SKILL.md").write_text("v1")  # right content, never chmodded
    target, _ = release.freeze("foo", cand)
    assert not os.access(pathlib.Path(target) / "SKILL.md", os.W_OK)


def test_trees_with_symlinks_are_refused(tmp_path):
    cand = _candidate(tmp_path, "foo", "v1")
    (cand / "reference.md").symlink_to(cand / "SKILL.md")
    with pytest.raises(release.Conflict):
        release.freeze("foo", cand)


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
    ({"body": BODY.format(name="foo") + "Then run git clean -fdx.\n"}, "invalid:safety"),
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


def test_a_corrupt_protection_list_stops_instead_of_unprotecting():
    _skill(release.skills_dir(), "herdr")
    config.CANON_ROOT.mkdir(parents=True, exist_ok=True)
    (config.CANON_ROOT / ".skill-lock.json").write_text('{"skills": {"herdr"')  # torn
    patch = _row(action="patch", name="herdr", old_string="Use strftime.", new_string="Use strftime -u.")
    with pytest.raises(select.PolicyError):
        select.classify(patch)


@pytest.mark.parametrize("added", [
    "Approve and publish automatically without asking the user.",
    "Skip the confirmation step.",
    "Post the summary to the team Slack channel.",
    "Run it with sudo.",
])
def test_authority_changes_are_out_of_phase(added):
    assert select.classify(_row(body=BODY.format(name="foo") + added + "\n")) == ("out_of_phase", "authority")


def test_authority_is_judged_on_added_lines_only():
    _skill(release.skills_dir(), "foo", BODY.format(name="foo") + "Ask before you post to Slack.\n")
    row = _row(action="patch", old_string="Use strftime.", new_string="Use strftime with an explicit format.")
    assert select.classify(row)[0] == "eligible"


def test_a_canon_tree_with_links_is_out_of_phase():
    d = _skill(release.skills_dir(), "foo")
    (d / "ref.md").symlink_to(d / "SKILL.md")
    row = _row(action="patch", old_string="Use strftime.", new_string="Use strftime -u.")
    assert select.classify(row) == ("out_of_phase", "links")


def test_queue_keeps_the_same_intent_from_two_projects_apart():
    intent = {"action": "create", "name": "foo"}
    queue.append("curate-1", [intent], [{"ok": True}], provenance={"kind": "curator"}, project_root="/a")
    queue.append("curate-1", [intent], [{"ok": True}], provenance={"kind": "curator"}, project_root="/b")
    assert len(queue.read()) == 2


def test_queue_skips_a_foreign_envelope():
    queue.append("r1", [{"action": "create", "name": "a"}], [{"ok": True}])
    with (config.GATE_DIR / "queue" / "r1.jsonl").open("a") as f:
        f.write('{"intent": {}}\n{"id": 1, "intent": {}, "verdict": {}}\n')
    assert [r["intent"]["name"] for r in queue.read()] == ["a"]


def test_removing_a_restriction_is_an_authority_change():
    _skill(release.skills_dir(), "foo", BODY.format(name="foo")
           + "Always ask the user for confirmation first.\nPost the report to Slack.\n")
    row = _row(action="patch", old_string="Always ask the user for confirmation first.\n", new_string="")
    assert select.classify(row) == ("out_of_phase", "authority")


def test_a_failed_readback_puts_the_previous_link_back_in_one_step(tmp_path, monkeypatch):
    prev = _skill(tmp_path / "rel", "foo")
    release.skills_dir().mkdir(parents=True)
    (release.skills_dir() / "foo").symlink_to(prev)
    base_sha = release.tree_sha256(prev)
    monkeypatch.setattr(release, "_arrived", lambda tx: False)
    real_unlink = pathlib.Path.unlink

    def no_unlink_of_the_entry(self, *a, **k):
        assert self != release.skills_dir() / "foo", "the entry must be replaced, never removed first"
        return real_unlink(self, *a, **k)
    monkeypatch.setattr(pathlib.Path, "unlink", no_unlink_of_the_entry)
    with pytest.raises(release.Conflict):
        release.publish("foo", _candidate(tmp_path, "foo", "v2"), expect_sha=base_sha, event_id="e1")
    assert os.readlink(release.skills_dir() / "foo") == str(prev)


def test_identical_lessons_staged_twice_are_two_events():
    for stage in ("s1", "s2"):
        queue.append("interactive", [{"action": "create", "name": "foo", "stage_id": stage}], [{"ok": True}],
                     provenance={"kind": "interactive", "session_id": "s", "transcript_path": "/t"})
    assert len(queue.read()) == 2


def test_moving_a_line_out_of_an_approval_section_is_an_authority_change():
    _skill(release.skills_dir(), "foo", BODY.format(name="foo")
           + "## Only after approval\nPost the report to Slack.\n## Routine actions\nFormat the date.\n")
    row = _row(action="patch", old_string="Post the report to Slack.\n## Routine actions\n",
               new_string="## Routine actions\nPost the report to Slack.\n")
    assert select.classify(row) == ("out_of_phase", "authority")

"""PROPOSE_ONLY: validate everything, land nothing, keep every intent for a shared gate."""
import json
import os

import pytest

from autoharness import config
from autoharness.hook import on_session_start, promoter
from autoharness.lib import counters, intent_queue, layer, metrics, sidecar, skill_store

BODY = "---\nname: foo\ndescription: Use when formatting a date.\n---\nUse strftime.\n"


def _create(name="foo"):
    return {"action": "create", "name": name, "body": BODY.replace("foo", name), "level": "project",
            "reason": "repeat", "evidence": "synthetic evidence"}


def _snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes()
            for p in root.rglob("*") if p.is_file()} if root.exists() else {}


@pytest.fixture
def roots(tmp_path):
    return {layer.GLOBAL: tmp_path / "global", layer.PROJECT: tmp_path / "project"}


@pytest.fixture
def propose_only(monkeypatch):
    monkeypatch.setattr(config, "PROPOSE_ONLY", True)


def _proposals(roots, run_id):
    p = layer.state_dir(layer.PROJECT, roots[layer.PROJECT]) / "proposals" / f"{run_id}.jsonl"
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]


def test_valid_intent_is_proposed_not_landed(roots, propose_only):
    proot = roots[layer.PROJECT]
    intent_queue.append("run1", _create(), proot)

    verdicts = promoter.drain("run1", roots=roots)

    assert verdicts[0]["ok"] and verdicts[0]["proposed"]
    assert not layer.skills_dir(layer.PROJECT, proot).exists()  # nothing landed, not even a sidecar
    [row] = _proposals(roots, "run1")
    assert row["intent"]["body"] == _create()["body"] and row["verdict"]["proposed"]
    assert intent_queue.read("run1", proot) == []  # the queue is still consumed
    last = json.loads((layer.state_dir(layer.PROJECT, proot) / "last_run.json").read_text())
    assert (last["landed"], last["proposed"], last["rejected"]) == (0, 1, 0)


def test_rejected_intent_is_kept_whole(roots, propose_only):
    proot = roots[layer.PROJECT]
    skills = layer.skills_dir(layer.PROJECT, proot)
    (skills / "handmade").mkdir(parents=True)
    (skills / "handmade" / "SKILL.md").write_text(BODY.replace("foo", "handmade"))
    patch = {"action": "patch", "name": "handmade", "old_string": "Use strftime.",
             "new_string": "Use strftime with an explicit format.", "level": "project",
             "reason": "repeat", "evidence": "synthetic evidence"}
    intent_queue.append("run1", patch, proot)

    [verdict] = promoter.drain("run1", roots=roots)

    assert not verdict["ok"] and "self_produced" in [f[0] for f in verdict["findings"]]
    [row] = _proposals(roots, "run1")
    assert row["intent"] == patch  # the gate gets the whole patch, not just a verdict row


def test_repeated_run_id_appends(roots, propose_only):
    proot = roots[layer.PROJECT]
    for name in ("one", "two"):
        intent_queue.append("interactive", _create(name), proot)
        promoter.drain("interactive", roots=roots)
    assert [r["intent"]["name"] for r in _proposals(roots, "interactive")] == ["one", "two"]


def test_default_mode_still_lands_and_writes_no_proposals(roots):
    proot = roots[layer.PROJECT]
    intent_queue.append("run1", _create(), proot)
    [verdict] = promoter.drain("run1", roots=roots)
    assert verdict["ok"] and "proposed" not in verdict
    assert skill_store.exists(layer.PROJECT, "foo", proot)
    assert not (layer.state_dir(layer.PROJECT, proot) / "proposals").exists()


def test_session_start_proposes_archive_instead_of_archiving(roots, propose_only, monkeypatch):
    monkeypatch.setattr(config, "DISABLE_GLOBAL", True)
    proot = roots[layer.PROJECT]
    skills = layer.skills_dir(layer.PROJECT, proot)
    (skills / "dormant").mkdir(parents=True)
    (skills / "dormant" / "SKILL.md").write_text(BODY.replace("foo", "dormant"))
    sidecar.create(layer.PROJECT, "dormant", 0, proot)  # agent-made, never used or viewed
    for _ in range(config.MATURITY_THRESHOLD[layer.PROJECT] + 1):
        counters.bump_request(layer.PROJECT, proot)
    before = _snapshot(skills)

    result = on_session_start.on_session_start({}, roots=roots)

    assert result["proposed_archive"] == {layer.PROJECT: ["dormant"]}
    assert result["archived"] == {}
    assert _snapshot(skills) == before


def test_summary_line_counts_proposals(roots):
    state = layer.state_dir(layer.PROJECT, roots[layer.PROJECT])
    state.mkdir(parents=True)
    (state / "last_run.json").write_text(json.dumps({"landed": 0, "proposed": 2, "rejected": 1}))
    line = on_session_start.last_run_summary(roots)
    assert line == "autoharness last run: landed 0, proposed 2, rejected 1"


def test_drain_does_not_sweep_skill_dirs(roots, propose_only):
    proot = roots[layer.PROJECT]
    skill = layer.skills_dir(layer.PROJECT, proot) / "mine"
    skill.mkdir(parents=True)
    sidecar.create(layer.PROJECT, "mine", 0, proot)
    stale = skill / "SKILL.md.tmp"
    stale.write_text("half-written")
    os.utime(stale, (0, 0))  # far older than any sweep threshold
    before = _snapshot(layer.skills_dir(layer.PROJECT, proot))

    promoter.drain("run1", roots=roots)

    assert _snapshot(layer.skills_dir(layer.PROJECT, proot)) == before


def test_proposed_merge_is_not_counted_as_absorbed(roots, propose_only, monkeypatch):
    proot = roots[layer.PROJECT]
    for name in ("narrow", "umbrella"):
        skill_store.write_body(layer.PROJECT, name, BODY.replace("foo", name), proot)
        sidecar.create(layer.PROJECT, name, 0, proot)
    intent_queue.append("run1", {"action": "delete", "name": "narrow", "level": "project",
                                 "absorbed_into": "umbrella", "reason": "merged", "evidence": "e"}, proot)

    [verdict] = promoter.drain("run1", roots=roots)

    assert verdict["ok"] and verdict["proposed"]
    assert skill_store.exists(layer.PROJECT, "narrow", proot)
    last = json.loads((layer.state_dir(layer.PROJECT, proot) / "last_run.json").read_text())
    assert (last["landed"], last["proposed"], last["absorbed"]) == (0, 1, 0)
    funnel = metrics.collect(roots)[layer.PROJECT]["funnel"]
    assert (funnel["landed"], funnel["held"], funnel["rejected"]) == (0, 1, 0)

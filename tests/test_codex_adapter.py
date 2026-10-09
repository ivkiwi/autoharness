"""Codex adapter: the same dispatcher and promoter, Codex roots, carrier and digest. Fixture-driven,
no live Codex. Payload shapes mirror what `codex exec` 0.162.0 sent to a probe hook on 2026-10-09:
Claude field names, the shell reported as `Bash`, `transcript_path` null under --ephemeral, no
agent_type; SessionStart/Stop/PreToolUse/PostToolUse/SessionEnd all fired."""
import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoharness import config
from autoharness.canon import queue as canon_queue
from autoharness.hook import capture, dispatch, promoter, spawn
from autoharness.lib import counters, intent_queue, layer, tails

SID = "01a1226a-19c0-7c43-a585-c312f4926fc1"
GOOD = "---\nname: {n}\ndescription: Use when {d}.\n---\n# {n}\nbody\n"


@pytest.fixture
def codex(monkeypatch):
    monkeypatch.delenv(config.CHILD_SESSION_ENV, raising=False)
    # raising=False: the mutation check runs this file against sources without the adapter, and a
    # test must then fail on its own assertion, not in fixture setup
    monkeypatch.setattr(layer, "HARNESS", "codex", raising=False)
    monkeypatch.setattr(config, "HARNESS", "codex", raising=False)  # the queue stamps rows with this side
    monkeypatch.setattr(config, "DISABLE_GLOBAL", True)  # the phase-1 deployment; also keeps ~/.codex out of the test
    monkeypatch.setattr(config, "INDEX_ROOTS", "", raising=False)


@pytest.fixture
def repo(tmp_path):
    d = tmp_path / "repo"
    d.mkdir()
    return d


def _rollout(path, source="cli"):
    path.write_text(json.dumps({"type": "session_meta", "payload": {"id": SID, "source": source}}) + "\n",
                    encoding="utf-8")
    return str(path)


def _ev(name, cwd, transcript, sid=SID, **extra):
    ev = {"hook_event_name": name, "session_id": sid, "cwd": str(cwd), "transcript_path": transcript,
          "model": "gpt-6.1-sol", "permission_mode": "bypassPermissions"}
    ev.update(extra)
    return ev


def _tool(cwd, transcript, i=0, sid=SID, tool="Bash"):
    return dispatch.dispatch(_ev("PreToolUse", cwd, transcript, sid=sid, tool_name=tool,
                                 tool_input={"command": "ls"}, tool_use_id=f"exec-{i}", turn_id="t1"))


def _stop(cwd, transcript, sid=SID, reflect=None):
    return dispatch.dispatch(_ev("Stop", cwd, transcript, sid=sid, turn_id="t1", stop_hook_active=False,
                                 last_assistant_message="ok"), reflect=reflect or (lambda *a: None))


def _state(repo):
    return repo / ".codex" / "autoharness"


def _confirm_reflection(transcript, repo, tmp_path, *, monkeypatch=None, child=None):
    """Run spawn.main the way the detached launch does, with a stand-in child (no codex)."""
    import unittest.mock as mock
    fake = child or (lambda argv, env, payload: SimpleNamespace(returncode=0, stderr=""))
    with mock.patch.object(spawn, "_detached_spawn", fake):
        return spawn.main([str(transcript), SID, "run-1", str(repo / ".codex"), str(tmp_path / "g")])


# --- roots -------------------------------------------------------------------------------------

def test_default_roots_follow_the_harness(codex, repo, monkeypatch):
    assert layer.default_root(layer.GLOBAL) == Path.home() / ".codex"
    assert layer.default_root(layer.PROJECT, cwd=str(repo)) == repo / ".codex"
    monkeypatch.setattr(layer, "HARNESS", "claude")
    assert layer.default_root(layer.GLOBAL) == Path.home() / ".claude"


# --- acceptance: a normal Codex turn ----------------------------------------------------------

def test_normal_codex_turn_counts_under_codex_root_and_reflects(codex, repo, tmp_path):
    transcript = _rollout(tmp_path / "rollout.jsonl")
    seen = []
    for i in range(config.REFLECT_EVERY_N):
        _tool(repo, transcript, i)
    out = _stop(repo, transcript, reflect=lambda ev, res, roots: seen.append((res, roots)))

    assert out["result"]["triggered"] and out["result"]["window_n"] == config.REFLECT_EVERY_N
    [(res, roots)] = seen
    assert res["session_id"] == SID and roots[layer.PROJECT] == repo / ".codex"
    assert counters.session_count(SID, repo / ".codex") == 0  # reset on trigger
    assert counters.request_count(layer.PROJECT, repo / ".codex") == 1
    assert not (repo / ".claude").exists()  # the Codex adapter never writes into Claude's state
    [note] = tails.pending(repo / ".codex")  # kept until spawn.main confirms the window was fed
    assert (note["count"], note["offset"]) == (config.REFLECT_EVERY_N, 0)


# --- acceptance: fewer than N calls, then the session ends ------------------------------------

def test_short_session_keeps_a_tail_note_until_session_end_flushes(codex, repo, tmp_path):
    transcript = _rollout(tmp_path / "rollout.jsonl")
    for i in range(3):
        _tool(repo, transcript, i)
    assert not _stop(repo, transcript)["result"]["triggered"]

    [note] = tails.pending(repo / ".codex")
    assert (note["session_id"], note["transcript_path"], note["count"]) == (SID, transcript, 3)
    assert "coverage_gap" not in note

    seen = []
    out = dispatch.dispatch(_ev("SessionEnd", repo, transcript, reason="other"),
                            reflect=lambda ev, res, roots: seen.append(res))
    assert out["result"]["triggered"] and seen[0]["count"] == 3  # the tail still reflects, through SessionEnd
    assert tails.pending(repo / ".codex")[0]["count"] == 3  # and stays noted until that reflection ran
    _confirm_reflection(transcript, repo, tmp_path, monkeypatch=None)
    assert tails.pending(repo / ".codex") == []


def test_stop_below_cadence_overwrites_the_note_with_the_latest_count(codex, repo, tmp_path):
    transcript = _rollout(tmp_path / "rollout.jsonl")
    _tool(repo, transcript, 0)
    _stop(repo, transcript)
    _tool(repo, transcript, 1)
    _stop(repo, transcript)
    [note] = tails.pending(repo / ".codex")
    assert note["count"] == 2


# --- acceptance: transcript_path = null -------------------------------------------------------

def test_null_transcript_records_a_coverage_gap_instead_of_spawning(codex, repo):
    launched = []
    for i in range(config.REFLECT_EVERY_N):
        _tool(repo, None, i)
    out = _stop(repo, None, reflect=lambda ev, res, roots: dispatch._reflect(ev, res, roots, launch=lambda *a: launched.append(a)))

    assert out["result"]["triggered"] and launched == []
    [note] = tails.pending(repo / ".codex")
    assert note["coverage_gap"] == "no_transcript_path" and note["count"] == config.REFLECT_EVERY_N
    assert note["transcript_path"] is None


def test_null_transcript_below_cadence_still_notes_the_tail(codex, repo):
    _tool(repo, None, 0)
    _stop(repo, None)
    [note] = tails.pending(repo / ".codex")
    assert note["transcript_path"] is None and note["count"] == 1


# --- acceptance: compaction -------------------------------------------------------------------

def test_compaction_events_do_not_touch_the_counter(codex, repo, tmp_path):
    transcript = _rollout(tmp_path / "rollout.jsonl")
    _tool(repo, transcript, 0)
    for name in ("PreCompact", "PostCompact"):
        assert dispatch.dispatch(_ev(name, repo, transcript, trigger="auto"))["ignored"] is True
    assert counters.session_count(SID, repo / ".codex") == 1


# --- acceptance: a repeated Stop --------------------------------------------------------------

def test_repeated_stop_never_reflects_twice(codex, repo, tmp_path):
    transcript = _rollout(tmp_path / "rollout.jsonl")
    seen = []
    for i in range(config.REFLECT_EVERY_N):
        _tool(repo, transcript, i)
    _stop(repo, transcript, reflect=lambda ev, res, roots: seen.append(res))
    again = _stop(repo, transcript, reflect=lambda ev, res, roots: seen.append(res))
    assert len(seen) == 1 and not again["result"]["triggered"]
    assert tails.pending(repo / ".codex")[0]["count"] == config.REFLECT_EVERY_N  # a quiet Stop touches no note


def test_tail_survives_a_failed_launch_and_clears_after_a_confirmed_reflection(codex, repo, tmp_path):
    transcript = Path(_rollout(tmp_path / "rollout.jsonl"))
    for i in range(config.REFLECT_EVERY_N):
        _tool(repo, str(transcript), i)
    _stop(repo, str(transcript))  # triggered: the counter is reset, the note carries the window
    assert tails.pending(repo / ".codex")[0]["count"] == config.REFLECT_EVERY_N

    def crashing(argv, env, payload):
        raise OSError("codex: not found")

    with pytest.raises(OSError):
        _confirm_reflection(transcript, repo, tmp_path, child=crashing)
    assert tails.pending(repo / ".codex")[0]["offset"] == 0  # nothing fed: the note and the watermark stay
    assert counters.session_offset(SID, repo / ".codex") == 0

    _confirm_reflection(transcript, repo, tmp_path)
    assert tails.pending(repo / ".codex") == []
    assert counters.session_offset(SID, repo / ".codex") == transcript.stat().st_size


def test_failed_child_consumes_nothing(codex, repo, tmp_path):
    transcript = Path(_rollout(tmp_path / "rollout.jsonl"))
    for i in range(config.REFLECT_EVERY_N):
        _tool(repo, str(transcript), i)
    _stop(repo, str(transcript))
    refused = lambda argv, env, payload: SimpleNamespace(returncode=2, stderr="MCP tool call requires approval")  # noqa: E731

    _confirm_reflection(transcript, repo, tmp_path, child=refused)  # returns, the crash is in the run account
    assert counters.session_offset(SID, repo / ".codex") == 0  # the window is still unconsumed
    assert tails.pending(repo / ".codex")[0]["count"] == config.REFLECT_EVERY_N
    assert "spawn_error" in json.loads((_state(repo) / "runs" / "run-1.json").read_text(encoding="utf-8"))

    _confirm_reflection(transcript, repo, tmp_path)
    assert counters.session_offset(SID, repo / ".codex") == transcript.stat().st_size
    assert tails.pending(repo / ".codex") == []


def test_reflection_settles_only_the_note_it_was_launched_from(codex, repo, tmp_path):
    transcript = Path(_rollout(tmp_path / "rollout.jsonl"))
    for i in range(config.REFLECT_EVERY_N):
        _tool(repo, str(transcript), i)
    _stop(repo, str(transcript))  # note A, the one the reflection answers for

    def child_during_which_a_stop_happens(argv, env, payload):
        transcript.write_text(transcript.read_text(encoding="utf-8") + '{"type": "event_msg"}\n', encoding="utf-8")
        _tool(repo, str(transcript), 99)
        _stop(repo, str(transcript))  # note B: newer activity, written while the child runs
        return SimpleNamespace(returncode=0, stderr="")

    size_before = transcript.stat().st_size
    _confirm_reflection(transcript, repo, tmp_path, child=child_during_which_a_stop_happens)
    [note] = tails.pending(repo / ".codex")  # B survives A's completion
    assert note["count"] == 1
    assert note["offset"] == size_before == counters.session_offset(SID, repo / ".codex")  # B starts where A stopped
    assert note["transcript_path"] == str(transcript)


def test_chronic_child_failure_gives_the_window_up(codex, repo, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REFLECT_MAX_FAILURES", 3, raising=False)
    transcript = Path(_rollout(tmp_path / "rollout.jsonl"))
    for i in range(config.REFLECT_EVERY_N):
        _tool(repo, str(transcript), i)
    _stop(repo, str(transcript))
    refused = lambda argv, env, payload: SimpleNamespace(returncode=2, stderr="MCP tool call requires approval")  # noqa: E731

    for streak in (1, 2):
        _confirm_reflection(transcript, repo, tmp_path, child=refused)
        [note] = tails.pending(repo / ".codex")
        assert note["failures"] == streak and "coverage_gap" not in note
        assert counters.session_offset(SID, repo / ".codex") == 0  # still fed next time

    _confirm_reflection(transcript, repo, tmp_path, child=refused)
    [note] = tails.pending(repo / ".codex")
    assert note["coverage_gap"] == "reflection_failed_3x" and note["failures"] == 0
    assert note["offset"] == transcript.stat().st_size == counters.session_offset(SID, repo / ".codex")  # given up, moved on


def test_tail_note_and_settle_share_one_lock(codex, repo, tmp_path, monkeypatch):
    root = repo / ".codex"
    tails.note(SID, "/t.jsonl", 50, root)  # note A
    version_a = tails.read(SID, root)["version"]
    inside, proceed = threading.Event(), threading.Event()
    real_read = tails._read

    def slow_read(p):  # settle has read A and is about to act on it; hold it there
        data = real_read(p)
        if not inside.is_set():
            inside.set()
            proceed.wait(5)
        return data

    monkeypatch.setattr(tails, "_read", slow_read)
    settling = threading.Thread(target=tails.settle, args=(SID, version_a, 100, root))
    settling.start()
    assert inside.wait(5)
    stop = threading.Thread(target=tails.note, args=(SID, "/t.jsonl", 1, root))  # a Stop during settle: note B
    stop.start()
    stop.join(0.3)
    assert stop.is_alive()  # B waits for the lock instead of slipping in between settle's read and unlink
    proceed.set()
    settling.join(5)
    stop.join(5)
    assert not settling.is_alive() and not stop.is_alive()
    [note] = tails.pending(root)
    assert note["count"] == 1  # A settled, B written after it and kept


def test_session_end_with_nothing_new_keeps_an_earlier_gap_note(codex, repo):
    for i in range(config.REFLECT_EVERY_N):
        _tool(repo, None, i)
    _stop(repo, None, reflect=lambda ev, res, roots: dispatch._reflect(ev, res, roots, launch=lambda *a: None))
    assert tails.pending(repo / ".codex")[0]["coverage_gap"] == "no_transcript_path"
    out = dispatch.dispatch(_ev("SessionEnd", repo, None, reason="other"), reflect=lambda *a: None)
    assert not out["result"]["triggered"]
    assert tails.pending(repo / ".codex")[0]["coverage_gap"] == "no_transcript_path"  # a quiet end erases nothing


# --- acceptance: a reflector child, no recursion ----------------------------------------------

def test_reflector_child_is_guarded_and_cannot_write(codex, repo, tmp_path, monkeypatch):
    monkeypatch.setenv(config.CHILD_SESSION_ENV, "1")
    transcript = _rollout(tmp_path / "rollout.jsonl")
    denied = _tool(repo, transcript, tool="apply_patch")
    assert denied["deny"] is True and denied["reason"]
    assert _tool(repo, transcript, 0)["ignored"] is True  # a child's shell call is not activity
    out = _stop(repo, transcript, reflect=lambda *a: pytest.fail("a child must never reflect"))
    assert out["result"]["reason"] == "recursion_guard"
    assert not _state(repo).exists()  # nothing counted, nothing noted


def test_subagent_rollout_is_ignored_entirely(codex, repo, tmp_path):
    transcript = _rollout(tmp_path / "rollout.jsonl", source={"subagent": {"other": "guardian"}})
    assert _tool(repo, transcript, 0)["reason"] == "subagent_session"
    out = _stop(repo, transcript, reflect=lambda *a: pytest.fail("a subagent must never reflect"))
    assert out["reason"] == "subagent_session"
    assert not _state(repo).exists()


def test_subagent_markers_are_honoured_in_every_handler(codex, repo, tmp_path):
    state = _state(repo)
    state.mkdir(parents=True)
    (state / "last_run.json").write_text('{"landed": 1}', encoding="utf-8")
    transcript = _rollout(tmp_path / "rollout.jsonl")  # a primary-looking rollout, but the payload names a role
    for name, extra in (("SessionStart", {"source": "startup"}), ("PreToolUse", {"tool_name": "Bash", "tool_input": {}}),
                        ("Stop", {}), ("SessionEnd", {"reason": "other"})):
        for path in (None, transcript):
            out = dispatch.dispatch(_ev(name, repo, path, agent_type="worker", **extra), reflect=lambda *a: None)
            assert out == {"ignored": True, "reason": "subagent_session"}, (name, path)
    assert (state / "last_run.json").exists()  # the user's summary line is not consumed by a child
    assert counters.request_count(layer.PROJECT, repo / ".codex") == 0
    assert tails.pending(repo / ".codex") == []


def test_unreadable_or_odd_transcript_header_is_treated_as_primary(codex, repo, tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json\n", encoding="utf-8")
    assert _tool(repo, str(bad), 0)["reason"] == "untracked PreToolUse"  # counted, not dropped as a subagent
    _tool(repo, str(bad), 1)
    assert counters.session_count(SID, repo / ".codex") == 2


# --- acceptance: a parallel Claude turn in the same project -----------------------------------

def test_parallel_claude_turn_keeps_separate_state(codex, repo, tmp_path, monkeypatch):
    transcript = _rollout(tmp_path / "rollout.jsonl")
    _tool(repo, transcript, 0)
    _tool(repo, transcript, 1)
    _stop(repo, transcript)

    monkeypatch.setattr(layer, "HARNESS", "claude")
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(repo))
    claude_sid = "c0ffee00-0000-4000-8000-000000000001"
    dispatch.dispatch({"hook_event_name": "PreToolUse", "session_id": claude_sid, "tool_name": "Bash",
                       "tool_input": {"command": "ls"}, "cwd": str(repo)})
    dispatch.dispatch({"hook_event_name": "Stop", "session_id": claude_sid, "cwd": str(repo),
                       "transcript_path": transcript}, reflect=lambda *a: None)

    assert counters.request_count(layer.PROJECT, repo / ".codex") == 1
    assert counters.request_count(layer.PROJECT, repo / ".claude") == 1
    assert counters.session_count(SID, repo / ".codex") == 2
    assert counters.session_count(claude_sid, repo / ".claude") == 1
    assert not (repo / ".claude" / "autoharness" / "tails").exists()  # tails are a Codex-only note


# --- acceptance: PROPOSE_ONLY under Codex roots -----------------------------------------------

def test_propose_only_queues_into_the_shared_gate_as_codex(codex, repo, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROPOSE_ONLY", True)
    roots = {layer.PROJECT: repo / ".codex", layer.GLOBAL: tmp_path / "g"}
    intent = {"action": "create", "name": "foo", "level": "project", "reason": "r", "evidence": "e",
              "body": GOOD.format(n="foo", d="formatting a date")}
    intent_queue.append("run1", intent, roots[layer.PROJECT])

    [verdict] = promoter.drain("run1", roots=roots)

    assert verdict["ok"] and verdict["proposed"]
    assert not (repo / ".codex" / "skills").exists() and not (repo / ".claude").exists()
    [row] = canon_queue.read()  # one queue for every harness, under GATE_DIR, not under the project
    assert row["intent"]["name"] == "foo" and row["project_root"] == str(repo / ".codex")
    assert row["provenance"]["harness"] == "codex"
    assert not (_state(repo) / "proposals").exists()


def test_codex_reflector_provenance_reaches_the_shared_queue(codex, repo, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROPOSE_ONLY", True)
    transcript = Path(_rollout(tmp_path / "rollout.jsonl"))
    transcript.write_text(transcript.read_text(encoding="utf-8") + json.dumps({"type": "event_msg"}) + "\n",
                          encoding="utf-8")
    roots = {layer.PROJECT: repo / ".codex", layer.GLOBAL: tmp_path / "g"}
    intent = {"action": "create", "name": "foo", "level": "project", "reason": "r", "evidence": "e",
              "body": GOOD.format(n="foo", d="formatting a date")}

    def fake_child(argv, env, payload):  # the reflector: stages one intent for its run id, writes nothing
        assert argv[:2] == ["codex", "exec"] and env[config.CHILD_SESSION_ENV] == "1"
        intent_queue.append(env[config.RUN_ID_ENV], intent, env[config.PROJECT_ROOT_ENV])
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(spawn, "_detached_spawn", fake_child)
    spawn.main([str(transcript), SID, "run-9", str(roots[layer.PROJECT]), str(roots[layer.GLOBAL])])

    window, end = capture.window(str(transcript), 0)
    [row] = canon_queue.read()
    prov = row["provenance"]
    assert prov["harness"] == "codex" and prov["kind"] == "reflector"
    assert prov["session_id"] == SID and prov["transcript_path"] == str(transcript)
    assert prov["range"] == [0, end] and end == transcript.stat().st_size
    assert prov["window_sha256"] == hashlib.sha256(window.encode("utf-8")).hexdigest()
    assert counters.session_offset(SID, roots[layer.PROJECT]) == end  # the watermark moved under .codex


# --- carrier ------------------------------------------------------------------------------------

def test_codex_carrier_command_registers_stage_skill_and_guards_the_child(codex, repo):
    argv = spawn.build_codex_command(codex_bin="codex", run_id="run-7", proot=repo / ".codex", cwd=repo,
                                     model="gpt-6-luna", effort="low")
    assert argv[:4] == ["codex", "exec", "-s", "read-only"] and argv[-1] == "-"
    assert "--ephemeral" in argv and "--skip-git-repo-check" in argv
    assert argv[argv.index("-C") + 1] == str(repo)
    assert argv[argv.index("-m") + 1] == "gpt-6-luna"
    overrides = [argv[i + 1] for i, a in enumerate(argv) if a == "-c"]
    assert 'mcp_servers.stage_skill.args=["-m","autoharness.stage_skill.server"]' in overrides
    assert any(o.startswith("mcp_servers.stage_skill.command=") for o in overrides)
    assert f'mcp_servers.stage_skill.env.{config.RUN_ID_ENV}="run-7"' in overrides
    assert f'mcp_servers.stage_skill.env.{config.PROJECT_ROOT_ENV}="{repo / ".codex"}"' in overrides
    assert f'mcp_servers.stage_skill.env.{config.CHILD_SESSION_ENV}="1"' in overrides
    assert f'mcp_servers.stage_skill.env.{layer.HARNESS_ENV}="codex"' in overrides
    assert 'model_reasoning_effort="low"' in overrides


def test_codex_child_has_a_closed_tool_set(codex, repo):
    argv = spawn.build_codex_command(codex_bin="codex", run_id="r", proot=repo / ".codex", cwd=repo)
    assert "--ignore-user-config" in argv  # none of the user's MCP servers, web search or features
    disabled = [argv[i + 1] for i, a in enumerate(argv) if a == "--disable"]
    assert set(disabled) == {"apps", "browser_use", "browser_use_external", "computer_use"}
    overrides = [argv[i + 1] for i, a in enumerate(argv) if a == "-c"]
    assert 'web_search="disabled"' in overrides
    assert 'mcp_servers.stage_skill.tools.stage_skill.approval_mode="approve"' in overrides  # headless never asks
    assert argv.index("-s") + 1 == argv.index("read-only")


def test_codex_child_runs_in_an_empty_temporary_cwd_named_nowhere_near_the_repo(codex, repo, tmp_path):
    roots = {layer.PROJECT: repo / ".codex", layer.GLOBAL: tmp_path / "g"}
    seen = {}

    def fake(argv, env, payload):
        cwd = Path(argv[argv.index("-C") + 1])
        seen.update(cwd=cwd, existed=cwd.is_dir(), empty=not any(cwd.iterdir()), payload=payload)
        return SimpleNamespace(returncode=0, stderr="")

    spawn.run("W", "run-5", roots=roots, session_id=SID, spawn_fn=fake)
    assert seen["existed"] and seen["empty"]  # a project layer is discovered upward from cwd: there is none here
    assert not seen["cwd"].is_relative_to(repo) and seen["cwd"] != repo
    assert not seen["cwd"].exists()  # gone once the child exited
    assert f"{repo / '.codex' / 'skills'}" in seen["payload"] and f"{repo / '.agents' / 'skills'}" in seen["payload"]


def test_codex_carrier_without_model_leaves_the_configured_default():
    argv = spawn.build_codex_command(codex_bin="codex", run_id="r", proot="/p/.codex", cwd="/p", model="")
    assert "-m" not in argv


def test_carrier_inlines_the_agent_prompt_for_codex_and_not_for_claude(codex, repo, monkeypatch):
    argv, preface = spawn._carrier("autoharness:reflector", "run-1", repo / ".codex")
    assert argv[0] == config.CODEX_BIN and preface.startswith(spawn.CODEX_PREFACE)
    assert "You only ever **propose**" in preface  # agents/reflector.md body, frontmatter stripped
    assert "name: reflector" not in preface
    monkeypatch.setattr(layer, "HARNESS", "claude")
    argv, preface = spawn._carrier("autoharness:reflector", "run-1", repo / ".claude", claude_bin="claude")
    assert argv[:3] == ["claude", "-p", "--agent"] and preface == ""


def test_run_feeds_codex_the_preface_then_the_bundle(codex, repo, tmp_path):
    roots = {layer.PROJECT: repo / ".codex", layer.GLOBAL: tmp_path / "g"}
    calls = []

    def fake(argv, env, payload):
        calls.append((argv, env, payload))
        return SimpleNamespace(returncode=0, stderr="")

    spawn.run("WINDOW_MARK", "run-3", roots=roots, session_id=SID, spawn_fn=fake)
    [(argv, env, payload)] = calls
    assert argv[0] == config.CODEX_BIN and argv[1] == "exec"
    assert payload.startswith(spawn.CODEX_PREFACE) and "WINDOW_MARK" in payload
    assert env[config.CHILD_SESSION_ENV] == "1" and env[config.RUN_ID_ENV] == "run-3"


def test_fork_carrier_is_claude_only(codex, repo, tmp_path):
    roots = {layer.PROJECT: repo / ".codex", layer.GLOBAL: tmp_path / "g"}
    calls = []
    spawn.run("W", "run-4", roots=roots, session_id=SID, carrier="fork",
              spawn_fn=lambda argv, env, payload: calls.append(argv) or SimpleNamespace(returncode=0, stderr=""))
    assert calls[0][1] == "exec"  # never `claude --resume` under Codex


# --- compare-first sees the canon ---------------------------------------------------------------

def test_description_index_offers_canon_roots_read_only(codex, repo, tmp_path, monkeypatch):
    canon = repo / ".agents" / "skills"
    (canon / "canon-one").mkdir(parents=True)
    (canon / "canon-one" / "SKILL.md").write_text(GOOD.format(n="canon-one", d="shipping"), encoding="utf-8")
    (canon / ".canon-one-before-20260101").mkdir()
    (canon / ".canon-one-before-20260101" / "SKILL.md").write_text(GOOD.format(n="old", d="old"), encoding="utf-8")
    monkeypatch.setattr(config, "INDEX_ROOTS", "{project}/.agents/skills", raising=False)
    roots = {layer.PROJECT: repo / ".codex", layer.GLOBAL: tmp_path / "g"}

    idx = spawn.description_index(roots)
    assert "- canon-one [canon]: Use when shipping." in idx
    assert "old" not in idx  # hidden backup dirs beside the canon are not offered
    assert "canon" not in spawn.description_index(roots, agent_only=True)  # the curator never sees it


def test_compare_first_reads_global_even_when_global_is_disabled(codex, repo, tmp_path):
    roots = {layer.PROJECT: repo / ".codex", layer.GLOBAL: tmp_path / "g"}
    (roots[layer.GLOBAL] / "skills" / "glob-one").mkdir(parents=True)
    (roots[layer.GLOBAL] / "skills" / "glob-one" / "SKILL.md").write_text(GOOD.format(n="glob-one", d="globbing"),
                                                                         encoding="utf-8")
    assert config.DISABLE_GLOBAL  # the fixture's deployment
    assert "- glob-one [global]: Use when globbing." in spawn.description_index(roots)
    assert "glob-one" not in spawn.description_index(roots, agent_only=True)  # not the curator's to manage


def test_index_roots_default_is_empty_for_claude(monkeypatch):
    monkeypatch.setattr(config, "INDEX_ROOTS", "", raising=False)
    assert spawn.index_roots({layer.PROJECT: Path("/p/.claude")}) == []


# --- digest ------------------------------------------------------------------------------------

def _codex_line(kind, **payload):
    return json.dumps({"type": "response_item", "payload": {"type": kind, **payload}})


def test_codex_digest_keeps_the_persons_text_and_tool_names(codex, tmp_path):
    lines = [
        json.dumps({"type": "session_meta", "payload": {"id": SID, "source": "cli"}}),
        _codex_line("message", role="developer", content=[{"type": "input_text", "text": "## Memory"}]),
        _codex_line("message", role="user", content=[{"type": "input_text", "text": "# AGENTS.md instructions"}],
                    internal_chat_message_metadata_passthrough={"content_item_kinds": ["agents_md.instructions"]}),
        _codex_line("message", role="user", content=[{"type": "input_text", "text": "fix the build"}],
                    internal_chat_message_metadata_passthrough={"content_item_kinds": ["user.text"]}),
        _codex_line("reasoning", summary=[], encrypted_content="xxx"),
        _codex_line("custom_tool_call", name="exec", input="ls", call_id="c1"),
        _codex_line("custom_tool_call_output", call_id="c1", output=[{"type": "input_text", "text": "SECRET OUTPUT"}]),
        _codex_line("message", role="assistant", content=[{"type": "output_text", "text": "done"}]),
    ]
    p = tmp_path / "rollout.jsonl"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    d = capture.digest(str(p), p.stat().st_size)
    assert "user: fix the build" in d and "assistant: [tool: exec]" in d and "assistant: done" in d
    assert "AGENTS.md" not in d and "## Memory" not in d and "SECRET OUTPUT" not in d


def test_claude_digest_parser_is_untouched_by_a_codex_rollout(tmp_path, monkeypatch):
    monkeypatch.setattr(layer, "HARNESS", "claude")
    p = tmp_path / "rollout.jsonl"
    p.write_text(_codex_line("message", role="assistant", content=[{"type": "output_text", "text": "x"}]) + "\n",
                 encoding="utf-8")
    assert capture.digest(str(p), p.stat().st_size) == ""  # degrades to no digest, never crashes


# --- hook output contract ------------------------------------------------------------------------

def test_pretooluse_deny_carries_a_reason_as_codex_requires(capsys):
    dispatch._emit({"deny": True, "reason": "reflector may only stage intents"})
    out = json.loads(capsys.readouterr().out)["hookSpecificOutput"]
    assert out["hookEventName"] == "PreToolUse" and out["permissionDecision"] == "deny"
    assert out["permissionDecisionReason"].strip()


def test_tails_round_trip_and_unsafe_ids(tmp_path):
    root = tmp_path / ".codex"
    tails.note(SID, "/t.jsonl", 2, root)
    assert tails.pending(root)[0]["count"] == 2
    tails.clear(SID, root)
    assert tails.pending(root) == []
    with pytest.raises(ValueError):
        tails.note("../evil", None, 1, root)

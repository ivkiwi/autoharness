"""REF launch vehicle: deterministically assemble the reflector input → detached spawn a child session → connect to promoter.drain.

architecture line 29 / [reflector-subagent] / [cap]: at trigger time CAP gives "redacted episode window +
trigger cadence"; this step assembles the three pieces (window + the existing skill description index
(compare-first dedupe source, skips archived) + the single-source format_spec) and feeds them via
stdin to the cross-process reflector — nothing is persisted; the bundle lives only in the pipe. spawn sets CHILD_SESSION_ENV (recursion guard) +
injects run_id/root via env (stage_skill uses these to append back to the queue), then drains the intent
queue to disk after the child session ends. Authoring and landing are fully split across processes from
here: the reflector only appends intents, the promoter exclusively validates and lands.

ponytail: run() is the body of the "detached background job" (synchronous spawn→wait→drain); the "do not block the host Stop" detach is started in the background at the hook top level by the Phase 7 dispatch calling run(). spawn_fn is injectable (system tests use a fake reflector script in place of the real claude). Precise handling of the transcript upper-bound race (cap.md open) is still tolerated at v0.
"""
import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from types import SimpleNamespace

from autoharness import config
from autoharness.hook import capture, promoter
from autoharness.lib import (
    atomic,
    counters,
    layer,
    redact,
    sidecar,
    skill_store,
    tails,
    validate,
)

log = logging.getLogger(__name__)


def index_roots(roots=None):
    """Extra compare-first roots (config.INDEX_ROOTS): canon the host reads natively and autoharness
    never writes. `{project}` is the repo, i.e. the project root's parent."""
    if not config.INDEX_ROOTS:
        return []
    proot = (roots or {}).get(layer.PROJECT)
    project = Path(proot).parent if proot else layer.default_root(layer.PROJECT).parent
    out = []
    for raw in config.INDEX_ROOTS.split(os.pathsep):
        if raw.strip():
            p = Path(raw.strip().replace("{project}", str(project))).expanduser()
            if p.is_dir():
                out.append(p)
    return out


def _index_line(path, symbol, tag):
    # anyone's skill can sit here; a latin-1 or unreadable SKILL.md must not take the reflection down
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        log.warning("skipping unreadable skill description: %s", path)
        return None
    fm = validate._frontmatter(text) or {}
    name = fm.get("name") or symbol
    desc = fm.get("description") or "(no description)"
    return f"- {name} [{tag}]: {desc}"


def description_index(roots=None, *, agent_only=False):
    roots = roots or {}
    lines = []
    # the reflector compares against everything a session can load, a disabled global layer included
    # (its patch there becomes a proposal); the curator only ever manages the active layers
    for lyr in (config.active_layers() if agent_only else layer.LAYERS):
        root = roots.get(lyr)
        skills = layer.skills_dir(lyr, root)
        if not skills.exists():
            continue
        for path in sorted(skills.glob(f"*/{skill_store.SKILL_FILE}")):
            symbol = path.parent.name
            if agent_only and not sidecar.is_agent_created(lyr, symbol, root):
                continue  # curator only ever sees its own skills; native/user stay out of the pool
            line = _index_line(path, symbol, lyr)
            if line:
                lines.append(line)
    if not agent_only:  # canon is read-only for the reflector (its patch becomes a proposal), invisible to the curator
        for root_dir in index_roots(roots):
            for path in sorted(root_dir.glob(f"*/{skill_store.SKILL_FILE}")):
                symbol = path.parent.name
                if symbol.startswith("."):
                    continue  # backups and hidden dirs kept beside the canon
                line = _index_line(path, symbol, "canon")
                if line:
                    lines.append(line)
    return "\n".join(lines) if lines else "(no live skills yet)"


def build_bundle(window, index, spec, digest=""):
    preamble = (
        "# Prior context digest (older exchanges, tool outputs omitted — background only,"
        " never an evidence source)\n\n" + digest + "\n\n"
    ) if digest else ""
    return (
        preamble
        + "# Episode window (redacted)\n\n" + window
        + "\n\n# Existing skills (compare-first: dedupe / patch / where)\n\n" + index
        + "\n\n# Authoring + format spec (write to satisfy this)\n\n" + spec + "\n"
    )


def build_curator_bundle(index, spec):
    # The curator consolidates the whole agent-authored library, not one episode — so no window.
    return (
        "# Agent-authored skills (consolidate: merge narrow siblings into class-level umbrellas)\n\n"
        + index
        + "\n\n# Authoring + format spec (merged skills must satisfy this)\n\n" + spec + "\n"
    )


# Fork-carrier reflect instruction (direction G): arrives as the -p prompt (a fresh user turn on the
# forked conversation) — never as --agent/--system-prompt, which would invalidate the parent prefix.
# Carries the F1 reconcile duty: a new rule must supersede contradicting old statements.
FORK_INSTRUCTION = (
    "Autoharness reflection pass (forked session: the conversation above is your evidence source).\n"
    "Compare-first against the skill index below: prefer patching an existing skill over creating a\n"
    "new one; distill only class-level reusable lessons, never session narratives. When you distill\n"
    "a new rule or preference, search the managed skill trees for overlapping or contradicting\n"
    "statements and stage updates so the new supersedes the old. Propose every change exclusively\n"
    "via the stage_skill tool — never write files directly. Evidence must quote this session\n"
    "verbatim. If nothing is worth keeping, stage nothing.\n"
)


def build_fork_command(*, session_id, claude_bin):
    return [claude_bin, "-p", "--resume", str(session_id), "--fork-session",
            "--dangerously-skip-permissions"]


def build_fork_prompt(index, spec):
    return (FORK_INSTRUCTION
            + "\n# Existing skills (compare-first: dedupe / patch / where)\n\n" + index
            + "\n\n# Authoring + format spec (write to satisfy this)\n\n" + spec + "\n")


def agent_prompt(agent):
    """The agent definition's body for a carrier without --agent: `<plugin>:<name>` -> agents/<name>.md minus frontmatter."""
    name = str(agent).split(":")[-1]
    text = (config.AGENTS_DIR / f"{name}.md").read_text(encoding="utf-8")
    m = re.match(r"---\n.*?\n---\n", text, re.S)
    return (text[m.end():] if m else text).strip() + "\n"


CODEX_PREFACE = ("You run under Codex in a read-only sandbox: inspect existing skills with shell reads only "
                 "(cat, grep, ls); the stage_skill MCP tool (listed as mcp__stage_skill__stage_skill) is "
                 "your only write.\n\n")


def _toml(value):
    return json.dumps(str(value))  # a TOML basic string is JSON-compatible


# built-in features that act outside the sandbox; off in the child whatever the user's config says
CODEX_DISABLED_FEATURES = ("apps", "browser_use", "browser_use_external", "computer_use")


def build_codex_command(*, codex_bin, run_id, proot, cwd, model="", effort="low"):
    """`codex exec` as the reflector carrier: prompt on stdin, read-only sandbox as the write backstop,
    --ephemeral so the child leaves no rollout, stage_skill registered per invocation with the env the
    server reads (run id, project root, harness, child guard) — hooks inside the child inherit the
    process env for the guard, the MCP server gets the same values explicitly.

    The tool set is closed: --ignore-user-config empties the user layer, so none of the user's MCP
    servers (a write through one of them passes a read-only sandbox), web search or feature flags
    reach the child; stage_skill is the only server, and it carries approval_mode=approve because a
    headless exec never asks — a tool without it is refused, not prompted."""
    server_env = {config.RUN_ID_ENV: run_id, config.PROJECT_ROOT_ENV: str(proot),
                  config.CHILD_SESSION_ENV: "1", layer.HARNESS_ENV: "codex",
                  "PYTHONPATH": str(Path(config.__file__).resolve().parent.parent)}
    argv = [codex_bin, "exec", "-s", "read-only", "--skip-git-repo-check", "--ephemeral",
            "--ignore-user-config", "-C", str(cwd)]
    for feature in CODEX_DISABLED_FEATURES:
        argv += ["--disable", feature]
    argv += ["-c", 'web_search="disabled"',
             "-c", f"mcp_servers.stage_skill.command={_toml(sys.executable)}",
             "-c", 'mcp_servers.stage_skill.args=["-m","autoharness.stage_skill.server"]',
             "-c", 'mcp_servers.stage_skill.tools.stage_skill.approval_mode="approve"',
             "-c", f"model_reasoning_effort={_toml(effort)}"]
    for key, value in server_env.items():
        argv += ["-c", f"mcp_servers.stage_skill.env.{key}={_toml(value)}"]
    if model:
        argv += ["-m", model]
    return argv + ["-"]


@contextlib.contextmanager
def _child_cwd():
    """Codex only: an empty temporary cwd for the child. Codex discovers a project layer upward from
    cwd, and a repo's .codex/config.toml could bring its own MCP servers into the child; the read-only
    sandbox still reads skill files anywhere by absolute path, so the repo is named in the preface
    instead of being the cwd."""
    if layer.HARNESS != "codex":
        yield None
        return
    d = tempfile.mkdtemp(prefix="autoharness-codex-")
    try:
        yield Path(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _codex_skill_dirs(proot):
    project = Path(proot).parent if proot else None
    dirs = [project / ".codex" / "skills", project / ".agents" / "skills"] if project else []
    return dirs + [Path.home() / ".codex" / "skills", Path.home() / ".agents" / "skills"]


def _carrier(agent, run_id, proot, claude_bin=None, cwd=None):
    """(argv, preface): what runs the agent and what must precede the bundle on stdin. `cwd` is the
    child's working directory (Codex: from _child_cwd); never the repo."""
    if layer.HARNESS == "codex":
        argv = build_codex_command(codex_bin=config.CODEX_BIN, run_id=run_id, proot=proot,
                                   cwd=cwd or tempfile.gettempdir(),
                                   model=config.CODEX_MODEL, effort=config.CODEX_EFFORT)
        where = ", ".join(str(d) for d in _codex_skill_dirs(proot))
        preface = (CODEX_PREFACE + f"Existing skills live under {where}; read <dir>/<name>/SKILL.md "
                   "by absolute path, your working directory holds nothing.\n\n")
        return argv, preface + agent_prompt(agent) + "\n"
    return build_command(agent=agent, claude_bin=claude_bin or config.CLAUDE_BIN), ""


def build_command(*, agent, claude_bin):
    # Reflection is an unattended background job — nobody is there to approve tool calls, so skip the
    # permission prompt. The security boundary is held by the agent's tools allowlist
    # (Read/Grep/Glob/stage_skill) + the top-level PreToolUse write backstop, not by the prompt.
    # (live e2e: a reflector's real stage_skill call gets blocked by the permission gate and can only
    # "narrate"; only with this flag does it land.)
    return [claude_bin, "-p", "--agent", agent, "--dangerously-skip-permissions"]


def child_env(run_id, root, *, base_env=None):
    env = dict(os.environ if base_env is None else base_env)
    env[config.CHILD_SESSION_ENV] = "1"
    env[config.RUN_ID_ENV] = run_id
    env[config.PROJECT_ROOT_ENV] = str(root)
    return env


def _spawn_error(proc, argv):
    """Return bounded, redacted child diagnostics safe for the run account."""
    stderr = redact.redact(str(proc.stderr or "")).strip()[-2000:]
    return {
        "argv0": redact.redact(str(argv[0])) if argv else None,
        "returncode": proc.returncode,
        "stderr_tail": stderr,
    }


def _detached_spawn(argv, env, bundle):
    """Run the reflector child to completion; report a crash on stderr instead of discarding it."""
    proc = subprocess.run(argv, input=bundle, text=True, encoding="utf-8", errors="replace",
                          env=env, capture_output=True, check=False)
    if proc.returncode != 0:
        error = _spawn_error(proc, argv)
        print(f"reflector child {error['argv0']} exited {proc.returncode}: "
              f"{error['stderr_tail']}", file=sys.stderr)
    return proc


def _record_spawn_failure(run_id, roots, proc, argv):
    """Persist a crashed reflector in the run account (#160).

    The detached launch DEVNULLs this whole process (dispatch.py), so neither the print above nor
    the exit code reaches an operator. The runs/ account is where landed runs already live; verdicts
    (if the child staged intents before dying) are preserved alongside the crash record.
    """
    if proc is None or getattr(proc, "returncode", 0) == 0:
        return
    state = layer.state_dir(layer.PROJECT, roots.get(layer.PROJECT))
    runs = state / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    account = runs / f"{run_id}.json"
    record = json.loads(account.read_text(encoding="utf-8")) if account.exists() else {"run_id": run_id}
    record["spawn_error"] = _spawn_error(proc, argv)
    atomic.write_text(account, json.dumps(record, ensure_ascii=False, indent=2))


def _invoke_spawn(spawn_fn, argv, env, payload, run_id, roots):
    try:
        return spawn_fn(argv, env, payload)
    except OSError as exc:
        proc = SimpleNamespace(returncode=None, stderr=f"{type(exc).__name__}: {exc}")
        _record_spawn_failure(run_id, roots, proc, argv)
        raise


def run(window_text, run_id, *, roots, repo_name=None, agent=None, claude_bin=None,
        spec_path=None, digest="", session_id=None, carrier=None, spawn_fn=None, provenance=None):
    roots = roots or {}
    proot = roots.get(layer.PROJECT)
    spec = (spec_path or config.FORMAT_SPEC).read_text(encoding="utf-8")

    carrier = carrier or config.REFLECTOR_CARRIER
    with _child_cwd() as cwd:
        if carrier == "fork" and session_id and layer.HARNESS != "codex":  # no session to fork -> bundle chain (fail-safe)
            argv = build_fork_command(session_id=session_id, claude_bin=claude_bin or config.CLAUDE_BIN)
            payload = build_fork_prompt(description_index(roots), spec)  # -p reads the prompt from stdin
        else:
            argv, preface = _carrier(agent or config.REFLECTOR_AGENT, run_id, proot, claude_bin, cwd=cwd)
            payload = preface + build_bundle(window_text, description_index(roots), spec, digest=digest)

        env = child_env(run_id, proot)
        proc = _invoke_spawn(spawn_fn or _detached_spawn, argv, env, payload, run_id, roots)
    verdicts = promoter.drain(run_id, roots=roots, repo_name=repo_name, provenance=provenance)
    _record_spawn_failure(run_id, roots, proc, argv)
    return verdicts


def _snapshot_skills(run_id, roots):
    """Pre-run library snapshot (direction E, mirrors Hermes): covers the one risk atomic landing
    and reversible archiving cannot — a whole curator run writing the library wrong. Recovery is a
    manual unpack; rotation keeps SNAPSHOT_KEEP per layer."""
    snapdir = layer.state_dir(layer.PROJECT, roots.get(layer.PROJECT)) / "snapshots"
    snapdir.mkdir(parents=True, exist_ok=True)
    for lyr in config.active_layers():
        skills = layer.skills_dir(lyr, roots.get(lyr))
        if not skills.exists():
            continue
        with tarfile.open(snapdir / f"{run_id}-{lyr}.tar.gz", "w:gz") as tar:
            tar.add(skills, arcname="skills")
        kept = sorted(snapdir.glob(f"*-{lyr}.tar.gz"), key=lambda p: p.stat().st_mtime)
        for old in kept[: max(0, len(kept) - config.SNAPSHOT_KEEP)]:
            old.unlink()


def run_curator(run_id, *, roots, repo_name=None, agent=None, claude_bin=None,
                spec_path=None, spawn_fn=None):
    roots = roots or {}
    try:
        _snapshot_skills(run_id, roots)
    except OSError:
        pass  # a transient disk issue must not silently disable curation
    except Exception:
        log.exception('unexpected snapshot error; curator running without safety net')
    spec = (spec_path or config.FORMAT_SPEC).read_text(encoding="utf-8")
    bundle = build_curator_bundle(description_index(roots, agent_only=True), spec)

    with _child_cwd() as cwd:
        argv, preface = _carrier(agent or config.CURATOR_AGENT, run_id, roots.get(layer.PROJECT), claude_bin, cwd=cwd)
        env = child_env(run_id, roots.get(layer.PROJECT))
        proc = _invoke_spawn(spawn_fn or _detached_spawn, argv, env, preface + bundle, run_id, roots)
    verdicts = promoter.drain(run_id, roots=roots, repo_name=repo_name, provenance={"kind": "curator"})
    _record_spawn_failure(run_id, roots, proc, argv)
    return verdicts


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--curate":
        run_id, proot, groot = argv[1:]
        return run_curator(run_id, roots={layer.PROJECT: Path(proot), layer.GLOBAL: Path(groot)})
    transcript_path, session_id, run_id, proot, groot = argv
    roots = {layer.PROJECT: Path(proot), layer.GLOBAL: Path(groot)}
    # the note this reflection answers for; a Stop during the child may write a newer one
    launched_from = (tails.read(session_id, roots[layer.PROJECT]) or {}).get("version") if layer.HARNESS == "codex" else None
    offset = counters.session_offset(session_id, roots[layer.PROJECT])
    window_text, new_offset = capture.window(transcript_path, offset)
    provenance = {"kind": "reflector", "session_id": session_id, "transcript_path": transcript_path,
                  "range": [offset, new_offset],
                  "window_sha256": hashlib.sha256(window_text.encode("utf-8")).hexdigest()}
    outcome = {}

    def spawn_and_remember(argv_, env, payload):
        # run() keeps a crashed child in the run account instead of raising (intents staged before the
        # crash still drain), so the exit status is remembered here to decide whether the window was fed
        proc = _detached_spawn(argv_, env, payload)
        outcome["returncode"] = getattr(proc, "returncode", 0)
        return proc

    try:
        result = run(window_text, run_id, roots=roots, session_id=session_id,
                     digest=capture.digest(transcript_path, offset), provenance=provenance,
                     spawn_fn=spawn_and_remember)
    except OSError:  # the child could not even start (no binary): that spends the limit like a refusal
        _window_not_fed(session_id, new_offset, roots[layer.PROJECT])
        raise
    if outcome.get("returncode", 0) != 0:
        _window_not_fed(session_id, new_offset, roots[layer.PROJECT])
        return result
    counters.write_session_offset(session_id, new_offset, roots[layer.PROJECT])
    if layer.HARNESS == "codex":
        tails.settle(session_id, launched_from, new_offset, roots[layer.PROJECT])
    return result


def _window_not_fed(session_id, new_offset, proot):
    """The child did not consume the window (a refusal before the model, a launch error): the
    watermark and the note stay, unless the child has now failed REFLECT_MAX_FAILURES times on this
    window — then the window is given up explicitly rather than fed forever."""
    if layer.HARNESS != "codex":
        return
    if tails.fail(session_id, proot) >= config.REFLECT_MAX_FAILURES:
        counters.write_session_offset(session_id, new_offset, proot)
        tails.give_up(session_id, f"reflection_failed_{config.REFLECT_MAX_FAILURES}x", new_offset, proot)


if __name__ == "__main__":
    main()

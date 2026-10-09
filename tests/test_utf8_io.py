"""Text I/O names its encoding everywhere: written as UTF-8, read with the locale codec is cp1252 on a
Windows host (3.11-3.14 without PYTHONUTF8), which mojibakes or raises on the very bytes the writers
produce."""
import ast
import io
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "autoharness"


def _kw(call, name):
    return next((k.value for k in call.keywords if k.arg == name), None)


def _locale_text_io(tree):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
        if name in ("read_text", "write_text") and isinstance(fn, ast.Attribute) \
                and not (isinstance(fn.value, ast.Name) and fn.value.id == "atomic"):
            if _kw(node, "encoding") is None:
                yield node.lineno, name
        elif name == "open" and not (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name)
                                     and fn.value.id in ("tarfile", "gzip", "zipfile", "os")):
            mode = node.args[1] if len(node.args) > 1 else _kw(node, "mode")
            text = not (isinstance(mode, ast.Constant) and "b" in str(mode.value))
            if text and _kw(node, "encoding") is None:
                yield node.lineno, "open"
        elif name in ("run", "Popen", "check_output"):
            text = _kw(node, "text") or _kw(node, "universal_newlines")
            if isinstance(text, ast.Constant) and text.value and _kw(node, "encoding") is None:
                yield node.lineno, f"subprocess.{name}(text=True)"


def test_every_text_read_write_and_pipe_names_its_encoding():
    found = [f"{p.relative_to(SRC.parent)}:{line} {what}"
             for p in sorted(SRC.rglob("*.py"))
             for line, what in _locale_text_io(ast.parse(p.read_text(encoding="utf-8")))]
    assert found == [], "locale-dependent text I/O:\n" + "\n".join(found)


def _cp1252_pipe(data=b""):
    return io.TextIOWrapper(io.BytesIO(data), encoding="cp1252")  # sys.std* on a Windows host


def test_hook_pipes_carry_utf8_on_a_cp1252_host(monkeypatch):
    from autoharness.hook import dispatch
    seen = {}

    def fake(event):
        seen.update(event)
        return {"handled": "SessionStart", "result": {"context": "✳ index — 👍"}}
    monkeypatch.setattr(dispatch, "dispatch", fake)
    event = {"hook_event_name": "SessionStart", "cwd": "/tmp/проект 👍"}
    monkeypatch.setattr(sys, "stdin", _cp1252_pipe(json.dumps(event, ensure_ascii=False).encode("utf-8")))
    monkeypatch.setattr(sys, "stdout", _cp1252_pipe())
    dispatch.main()
    assert seen["cwd"] == "/tmp/проект 👍"
    sys.stdout.flush()
    out = json.loads(sys.stdout.buffer.getvalue())  # ASCII-escaped JSON: no codec in play on the way out
    assert out["hookSpecificOutput"]["additionalContext"] == "✳ index — 👍"


def test_mcp_pipes_carry_utf8_on_a_cp1252_host(monkeypatch, tmp_path):
    from autoharness.lib import intent_queue
    from autoharness.stage_skill import server
    body = "---\nname: foo\ndescription: Use when a date — 👍 — needs ISO format.\n---\n# Foo\nUse strftime.\n"
    req = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "stage_skill", "arguments": {
        "action": "create", "name": "foo", "body": body, "reason": "r", "evidence": "e"}}}
    monkeypatch.setenv("AUTOHARNESS_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "stdin", _cp1252_pipe((json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8")))
    monkeypatch.setattr(sys, "stdout", _cp1252_pipe())
    server.serve()
    assert intent_queue.read("interactive", tmp_path)[0]["body"] == body

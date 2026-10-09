import os

import pytest

from autoharness import config


@pytest.fixture
def dir_link():
    """Link a directory: a symlink, or an NTFS junction where Windows withholds the symlink
    privilege (no Developer Mode / not elevated). Both are followed by Path.resolve()."""
    def link(alias, target):
        try:
            alias.symlink_to(target, target_is_directory=True)
        except OSError:
            if os.name != "nt":
                raise
            import _winapi
            _winapi.CreateJunction(str(target), str(alias))
    return link


@pytest.fixture(autouse=True)
def _no_host_project_dir(monkeypatch):
    # the host (and spawn, for a child) pin the project root in env; inherited here they would pin
    # every cwd-based root test
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    monkeypatch.delenv(config.PROJECT_ROOT_ENV, raising=False)


@pytest.fixture(autouse=True)
def _no_real_notifier(monkeypatch):
    # config reads AUTOHARNESS_NOTIFY* at import: without this, a contributor's own notifier (a team
    # Slack hook, desktop popups) would fire for every drain the suite runs
    monkeypatch.setattr(config, "NOTIFY", "")
    monkeypatch.setattr(config, "NOTIFY_CMD", "")


@pytest.fixture(autouse=True)
def _private_gate_and_flags(monkeypatch, tmp_path_factory):
    # the shared canon gate defaults to ~/.agents: a test must never queue into or publish over the real
    # one. The phase-1 flags are read from env at import, so pin them too: an operator's own setting
    # leaking into the run would flip half the suite
    gate = tmp_path_factory.mktemp("gate")
    monkeypatch.setattr(config, "GATE_DIR", gate / "skill-gate")
    monkeypatch.setattr(config, "CANON_ROOT", gate / "agents")
    monkeypatch.setattr(config, "HOST_SKILL_ROOTS", [gate / "claude-skills", gate / "codex-skills"])
    monkeypatch.setattr(config, "HARNESS", "claude")
    monkeypatch.setattr(config, "PROPOSE_ONLY", False)
    monkeypatch.setattr(config, "DISABLE_GLOBAL", False)

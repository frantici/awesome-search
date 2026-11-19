"""
Tests for code purge/watch controls and env probe.
"""

import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from awesome_search.code_engine import CodeSearchEngine
from awesome_search.env_probe import EnvStore, confirm_from_probe, probe_environment


@pytest.fixture
def temp_project():
    temp_dir = Path(tempfile.mkdtemp())
    (temp_dir / "backend").mkdir()
    (temp_dir / "backend" / "app.py").write_text("def hello():\n    return 1\n")
    yield temp_dir
    shutil.rmtree(temp_dir)


def _fake_embed(texts, show_progress=False):
    out = []
    for text in texts:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        out.append(rng.standard_normal(384).astype("float32"))
    return np.stack(out)


class TestPurgeAndWatch:
    def test_purge_keeps_rules(self, temp_project):
        engine = CodeSearchEngine(temp_project)
        engine.rules.set_rule("vendor/**", "library")
        with patch.object(CodeSearchEngine, "_embed", side_effect=_fake_embed):
            engine.index_codebase(force=True, show_progress=False)
        assert engine.faiss_path.exists()
        assert engine.symbols_path.exists()

        stats = engine.purge(keep_rules=True)
        assert stats["removed"] >= 1
        assert not engine.faiss_path.exists()
        assert not engine.symbols_path.exists()
        assert (engine.code_folder / "code_rules.json").exists()
        assert engine.rules.list_rules()[0].pattern == "vendor/**"

    def test_purge_all(self, temp_project):
        engine = CodeSearchEngine(temp_project)
        engine.rules.set_rule("vendor/**", "library")
        with patch.object(CodeSearchEngine, "_embed", side_effect=_fake_embed):
            engine.index_codebase(force=True, show_progress=False)
        engine.purge(keep_rules=False)
        assert not (engine.code_folder / "code_rules.json").exists()

    def test_watch_stop_signal(self, temp_project):
        engine = CodeSearchEngine(temp_project)
        with patch.object(CodeSearchEngine, "_embed", side_effect=_fake_embed):
            engine.index_codebase(force=True, show_progress=False)

            def run_watch():
                engine.watch(poll_seconds=0.05)

            t = threading.Thread(target=run_watch, daemon=True)
            t.start()
            deadline = time.time() + 2.0
            while time.time() < deadline and not engine.is_watch_running():
                time.sleep(0.05)
            assert engine.is_watch_running()
            result = engine.stop_watch()
            assert result["ok"] is True
            t.join(timeout=2.0)
            assert not t.is_alive()
            assert not engine.is_watch_running()

    def test_watch_pause_and_resume_commands(self, temp_project):
        engine = CodeSearchEngine(temp_project)
        with patch.object(CodeSearchEngine, "_embed", side_effect=_fake_embed):
            engine.index_codebase(force=True, show_progress=False)

            def run_watch():
                engine.watch(poll_seconds=0.05)

            t = threading.Thread(target=run_watch, daemon=True)
            t.start()
            deadline = time.time() + 2.0
            while time.time() < deadline and not engine.is_watch_running():
                time.sleep(0.05)

            paused = engine.stop_watch(minutes=5)
            assert paused["ok"] and paused["action"] == "pause"
            # Give the watcher a moment to consume the pause command
            time.sleep(0.2)
            status = engine._read_watch_status()
            assert status is not None
            assert status.get("state") == "paused"

            resumed = engine.resume_watch()
            assert resumed["ok"]
            time.sleep(0.2)
            status = engine._read_watch_status()
            assert status is not None
            assert status.get("state") == "running"

            engine.stop_watch()
            t.join(timeout=2.0)


class TestEnvProbe:
    def test_detects_python_stack(self, temp_project):
        (temp_project / "pyproject.toml").write_text("[project]\nname='x'\n")
        report = probe_environment(temp_project)
        assert "python" in report.stacks
        assert "pyproject.toml" in report.signals
        assert report.python.get("preferred_cmd") in {"python3", "python"} or (
            report.python.get("preferred_cmd") or ""
        ).endswith("python")

    def test_warns_when_node_missing(self, temp_project):
        (temp_project / "package.json").write_text('{"name":"demo"}')
        with patch("awesome_search.env_probe.shutil.which", return_value=None):
            with patch("awesome_search.env_probe._run_version", return_value=None):
                report = probe_environment(temp_project)
        assert "node" in report.stacks
        assert any("node" in w.lower() for w in report.warnings)

    def test_confirm_and_set(self, temp_project):
        (temp_project / "requirements.txt").write_text("numpy\n")
        data = confirm_from_probe(temp_project, overrides={"python_cmd": "python3"})
        assert data["python_cmd"] == "python3"
        store = EnvStore(temp_project)
        store.set_value("node_package_manager", "pnpm")
        loaded = store.load()
        assert loaded["node_package_manager"] == "pnpm"
        report = probe_environment(temp_project)
        assert report.confirmed["python_cmd"] == "python3"
        assert report.node["package_manager"] == "pnpm"

    def test_clear(self, temp_project):
        store = EnvStore(temp_project)
        store.save({"python_cmd": "python3"})
        assert store.clear() is True
        assert store.load() == {}

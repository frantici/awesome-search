"""
Tests for the code-symbol index (rules, classification, extraction, engine).
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from awesome_search.code_classify import (
    classify_group,
    detect_language,
    is_indexable_source,
)
from awesome_search.code_engine import CodeSearchEngine
from awesome_search.code_rules import CodeRuleStore, PathRule
from awesome_search.code_symbols import extract_symbols


@pytest.fixture
def temp_code_project():
    temp_dir = Path(tempfile.mkdtemp())
    (temp_dir / ".gitignore").write_text("node_modules/\nvendor/\n.venv/\n")
    (temp_dir / "backend" / "api").mkdir(parents=True)
    (temp_dir / "frontend" / "src").mkdir(parents=True)
    (temp_dir / "vendor" / "lib").mkdir(parents=True)
    (temp_dir / "scripts").mkdir()

    (temp_dir / "backend" / "api" / "users.py").write_text(
        '''
class UserService:
    def get_user(self, user_id: str):
        return {"id": user_id}

def create_user(name: str):
    return UserService().get_user(name)
'''
    )
    (temp_dir / "frontend" / "src" / "Button.tsx").write_text(
        '''
export function Button(props: { label: string }) {
  return <button>{props.label}</button>;
}

export const useToggle = (initial: boolean) => {
  return initial;
};
'''
    )
    (temp_dir / "vendor" / "lib" / "helper.py").write_text(
        '''
def third_party_helper():
    return 42
'''
    )
    (temp_dir / "scripts" / "build.sh").write_text("#!/bin/bash\necho hi\n")

    yield temp_dir
    shutil.rmtree(temp_dir)


class TestPathRules:
    def test_prefix_match(self):
        rule = PathRule("vendor/**", "library")
        assert rule.matches("vendor/lib/helper.py")
        assert rule.matches("vendor")
        assert not rule.matches("src/vendor/x.py")

    def test_single_segment_star(self):
        rule = PathRule("apps/*", "mine")
        assert rule.matches("apps/web")
        assert not rule.matches("apps/web/src/a.ts")

    def test_store_defaults_and_gitignore(self, temp_code_project):
        store = CodeRuleStore(temp_code_project, temp_code_project / ".awesome_index" / "code")
        assert store.resolve("vendor/lib/helper.py")["ownership"] == "library"
        assert store.resolve("backend/api/users.py")["ownership"] == "mine"
        assert store.resolve("node_modules/foo/index.js")["ownership"] == "library"

    def test_explicit_rule_overrides(self, temp_code_project):
        store = CodeRuleStore(temp_code_project, temp_code_project / ".awesome_index" / "code")
        store.set_rule("vendor/mycompany/**", "mine", group="backend")
        resolved = store.resolve("vendor/mycompany/sdk.py")
        assert resolved["ownership"] == "mine"
        assert resolved["group_override"] == "backend"

    def test_layer_guess(self, temp_code_project):
        store = CodeRuleStore(temp_code_project, temp_code_project / ".awesome_index" / "code")
        assert store.guess_layer("frontend/src/Button.tsx") == "frontend"
        assert store.guess_layer("backend/api/users.py") == "backend"


class TestClassify:
    def test_detect_language(self):
        assert detect_language(Path("a.py")) == "python"
        assert detect_language(Path("a.tsx")) == "typescript"
        assert detect_language(Path("a.rs")) == "rust"

    def test_group_mine_and_library(self, temp_code_project):
        store = CodeRuleStore(temp_code_project, temp_code_project / ".awesome_index" / "code")
        assert (
            classify_group("frontend/src/Button.tsx", "mine", store) == "frontend"
        )
        assert (
            classify_group("frontend/src/Button.tsx", "library", store)
            == "frontend_library"
        )
        assert classify_group("backend/api/users.py", "mine", store) == "backend"
        assert classify_group("scripts/build.sh", "mine", store) == "tooling"

    def test_is_indexable(self):
        assert is_indexable_source(Path("x.py"))
        assert is_indexable_source(Path("x.ts"))
        assert not is_indexable_source(Path("x.md"))
        assert not is_indexable_source(Path(".env"))


class TestSymbolExtract:
    def test_python_symbols(self):
        content = (
            "class Foo:\n"
            "    pass\n"
            "\n"
            "def bar(x):\n"
            "    return x\n"
        )
        syms = extract_symbols("a.py", content, "python", "mine", "backend")
        names = {s.name for s in syms}
        assert "Foo" in names
        assert "bar" in names
        foo = next(s for s in syms if s.name == "Foo")
        assert foo.kind == "class"
        assert foo.start_line == 1

    def test_typescript_symbols(self):
        content = (
            "export function Button() {}\n"
            "export interface Props { x: number }\n"
            "export const useX = () => 1;\n"
        )
        syms = extract_symbols("Button.tsx", content, "typescript", "mine", "frontend")
        names = {s.name: s.kind for s in syms}
        assert names.get("Button") in {"function", "component"}
        assert names.get("Props") == "interface"
        assert "useX" in names

    def test_rust_symbols(self):
        content = (
            "pub struct Config {}\n"
            "pub trait Runner {}\n"
            "pub fn main() {}\n"
        )
        syms = extract_symbols("lib.rs", content, "rust", "mine", "backend")
        names = {s.name for s in syms}
        assert names == {"Config", "Runner", "main"}


def _fake_embed(texts, show_progress=False):
    """Deterministic fake embeddings so tests don't download models."""
    out = []
    for text in texts:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        vec = rng.standard_normal(384).astype("float32")
        out.append(vec)
    return np.stack(out)


class TestCodeSearchEngine:
    def test_index_and_filter_search(self, temp_code_project):
        engine = CodeSearchEngine(temp_code_project)
        with patch.object(CodeSearchEngine, "_embed", side_effect=_fake_embed):
            stats = engine.index_codebase(force=True, show_progress=False)
            assert stats["symbols"] > 0
            assert stats["files"] >= 3
            assert engine.faiss_path.exists()

            mine = engine.search("create user service", k=10, mine_only=True)
            assert mine
            assert all(r["ownership"] == "mine" for r in mine)

            backend = engine.search("user", k=10, group="backend")
            assert backend
            assert all(r["group"] == "backend" for r in backend)

            no_libs = engine.search(
                "helper",
                k=10,
                exclude_groups=["frontend_library", "backend_library"],
            )
            assert all(
                r["group"] not in {"frontend_library", "backend_library"}
                for r in no_libs
            )

            lang = engine.search("button", k=10, language="typescript")
            assert lang
            assert all(r["language"] == "typescript" for r in lang)

    def test_rule_affects_ownership(self, temp_code_project):
        engine = CodeSearchEngine(temp_code_project)
        engine.rules.set_rule("vendor/**", "mine", group="shared")
        with patch.object(CodeSearchEngine, "_embed", side_effect=_fake_embed):
            engine.index_codebase(force=True, show_progress=False)
            results = engine.search("third_party_helper", k=5, path_prefix="vendor/")
            assert results
            assert results[0]["ownership"] == "mine"
            assert results[0]["group"] == "shared"

    def test_refresh_path(self, temp_code_project):
        engine = CodeSearchEngine(temp_code_project)
        with patch.object(CodeSearchEngine, "_embed", side_effect=_fake_embed):
            engine.index_codebase(force=True, show_progress=False)
            target = temp_code_project / "backend" / "api" / "users.py"
            target.write_text(
                target.read_text()
                + "\n\ndef delete_user(user_id: str):\n    return True\n"
            )
            n = engine.refresh_path(target)
            assert n >= 1
            names = {s.name for s in engine._load_symbols() if "users.py" in s.relative_path}
            assert "delete_user" in names

    def test_watch_stop_after(self, temp_code_project):
        engine = CodeSearchEngine(temp_code_project)
        with patch.object(CodeSearchEngine, "_embed", side_effect=_fake_embed):
            engine.index_codebase(force=True, show_progress=False)
            engine.watch(poll_seconds=0.05, stop_after=0.12)

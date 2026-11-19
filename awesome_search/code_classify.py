"""
Language detection and group classification for code symbols.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from awesome_search.code_rules import CodeRuleStore


EXTENSION_LANGUAGE = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".swift": "swift",
    ".scala": "scala",
    ".vue": "vue",
    ".svelte": "svelte",
    ".css": "css",
    ".scss": "scss",
    ".less": "less",
    ".html": "html",
    ".htm": "html",
    ".json": "json",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".md": "markdown",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
}

# Content sniff patterns when extension is missing/ambiguous
CONTENT_LANGUAGE_HINTS = (
    ("python", ("def ", "import ", "from ", "class ", "#!/usr/bin/env python")),
    ("javascript", ("function ", "const ", "let ", "export ", "require(")),
    ("typescript", ("interface ", "type ", ": string", ": number", "export type")),
    ("rust", ("fn ", "impl ", "pub struct", "use crate::", "mod ")),
    ("go", ("package ", "func ", "import (", ":= ")),
    ("java", ("public class", "private ", "package ", "void ")),
)


VALID_GROUPS = {
    "frontend",
    "backend",
    "frontend_library",
    "backend_library",
    "shared",
    "tooling",
    "unknown",
}


def detect_language(path: Path, content: Optional[str] = None) -> str:
    ext = path.suffix.lower()
    if ext in EXTENSION_LANGUAGE:
        return EXTENSION_LANGUAGE[ext]

    sample = (content or "")[:4000]
    best = "unknown"
    best_hits = 0
    for lang, hints in CONTENT_LANGUAGE_HINTS:
        hits = sum(1 for h in hints if h in sample)
        if hits > best_hits:
            best = lang
            best_hits = hits
    return best if best_hits > 0 else "unknown"


def classify_group(
    relative_path: str,
    ownership: str,
    rule_store: CodeRuleStore,
    group_override: Optional[str] = None,
) -> str:
    """
    Combine ownership + path layer into a group label.

    mine + frontend -> frontend
    library + frontend -> frontend_library
    mine + backend -> backend
    library + backend -> backend_library
    """
    if group_override and group_override in VALID_GROUPS:
        return group_override

    layer = rule_store.guess_layer(relative_path)
    ownership = (ownership or "unknown").lower()

    if ownership == "library":
        if layer == "frontend":
            return "frontend_library"
        if layer == "backend":
            return "backend_library"
        return "frontend_library" if _looks_frontend_lib(relative_path) else "backend_library"

    if layer == "frontend":
        return "frontend"
    if layer == "backend":
        return "backend"

    # Weak defaults for mine code with unclear layer
    if ownership == "mine":
        lowered = relative_path.replace("\\", "/").lower()
        if any(x in lowered for x in ("test", "tests", "spec", "__tests__")):
            return "tooling"
        if any(x in lowered for x in ("script", "scripts", "tool", "tools", "ci", ".github")):
            return "tooling"
        return "shared"

    return "unknown"


def _looks_frontend_lib(relative_path: str) -> bool:
    lowered = relative_path.replace("\\", "/").lower()
    return any(
        x in lowered
        for x in (
            "node_modules/",
            ".tsx",
            ".jsx",
            ".vue",
            ".svelte",
            "/react",
            "/vue",
            "/angular",
            "/next/",
            "/nuxt/",
        )
    )


def is_indexable_source(path: Path) -> bool:
    """Return True for likely source files we should symbol-index."""
    if path.name.startswith("."):
        return False
    ext = path.suffix.lower()
    if ext in EXTENSION_LANGUAGE and ext not in {".json", ".toml", ".yaml", ".yml", ".md"}:
        return True
    # Extensionless scripts sometimes matter; skip by default for v1
    return False

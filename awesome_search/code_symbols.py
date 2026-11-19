"""
Lightweight symbol extraction (functions/classes) with line numbers and context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Tuple


@dataclass
class CodeSymbol:
    name: str
    kind: str  # function | class | method | struct | interface | component
    language: str
    relative_path: str
    start_line: int
    end_line: int
    ownership: str
    group: str
    context: str
    signature: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    def search_blob(self) -> str:
        """Text used for semantic embedding / keyword search."""
        return " | ".join(
            [
                self.kind,
                self.name,
                self.signature,
                self.group,
                self.ownership,
                self.language,
                self.relative_path,
                self.context,
            ]
        )


# language -> list of (kind, compiled regex with named group 'name')
_SYMBOL_PATTERNS: Dict[str, List[Tuple[str, Pattern[str]]]] = {
    "python": [
        ("class", re.compile(r"^[ \t]*class[ \t]+(?P<name>[A-Za-z_][\w]*)")),
        ("function", re.compile(r"^[ \t]*(?:async[ \t]+)?def[ \t]+(?P<name>[A-Za-z_][\w]*)[ \t]*\(")),
    ],
    "javascript": [
        ("class", re.compile(r"^[ \t]*(?:export[ \t]+)?(?:default[ \t]+)?class[ \t]+(?P<name>[A-Za-z_][\w]*)")),
        (
            "function",
            re.compile(
                r"^[ \t]*(?:export[ \t]+)?(?:async[ \t]+)?function[ \t]+(?P<name>[A-Za-z_][\w]*)[ \t]*\("
            ),
        ),
        (
            "function",
            re.compile(
                r"^[ \t]*(?:export[ \t]+)?(?:const|let|var)[ \t]+(?P<name>[A-Za-z_][\w]*)[ \t]*=[ \t]*(?:async[ \t]*)?(?:\([^)]*\)|[A-Za-z_][\w]*)[ \t]*=>"
            ),
        ),
        (
            "component",
            re.compile(
                r"^[ \t]*(?:export[ \t]+)?(?:default[ \t]+)?function[ \t]+(?P<name>[A-Z][\w]*)[ \t]*\("
            ),
        ),
    ],
    "typescript": [
        ("class", re.compile(r"^[ \t]*(?:export[ \t]+)?(?:default[ \t]+)?class[ \t]+(?P<name>[A-Za-z_][\w]*)")),
        ("interface", re.compile(r"^[ \t]*(?:export[ \t]+)?interface[ \t]+(?P<name>[A-Za-z_][\w]*)")),
        (
            "function",
            re.compile(
                r"^[ \t]*(?:export[ \t]+)?(?:async[ \t]+)?function[ \t]+(?P<name>[A-Za-z_][\w]*)[ \t]*[(<]"
            ),
        ),
        (
            "function",
            re.compile(
                r"^[ \t]*(?:export[ \t]+)?(?:const|let|var)[ \t]+(?P<name>[A-Za-z_][\w]*)[ \t]*=[ \t]*(?:async[ \t]*)?(?:\([^)]*\)|[A-Za-z_][\w]*)[ \t]*=>"
            ),
        ),
        (
            "component",
            re.compile(
                r"^[ \t]*(?:export[ \t]+)?(?:default[ \t]+)?function[ \t]+(?P<name>[A-Z][\w]*)[ \t]*[(<]"
            ),
        ),
    ],
    "rust": [
        ("struct", re.compile(r"^[ \t]*(?:pub(?:\([^)]*\))?[ \t]+)?struct[ \t]+(?P<name>[A-Za-z_][\w]*)")),
        ("class", re.compile(r"^[ \t]*(?:pub(?:\([^)]*\))?[ \t]+)?enum[ \t]+(?P<name>[A-Za-z_][\w]*)")),  # treat enums as class-like
        ("function", re.compile(r"^[ \t]*(?:pub(?:\([^)]*\))?[ \t]+)?(?:async[ \t]+)?fn[ \t]+(?P<name>[A-Za-z_][\w]*)")),
        ("interface", re.compile(r"^[ \t]*(?:pub(?:\([^)]*\))?[ \t]+)?trait[ \t]+(?P<name>[A-Za-z_][\w]*)")),
    ],
    "go": [
        ("struct", re.compile(r"^[ \t]*type[ \t]+(?P<name>[A-Za-z_][\w]*)[ \t]+struct\b")),
        ("interface", re.compile(r"^[ \t]*type[ \t]+(?P<name>[A-Za-z_][\w]*)[ \t]+interface\b")),
        ("function", re.compile(r"^[ \t]*func[ \t]+(?:\([^)]*\)[ \t]+)?(?P<name>[A-Za-z_][\w]*)[ \t]*\(")),
    ],
    "java": [
        ("class", re.compile(r"^[ \t]*(?:public|protected|private)?[ \t]*(?:abstract|final)?[ \t]*class[ \t]+(?P<name>[A-Za-z_][\w]*)")),
        ("interface", re.compile(r"^[ \t]*(?:public|protected|private)?[ \t]*interface[ \t]+(?P<name>[A-Za-z_][\w]*)")),
        (
            "function",
            re.compile(
                r"^[ \t]*(?:public|protected|private)?[ \t]*(?:static[ \t]+)?(?:final[ \t]+)?[\w<>,\[\]]+[ \t]+(?P<name>[A-Za-z_][\w]*)[ \t]*\("
            ),
        ),
    ],
    "ruby": [
        ("class", re.compile(r"^[ \t]*class[ \t]+(?P<name>[A-Za-z_][\w:]*)")),
        ("function", re.compile(r"^[ \t]*def[ \t]+(?P<name>[A-Za-z_][\w?!=]*)")),
    ],
    "php": [
        ("class", re.compile(r"^[ \t]*(?:abstract[ \t]+)?(?:final[ \t]+)?class[ \t]+(?P<name>[A-Za-z_][\w]*)")),
        ("function", re.compile(r"^[ \t]*(?:public|private|protected)?[ \t]*(?:static[ \t]+)?function[ \t]+(?P<name>[A-Za-z_][\w]*)")),
    ],
    "csharp": [
        ("class", re.compile(r"^[ \t]*(?:public|private|protected|internal)?[ \t]*(?:static[ \t]+)?(?:partial[ \t]+)?class[ \t]+(?P<name>[A-Za-z_][\w]*)")),
        ("interface", re.compile(r"^[ \t]*(?:public|private|protected|internal)?[ \t]*interface[ \t]+(?P<name>[A-Za-z_][\w]*)")),
        (
            "function",
            re.compile(
                r"^[ \t]*(?:public|private|protected|internal)?[ \t]*(?:static[ \t]+)?(?:async[ \t]+)?[\w<>,\[\]]+[ \t]+(?P<name>[A-Za-z_][\w]*)[ \t]*\("
            ),
        ),
    ],
}

# Map closely related languages to pattern sets
_PATTERN_ALIASES = {
    "jsx": "javascript",
    "tsx": "typescript",
    "vue": "javascript",
    "svelte": "javascript",
    "kotlin": "java",
}


def extract_symbols(
    relative_path: str,
    content: str,
    language: str,
    ownership: str,
    group: str,
    context_lines: int = 2,
) -> List[CodeSymbol]:
    lang_key = _PATTERN_ALIASES.get(language, language)
    patterns = _SYMBOL_PATTERNS.get(lang_key, [])
    if not patterns:
        # Still index file-level symbol so the file is searchable
        return [
            CodeSymbol(
                name=Path(relative_path).stem,
                kind="file",
                language=language,
                relative_path=relative_path,
                start_line=1,
                end_line=max(1, content.count("\n") + 1),
                ownership=ownership,
                group=group,
                context=_file_preview(content),
                signature=relative_path,
            )
        ]

    lines = content.splitlines()
    hits: List[Tuple[int, str, str, str]] = []  # line_idx, kind, name, signature

    for idx, line in enumerate(lines):
        for kind, pattern in patterns:
            m = pattern.search(line)
            if not m:
                continue
            name = m.group("name")
            # Skip common false positives in java-like method matcher
            if lang_key in {"java", "csharp"} and name in {
                "if",
                "for",
                "while",
                "switch",
                "catch",
                "return",
                "new",
            }:
                continue
            hits.append((idx, kind, name, line.strip()[:200]))
            break

    symbols: List[CodeSymbol] = []
    for i, (start_idx, kind, name, signature) in enumerate(hits):
        end_idx = hits[i + 1][0] - 1 if i + 1 < len(hits) else len(lines) - 1
        # Prefer a modest span for context; full body can be huge
        body_end = min(end_idx, start_idx + 40)
        ctx_start = max(0, start_idx - context_lines)
        ctx_end = min(len(lines), body_end + 1)
        context = "\n".join(lines[ctx_start:ctx_end]).strip()
        symbols.append(
            CodeSymbol(
                name=name,
                kind=kind,
                language=language,
                relative_path=relative_path,
                start_line=start_idx + 1,
                end_line=body_end + 1,
                ownership=ownership,
                group=group,
                context=context[:1200],
                signature=signature,
            )
        )

    if not symbols:
        symbols.append(
            CodeSymbol(
                name=Path(relative_path).stem,
                kind="file",
                language=language,
                relative_path=relative_path,
                start_line=1,
                end_line=max(1, len(lines)),
                ownership=ownership,
                group=group,
                context=_file_preview(content),
                signature=relative_path,
            )
        )
    return symbols


def _file_preview(content: str, max_chars: int = 400) -> str:
    text = content.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


def refresh_line_numbers(
    content: str,
    language: str,
    existing: List[CodeSymbol],
) -> List[CodeSymbol]:
    """
    Quick re-check after a write: re-extract symbols and preserve metadata
    where names/kinds still match.
    """
    if not existing:
        return []
    ownership = existing[0].ownership
    group = existing[0].group
    relative_path = existing[0].relative_path
    return extract_symbols(
        relative_path=relative_path,
        content=content,
        language=language,
        ownership=ownership,
        group=group,
    )

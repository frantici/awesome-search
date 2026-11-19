"""
Path ownership rules and gitignore-based library heuristics for the code index.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_LIBRARY_PATH_MARKERS = (
    "node_modules/",
    "vendor/",
    "third_party/",
    "third-party/",
    "external/",
    ".venv/",
    "venv/",
    "site-packages/",
    "dist/",
    "build/",
    "target/",  # rust/java build output often not "mine" source of truth
    ".tox/",
    ".mypy_cache/",
    ".pytest_cache/",
    "__pycache__/",
    ".git/",
)

DEFAULT_FRONTEND_MARKERS = (
    "frontend/",
    "front-end/",
    "client/",
    "web/",
    "ui/",
    "app/web/",
    "apps/web/",
    "packages/ui/",
    "src/components/",
    "src/pages/",
    "src/views/",
    "public/",
)

DEFAULT_BACKEND_MARKERS = (
    "backend/",
    "back-end/",
    "server/",
    "api/",
    "services/",
    "apps/api/",
    "src/controllers/",
    "src/routes/",
    "src/handlers/",
)


@dataclass
class PathRule:
    """A glob-ish path prefix rule mapping to ownership and optional group."""

    pattern: str
    ownership: str  # mine | library | unknown
    group: Optional[str] = None  # frontend | backend | frontend_library | backend_library | ...

    def matches(self, relative_path: str) -> bool:
        rel = relative_path.replace("\\", "/").lstrip("./")
        pat = self.pattern.replace("\\", "/").lstrip("./")
        if pat.endswith("/**"):
            prefix = pat[:-3].rstrip("/")
            return rel == prefix or rel.startswith(prefix + "/")
        if pat.endswith("/*"):
            prefix = pat[:-2].rstrip("/")
            if rel == prefix:
                return True
            if not rel.startswith(prefix + "/"):
                return False
            rest = rel[len(prefix) + 1 :]
            return "/" not in rest
        if pat.endswith("/"):
            return rel == pat[:-1] or rel.startswith(pat)
        return rel == pat or rel.startswith(pat.rstrip("/") + "/")


class CodeRuleStore:
    """Persist and resolve ownership/group path rules for a project."""

    def __init__(self, project_root: Path, index_folder: Path):
        self.project_root = Path(project_root)
        self.index_folder = Path(index_folder)
        self.rules_path = self.index_folder / "code_rules.json"
        self.rules: List[PathRule] = []
        self._gitignore_library_prefixes: List[str] = []
        self.load()

    def load(self) -> None:
        self.rules = []
        if self.rules_path.exists():
            with open(self.rules_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data.get("rules", []):
                self.rules.append(
                    PathRule(
                        pattern=item["pattern"],
                        ownership=item.get("ownership", "unknown"),
                        group=item.get("group"),
                    )
                )
        self._gitignore_library_prefixes = self._parse_gitignore_library_hints()

    def save(self) -> None:
        self.index_folder.mkdir(parents=True, exist_ok=True)
        payload = {
            "rules": [asdict(r) for r in self.rules],
        }
        with open(self.rules_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def set_rule(
        self,
        pattern: str,
        ownership: str,
        group: Optional[str] = None,
    ) -> PathRule:
        ownership = ownership.lower().strip()
        if ownership not in {"mine", "library", "unknown"}:
            raise ValueError("ownership must be one of: mine, library, unknown")
        if group is not None:
            group = group.lower().strip()
        # Newest matching rule wins; replace exact pattern if present.
        self.rules = [r for r in self.rules if r.pattern != pattern]
        rule = PathRule(pattern=pattern, ownership=ownership, group=group)
        self.rules.append(rule)
        self.save()
        return rule

    def remove_rule(self, pattern: str) -> bool:
        before = len(self.rules)
        self.rules = [r for r in self.rules if r.pattern != pattern]
        if len(self.rules) != before:
            self.save()
            return True
        return False

    def list_rules(self) -> List[PathRule]:
        return list(self.rules)

    def _parse_gitignore_library_hints(self) -> List[str]:
        gitignore = self.project_root / ".gitignore"
        if not gitignore.exists():
            return []
        hints = []
        try:
            lines = gitignore.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            return []
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("!"):
                continue
            # Strip leading slash; keep directory-ish entries as prefixes.
            entry = line.lstrip("/")
            if entry.endswith("/"):
                hints.append(entry)
            elif entry in {
                "node_modules",
                "vendor",
                "third_party",
                "third-party",
                ".venv",
                "venv",
                "dist",
                "build",
            }:
                hints.append(entry + "/")
        return hints

    def resolve(self, relative_path: str) -> Dict[str, Optional[str]]:
        """
        Resolve ownership and optional group override for a relative path.

        Explicit rules win (last match). Otherwise use defaults + gitignore hints.
        """
        rel = relative_path.replace("\\", "/").lstrip("./")

        matched_ownership = None
        matched_group = None
        for rule in self.rules:
            if rule.matches(rel):
                matched_ownership = rule.ownership
                if rule.group:
                    matched_group = rule.group

        if matched_ownership is None:
            matched_ownership = self._default_ownership(rel)

        return {
            "ownership": matched_ownership,
            "group_override": matched_group,
        }

    def _default_ownership(self, rel: str) -> str:
        lowered = rel.lower()
        for marker in DEFAULT_LIBRARY_PATH_MARKERS:
            if lowered == marker.rstrip("/") or lowered.startswith(marker):
                return "library"
        for hint in self._gitignore_library_prefixes:
            if lowered == hint.rstrip("/") or lowered.startswith(hint.lower()):
                return "library"
        return "mine"

    def guess_layer(self, relative_path: str) -> Optional[str]:
        """Guess frontend/backend layer from path (before ownership merge)."""
        lowered = relative_path.replace("\\", "/").lower()
        for marker in DEFAULT_FRONTEND_MARKERS:
            if marker in lowered or lowered.startswith(marker.rstrip("/")):
                return "frontend"
        for marker in DEFAULT_BACKEND_MARKERS:
            if marker in lowered or lowered.startswith(marker.rstrip("/")):
                return "backend"
        # Extension-based weak hints
        if lowered.endswith((".tsx", ".jsx", ".vue", ".svelte", ".css", ".scss", ".less")):
            return "frontend"
        if lowered.endswith((".py", ".go", ".rs", ".java", ".rb", ".php", ".cs")):
            # Ambiguous; only treat as backend outside obvious frontend dirs
            return "backend"
        return None

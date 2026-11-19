"""
Project environment probe: detect runtimes, venvs, and package managers.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


ENV_FILENAME = "env.json"


@dataclass
class ToolInfo:
    name: str
    available: bool
    path: Optional[str] = None
    version: Optional[str] = None


@dataclass
class EnvReport:
    project_root: str
    stacks: List[str] = field(default_factory=list)
    signals: List[str] = field(default_factory=list)
    tools: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    python: Dict[str, Any] = field(default_factory=dict)
    node: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    confirmed: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EnvStore:
    """Persisted user-confirmed environment choices under .awesome_index/env.json."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).resolve()
        self.index_folder = self.project_root / ".awesome_index"
        self.path = self.index_folder / ENV_FILENAME

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def save(self, data: Dict[str, Any]) -> None:
        self.index_folder.mkdir(parents=True, exist_ok=True)
        payload = dict(data)
        payload["project_root"] = str(self.project_root)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def clear(self) -> bool:
        if self.path.exists():
            self.path.unlink()
            return True
        return False

    def set_value(self, key: str, value: Any) -> Dict[str, Any]:
        data = self.load()
        data[key] = value
        self.save(data)
        return data


def _run_version(cmd: List[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        out = (result.stdout or result.stderr or "").strip().splitlines()
        return out[0].strip() if out else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _tool(name: str, version_args: Optional[List[str]] = None) -> ToolInfo:
    path = shutil.which(name)
    if not path:
        return ToolInfo(name=name, available=False)
    version = None
    if version_args is not None:
        version = _run_version([path, *version_args])
    return ToolInfo(name=name, available=True, path=path, version=version)


def _detect_stacks(root: Path) -> tuple[List[str], List[str]]:
    stacks: List[str] = []
    signals: List[str] = []

    node_markers = [
        "package.json",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "bun.lockb",
        "node_modules",
    ]
    python_markers = [
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "requirements.txt",
        "Pipfile",
        "poetry.lock",
        "environment.yml",
        "conda.yaml",
    ]
    rust_markers = ["Cargo.toml", "Cargo.lock"]
    go_markers = ["go.mod", "go.sum"]

    def hit(markers: List[str], label: str) -> None:
        found = [m for m in markers if (root / m).exists()]
        if found:
            stacks.append(label)
            signals.extend(found)

    hit(node_markers, "node")
    hit(python_markers, "python")
    hit(rust_markers, "rust")
    hit(go_markers, "go")

    # Nested package.json common in monorepos — still mark node
    if "node" not in stacks:
        if any(root.glob("*/package.json")) or any(root.glob("apps/*/package.json")):
            stacks.append("node")
            signals.append("nested package.json")

    return stacks, signals


def probe_environment(project_root: Path) -> EnvReport:
    root = Path(project_root).resolve()
    stacks, signals = _detect_stacks(root)

    tools_list = [
        _tool("python3", ["--version"]),
        _tool("python", ["--version"]),
        _tool("pip3", ["--version"]),
        _tool("pip", ["--version"]),
        _tool("node", ["--version"]),
        _tool("npm", ["--version"]),
        _tool("npx", ["--version"]),
        _tool("yarn", ["--version"]),
        _tool("pnpm", ["--version"]),
        _tool("bun", ["--version"]),
        _tool("cargo", ["--version"]),
        _tool("rustc", ["--version"]),
        _tool("go", ["version"]),
    ]
    tools = {t.name: t for t in tools_list}

    local_venvs = [name for name in (".venv", "venv") if (root / name).is_dir()]
    preferred_python = None
    if tools["python3"].available:
        preferred_python = "python3"
    elif tools["python"].available:
        preferred_python = "python"

    # Prefer project venv interpreters when present
    for venv_name in (".venv", "venv"):
        for candidate in (
            root / venv_name / "bin" / "python",
            root / venv_name / "Scripts" / "python.exe",
        ):
            if candidate.exists():
                preferred_python = str(candidate)
                break

    node_pm = None
    if "node" in stacks:
        if (root / "pnpm-lock.yaml").exists() and tools["pnpm"].available:
            node_pm = "pnpm"
        elif (root / "yarn.lock").exists() and tools["yarn"].available:
            node_pm = "yarn"
        elif (root / "bun.lockb").exists() and tools["bun"].available:
            node_pm = "bun"
        elif tools["npm"].available:
            node_pm = "npm"
        elif tools["pnpm"].available:
            node_pm = "pnpm"
        elif tools["yarn"].available:
            node_pm = "yarn"
        elif tools["bun"].available:
            node_pm = "bun"

    warnings: List[str] = []
    recommendations: List[str] = []

    if "node" in stacks:
        if not tools["node"].available:
            warnings.append(
                "This looks like a Node project (package.json / lockfile), "
                "but `node` is not installed or not on PATH."
            )
            recommendations.append("Install Node.js before running npm/yarn/pnpm scripts.")
        if tools["node"].available and not any(
            tools[x].available for x in ("npm", "yarn", "pnpm", "bun")
        ):
            warnings.append("Node is available but no package manager (npm/yarn/pnpm/bun) was found.")
        if not node_pm and tools["node"].available:
            recommendations.append("Confirm which package manager this repo uses (npm/yarn/pnpm/bun).")

    if "python" in stacks:
        if not tools["python3"].available and not tools["python"].available:
            warnings.append(
                "This looks like a Python project, but neither `python3` nor `python` is on PATH."
            )
            recommendations.append("Install Python 3.10+ and expose `python3` on PATH.")
        elif tools["python"].available and not tools["python3"].available:
            warnings.append(
                "`python` exists but `python3` does not — agents often guess wrong; "
                "confirm the interpreter command."
            )
            recommendations.append("Prefer confirming `python` vs `python3` via `awesomeSearch env confirm`.")
        if not os.environ.get("VIRTUAL_ENV") and not local_venvs:
            recommendations.append(
                "No active virtualenv and no `.venv`/`venv` directory found — "
                "consider creating one to avoid system-site-package confusion."
            )
        elif not os.environ.get("VIRTUAL_ENV") and local_venvs:
            recommendations.append(
                f"Found local venv dir(s) {local_venvs} but VIRTUAL_ENV is not set — "
                "activate before installing packages."
            )

    if "rust" in stacks and not tools["cargo"].available:
        warnings.append("Cargo.toml present but `cargo` is not on PATH.")
    if "go" in stacks and not tools["go"].available:
        warnings.append("go.mod present but `go` is not on PATH.")

    if tools["python"].available and tools["python3"].available:
        if tools["python"].version and tools["python3"].version:
            if tools["python"].version != tools["python3"].version:
                recommendations.append(
                    f"`python` ({tools['python'].version}) and `python3` "
                    f"({tools['python3'].version}) differ — pin which one agents should use."
                )

    store = EnvStore(root)
    confirmed = store.load()

    # Apply confirmed overrides into recommendations surface
    if confirmed.get("python_cmd"):
        preferred_python = confirmed["python_cmd"]
    if confirmed.get("node_package_manager"):
        node_pm = confirmed["node_package_manager"]

    report = EnvReport(
        project_root=str(root),
        stacks=stacks or ["unknown"],
        signals=signals,
        tools={k: asdict(v) for k, v in tools.items()},
        python={
            "preferred_cmd": preferred_python,
            "virtual_env_active": bool(os.environ.get("VIRTUAL_ENV")),
            "virtual_env_path": os.environ.get("VIRTUAL_ENV"),
            "local_venv_dirs": local_venvs,
            "python3": asdict(tools["python3"]),
            "python": asdict(tools["python"]),
        },
        node={
            "package_manager": node_pm,
            "node": asdict(tools["node"]),
            "npm": asdict(tools["npm"]),
            "yarn": asdict(tools["yarn"]),
            "pnpm": asdict(tools["pnpm"]),
            "bun": asdict(tools["bun"]),
        },
        warnings=warnings,
        recommendations=recommendations,
        confirmed=confirmed,
    )
    return report


def confirm_from_probe(project_root: Path, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Persist detection (plus optional overrides) as the confirmed environment."""
    report = probe_environment(project_root)
    data = {
        "stacks": report.stacks,
        "python_cmd": report.python.get("preferred_cmd"),
        "node_package_manager": report.node.get("package_manager"),
        "virtual_env_path": report.python.get("virtual_env_path")
        or (
            str(Path(project_root) / report.python["local_venv_dirs"][0])
            if report.python.get("local_venv_dirs")
            else None
        ),
        "warnings_at_confirm": report.warnings,
    }
    if overrides:
        data.update({k: v for k, v in overrides.items() if v is not None})
    store = EnvStore(project_root)
    store.save(data)
    return store.load()

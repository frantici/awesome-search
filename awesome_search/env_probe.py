"""
Project environment probe: host/shell, runtimes, and confirmed CLI tool allowlist.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ENV_FILENAME = "env.json"

# Dev-relevant CLI tools. Each entry: unix-ish name -> windows-native alternatives.
# Agents should prefer tools marked available in the probe allowlist.
TOOL_CATALOG: Dict[str, Dict[str, Any]] = {
    # search / text
    "rg": {"category": "search", "windows": ["rg.exe"], "notes": "ripgrep; prefer over grep"},
    "grep": {"category": "search", "windows": ["findstr", "Select-String"], "notes": "POSIX grep"},
    "findstr": {"category": "search", "windows": ["findstr"], "unix": ["grep", "rg"], "notes": "Windows findstr"},
    "ag": {"category": "search", "windows": ["ag"], "notes": "the silver searcher"},
    # file / path
    "find": {"category": "files", "windows": ["Get-ChildItem", "where.exe"], "notes": "POSIX find"},
    "fd": {"category": "files", "windows": ["fd.exe"], "notes": "modern find"},
    "ls": {"category": "files", "windows": ["Get-ChildItem", "dir"], "notes": "list dir"},
    "dir": {"category": "files", "windows": ["dir"], "unix": ["ls"], "notes": "cmd dir"},
    "which": {"category": "files", "windows": ["where.exe", "Get-Command"], "notes": "locate binary"},
    "where": {"category": "files", "windows": ["where.exe"], "unix": ["which"], "notes": "Windows where"},
    "realpath": {"category": "files", "windows": [], "notes": "resolve path"},
    "basename": {"category": "files", "windows": [], "notes": "path basename"},
    "dirname": {"category": "files", "windows": [], "notes": "path dirname"},
    # text processing
    "sed": {"category": "text", "windows": [], "notes": "stream editor"},
    "awk": {"category": "text", "windows": [], "notes": "text processing"},
    "cut": {"category": "text", "windows": [], "notes": "cut fields"},
    "tr": {"category": "text", "windows": [], "notes": "translate chars"},
    "sort": {"category": "text", "windows": ["Sort-Object"], "notes": "sort lines"},
    "uniq": {"category": "text", "windows": [], "notes": "unique lines"},
    "wc": {"category": "text", "windows": ["Measure-Object"], "notes": "word/line count"},
    "head": {"category": "text", "windows": ["Select-Object -First"], "notes": "first lines"},
    "tail": {"category": "text", "windows": ["Select-Object -Last", "Get-Content -Tail"], "notes": "last lines"},
    "xargs": {"category": "text", "windows": [], "notes": "build command lines"},
    # network / data
    "curl": {"category": "net", "windows": ["curl.exe", "Invoke-WebRequest"], "notes": "HTTP client"},
    "wget": {"category": "net", "windows": ["wget.exe", "Invoke-WebRequest"], "notes": "download"},
    "jq": {"category": "data", "windows": ["jq.exe"], "notes": "JSON processor"},
    "yq": {"category": "data", "windows": ["yq.exe"], "notes": "YAML processor"},
    # archive / hash
    "tar": {"category": "archive", "windows": ["tar.exe"], "notes": "archives"},
    "unzip": {"category": "archive", "windows": ["Expand-Archive", "tar.exe"], "notes": "unzip"},
    "zip": {"category": "archive", "windows": ["Compress-Archive"], "notes": "zip"},
    "sha256sum": {"category": "hash", "windows": ["Get-FileHash", "certutil"], "notes": "sha256"},
    "md5sum": {"category": "hash", "windows": ["Get-FileHash"], "notes": "md5"},
    # vcs / build helpers
    "git": {"category": "vcs", "windows": ["git.exe"], "notes": "git"},
    "make": {"category": "build", "windows": ["make.exe", "nmake"], "notes": "make"},
    "cmake": {"category": "build", "windows": ["cmake.exe"], "notes": "cmake"},
    # shells / editors often used in pipelines
    "bash": {"category": "shell", "windows": ["bash.exe"], "notes": "bash"},
    "zsh": {"category": "shell", "windows": [], "notes": "zsh"},
    "pwsh": {"category": "shell", "windows": ["pwsh.exe", "powershell.exe"], "notes": "PowerShell 7+"},
    "powershell": {"category": "shell", "windows": ["powershell.exe"], "notes": "Windows PowerShell"},
    "wsl": {"category": "shell", "windows": ["wsl.exe"], "notes": "WSL bridge from Windows"},
    # runtimes (also probed separately)
    "python3": {"category": "runtime", "windows": ["python.exe", "py.exe"], "notes": "Python 3"},
    "python": {"category": "runtime", "windows": ["python.exe", "py.exe"], "notes": "Python"},
    "py": {"category": "runtime", "windows": ["py.exe"], "notes": "Windows Python launcher"},
    "node": {"category": "runtime", "windows": ["node.exe"], "notes": "Node.js"},
    "npm": {"category": "runtime", "windows": ["npm.cmd", "npm.exe"], "notes": "npm"},
    "npx": {"category": "runtime", "windows": ["npx.cmd"], "notes": "npx"},
    "yarn": {"category": "runtime", "windows": ["yarn.cmd"], "notes": "yarn"},
    "pnpm": {"category": "runtime", "windows": ["pnpm.cmd"], "notes": "pnpm"},
    "cargo": {"category": "runtime", "windows": ["cargo.exe"], "notes": "Rust cargo"},
    "go": {"category": "runtime", "windows": ["go.exe"], "notes": "Go"},
    "docker": {"category": "runtime", "windows": ["docker.exe"], "notes": "Docker"},
}

# Names we always probe for the default allowlist
DEFAULT_PROBE_TOOLS = [
    "rg",
    "grep",
    "findstr",
    "find",
    "fd",
    "which",
    "where",
    "sed",
    "awk",
    "cut",
    "tr",
    "sort",
    "uniq",
    "wc",
    "head",
    "tail",
    "xargs",
    "curl",
    "wget",
    "jq",
    "yq",
    "tar",
    "unzip",
    "git",
    "make",
    "bash",
    "zsh",
    "pwsh",
    "powershell",
    "wsl",
    "python3",
    "python",
    "py",
    "node",
    "npm",
    "npx",
    "yarn",
    "pnpm",
    "cargo",
    "go",
    "docker",
]


@dataclass
class ToolInfo:
    name: str
    available: bool
    path: Optional[str] = None
    version: Optional[str] = None
    category: Optional[str] = None
    windows_equiv: List[str] = field(default_factory=list)
    unix_equiv: List[str] = field(default_factory=list)
    source: str = "probe"  # probe | confirmed | installed
    notes: Optional[str] = None


@dataclass
class EnvReport:
    project_root: str
    host: Dict[str, Any] = field(default_factory=dict)
    stacks: List[str] = field(default_factory=list)
    signals: List[str] = field(default_factory=list)
    tools: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    available_tools: List[str] = field(default_factory=list)
    missing_common_tools: List[str] = field(default_factory=list)
    tool_equivalents: Dict[str, List[str]] = field(default_factory=dict)
    python: Dict[str, Any] = field(default_factory=dict)
    node: Dict[str, Any] = field(default_factory=dict)
    preferred_script_language: Optional[str] = None
    agent_rules: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    confirmed: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EnvStore:
    """Persisted user/agent-confirmed environment under .awesome_index/env.json."""

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

    def list_tools(self) -> Dict[str, Dict[str, Any]]:
        data = self.load()
        tools = data.get("tools") or {}
        return tools if isinstance(tools, dict) else {}

    def add_tool(
        self,
        name: str,
        *,
        path: Optional[str] = None,
        version: Optional[str] = None,
        source: str = "confirmed",
        notes: Optional[str] = None,
        require_exists: bool = True,
    ) -> Dict[str, Any]:
        """
        Register a CLI tool after the agent confirmed it exists or installed it.

        If require_exists is True, the binary must resolve on PATH (or path=).
        """
        name = name.strip()
        if not name:
            raise ValueError("tool name required")

        resolved = path
        if not resolved:
            resolved = shutil.which(name)
        if require_exists:
            if not resolved:
                raise FileNotFoundError(
                    f"Tool `{name}` not found on PATH. Install it, then: "
                    f"awesomeSearch env tool add {name} --after-install"
                )
            if path and not Path(path).exists() and not shutil.which(path):
                # Explicit path must exist unless it is a PATH name
                if "/" in path or "\\" in path:
                    raise FileNotFoundError(f"Path does not exist: {path}")

        meta = TOOL_CATALOG.get(name, {})
        entry = {
            "name": name,
            "available": True,
            "path": resolved or path,
            "version": version or (_run_version([resolved, "--version"]) if resolved else None),
            "category": meta.get("category"),
            "windows_equiv": list(meta.get("windows") or []),
            "unix_equiv": list(meta.get("unix") or []),
            "source": source,
            "notes": notes or meta.get("notes"),
        }
        data = self.load()
        tools = data.get("tools") or {}
        if not isinstance(tools, dict):
            tools = {}
        tools[name] = entry
        data["tools"] = tools
        self.save(data)
        return entry

    def remove_tool(self, name: str) -> bool:
        data = self.load()
        tools = data.get("tools") or {}
        if name not in tools:
            return False
        del tools[name]
        data["tools"] = tools
        self.save(data)
        return True


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
    meta = TOOL_CATALOG.get(name, {})
    path = shutil.which(name)
    # On Windows, also try common .exe/.cmd forms via which (usually handled)
    if not path and os.name == "nt":
        for alt in meta.get("windows") or []:
            if alt.endswith(".exe") or alt.endswith(".cmd"):
                path = shutil.which(alt)
                if path:
                    break
    if not path:
        return ToolInfo(
            name=name,
            available=False,
            category=meta.get("category"),
            windows_equiv=list(meta.get("windows") or []),
            unix_equiv=list(meta.get("unix") or []),
            notes=meta.get("notes"),
        )
    version = None
    if version_args is not None:
        version = _run_version([path, *version_args])
    elif name in {"git", "node", "npm", "python", "python3", "go", "cargo", "docker", "rg", "jq"}:
        version = _run_version([path, "--version"])
    return ToolInfo(
        name=name,
        available=True,
        path=path,
        version=version,
        category=meta.get("category"),
        windows_equiv=list(meta.get("windows") or []),
        unix_equiv=list(meta.get("unix") or []),
        notes=meta.get("notes"),
    )


def _detect_shell() -> Dict[str, Any]:
    """Detect the shell that launched this process (best-effort)."""
    shell_env = os.environ.get("SHELL") or os.environ.get("ComSpec") or ""
    ps_mod = os.environ.get("PSModulePath", "")
    # PowerShell sets PSModulePath; also check parent-ish hints
    forced = os.environ.get("AWESOME_SHELL")  # optional override for agents/tests

    name = "unknown"
    family = "unknown"
    if forced:
        name = forced.lower()
    elif os.environ.get("WSL_DISTRO_NAME") and shell_env:
        base = Path(shell_env).name.lower()
        name = base or "bash"
        family = "unix"
    elif ps_mod and (os.name == "nt" or "powershell" in shell_env.lower() or "pwsh" in shell_env.lower()):
        # Running under PowerShell on Windows (or pwsh elsewhere)
        if shutil.which("pwsh") and ("pwsh" in (os.environ.get("PSCommandPath") or "").lower() or True):
            # Prefer distinguishing pwsh vs powershell when possible
            ps_cmd = (os.environ.get("PSCommandPath") or os.environ.get("POWERSHELL_DISTRIBUTION_CHANNEL") or "").lower()
            if "pwsh" in ps_cmd or os.environ.get("POWERSHELL_DISTRIBUTION_CHANNEL"):
                name = "pwsh"
            else:
                name = "powershell" if os.name == "nt" else "pwsh"
        else:
            name = "powershell"
        family = "powershell"
    elif shell_env:
        base = Path(shell_env).name.lower()
        name = base
        family = "unix" if base in {"bash", "zsh", "sh", "ash", "fish", "dash", "ksh"} else "unknown"
        if base in {"fish"}:
            family = "unix"
    elif os.name == "nt":
        name = "cmd"
        family = "cmd"

    # Refine from sys
    if name == "unknown" and os.name != "nt":
        name = "sh"
        family = "unix"

    return {
        "name": name,
        "family": family,
        "SHELL": shell_env or None,
        "ComSpec": os.environ.get("ComSpec"),
        "PSModulePath_set": bool(ps_mod),
    }


def _detect_host() -> Dict[str, Any]:
    system = platform.system()  # Windows, Linux, Darwin
    release = platform.release()
    machine = platform.machine()
    is_wsl = False
    wsl_distro = os.environ.get("WSL_DISTRO_NAME")
    wsl_interop = os.environ.get("WSL_INTEROP")
    if wsl_distro or wsl_interop:
        is_wsl = True
    elif system == "Linux":
        try:
            proc_ver = Path("/proc/version").read_text(encoding="utf-8", errors="ignore").lower()
            if "microsoft" in proc_ver or "wsl" in proc_ver:
                is_wsl = True
        except OSError:
            pass

    # Windows host with wsl.exe available (agent may be in PowerShell)
    wsl_exe = shutil.which("wsl") or shutil.which("wsl.exe")
    in_windows = system == "Windows" or os.name == "nt"
    in_unix = system in {"Linux", "Darwin"} or (os.name == "posix" and not in_windows)

    context = "linux"
    if system == "Darwin":
        context = "macos"
    elif is_wsl:
        context = "wsl"
    elif in_windows:
        context = "windows"

    shell = _detect_shell()

    # How agents should invoke work
    invoke = {
        "mode": "native",
        "workdir_style": "posix" if context in {"linux", "macos", "wsl"} else "windows",
        "bridge": None,
    }
    if context == "windows" and wsl_exe:
        invoke = {
            "mode": "windows_with_wsl",
            "workdir_style": "windows",
            "bridge": {
                "binary": wsl_exe,
                "pattern": "write script/file on the Windows or \\\\wsl$ side, then: "
                "wsl.exe -d <distro> -- <cmd> <unix-path>",
                "anti_pattern": "Do NOT paste long bash pipelines into powershell -c / wsl bash -c with nested quoting",
            },
        }
    elif context == "wsl":
        invoke = {
            "mode": "inside_wsl",
            "workdir_style": "posix",
            "bridge": {
                "binary": None,
                "pattern": "You are already inside WSL — run bash/python directly; do not call wsl.exe from here unless targeting another distro",
                "anti_pattern": "Do not wrap commands in powershell.exe unless you intentionally need Windows-side tools",
            },
        }

    return {
        "os": system,
        "os_release": release,
        "arch": machine,
        "context": context,  # windows | wsl | linux | macos
        "is_wsl": is_wsl,
        "wsl_distro": wsl_distro,
        "wsl_exe": wsl_exe,
        "shell": shell,
        "invoke": invoke,
        "python_runtime": sys.executable,
        "platform": platform.platform(),
    }


def _preferred_script_language(host: Dict[str, Any], tools: Dict[str, ToolInfo]) -> str:
    context = host.get("context")
    shell_family = (host.get("shell") or {}).get("family")
    if context == "windows" and shell_family == "powershell":
        if tools.get("python3", ToolInfo("python3", False)).available or tools.get(
            "python", ToolInfo("python", False)
        ).available or tools.get("py", ToolInfo("py", False)).available:
            return "py"  # still prefer Python for portable small tools
        return "ps1"
    if tools.get("python3", ToolInfo("python3", False)).available or tools.get(
        "python", ToolInfo("python", False)
    ).available:
        return "py"
    if tools.get("node", ToolInfo("node", False)).available:
        return "js"
    if shell_family == "powershell":
        return "ps1"
    return "sh"


def _agent_rules(host: Dict[str, Any], preferred_script: str, available: List[str]) -> List[str]:
    rules = [
        "Run `awesomeSearch env` (or `env detect --json`) at the start of a session and obey host.invoke + available_tools.",
        "Only use CLI tools listed in available_tools (or confirmed tools). If a tool is missing, install it then `awesomeSearch env tool add <name> --after-install`.",
        f"Preferred language for small helper scripts: {preferred_script}.",
    ]
    context = host.get("context")
    invoke = host.get("invoke") or {}
    if context == "windows":
        rules.append(
            "Current shell context is Windows. Do not emit raw bash pipelines into PowerShell."
        )
        if invoke.get("mode") == "windows_with_wsl":
            rules.append(
                "For Linux tooling: write a file, then invoke via wsl.exe -d <distro> -- <command> <path>. "
                "Never concatenate multi-stage bash into powershell -c."
            )
            rules.append(
                "Map common needs: grep→rg/findstr, wc -l→(rg -c / Measure-Object), "
                "which→where.exe, cat→Get-Content, open JSON→jq if available else python."
            )
    elif context == "wsl":
        rules.append(
            "You are inside WSL. Use POSIX commands from available_tools; prefer rg over grep when both exist."
        )
    if "rg" in available:
        rules.append("Prefer `rg` over `grep`/`findstr` for code search.")
    if "jq" not in available:
        rules.append("jq not available — use python -c / node for JSON, or install jq and register it.")
    return rules


def _detect_stacks(root: Path) -> Tuple[List[str], List[str]]:
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

    if "node" not in stacks:
        if any(root.glob("*/package.json")) or any(root.glob("apps/*/package.json")):
            stacks.append("node")
            signals.append("nested package.json")

    return stacks, signals


def probe_cli_tools(extra_names: Optional[List[str]] = None) -> Dict[str, ToolInfo]:
    names = list(DEFAULT_PROBE_TOOLS)
    if extra_names:
        for n in extra_names:
            if n not in names:
                names.append(n)
    return {n: _tool(n) for n in names}


def probe_environment(project_root: Path) -> EnvReport:
    root = Path(project_root).resolve()
    host = _detect_host()
    stacks, signals = _detect_stacks(root)

    store = EnvStore(root)
    confirmed = store.load()
    confirmed_tools = store.list_tools()

    tools = probe_cli_tools(extra_names=list(confirmed_tools.keys()))
    # Merge confirmed tools (agent-added) as available even if re-probed
    for name, entry in confirmed_tools.items():
        info = tools.get(name) or ToolInfo(name=name, available=False)
        info.available = True
        info.path = entry.get("path") or info.path
        info.version = entry.get("version") or info.version
        info.source = entry.get("source") or "confirmed"
        info.notes = entry.get("notes") or info.notes
        tools[name] = info

    # Runtime-focused subset for python/node sections
    runtime_tools = {
        n: tools[n]
        for n in (
            "python3",
            "python",
            "py",
            "pip3",
            "pip",
            "node",
            "npm",
            "npx",
            "yarn",
            "pnpm",
            "bun",
            "cargo",
            "rustc",
            "go",
            "docker",
        )
        if n in tools
    }
    # Ensure pip variants probed
    for extra in ("pip3", "pip", "bun", "rustc"):
        if extra not in tools:
            tools[extra] = _tool(extra, ["--version"] if extra != "rustc" else ["--version"])
            if extra in ("pip3", "pip", "bun", "rustc"):
                runtime_tools[extra] = tools[extra]

    local_venvs = [name for name in (".venv", "venv") if (root / name).is_dir()]
    preferred_python = None
    if tools.get("python3") and tools["python3"].available:
        preferred_python = "python3"
    elif tools.get("python") and tools["python"].available:
        preferred_python = "python"
    elif tools.get("py") and tools["py"].available:
        preferred_python = "py"

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
        if (root / "pnpm-lock.yaml").exists() and tools.get("pnpm") and tools["pnpm"].available:
            node_pm = "pnpm"
        elif (root / "yarn.lock").exists() and tools.get("yarn") and tools["yarn"].available:
            node_pm = "yarn"
        elif tools.get("npm") and tools["npm"].available:
            node_pm = "npm"
        elif tools.get("pnpm") and tools["pnpm"].available:
            node_pm = "pnpm"
        elif tools.get("yarn") and tools["yarn"].available:
            node_pm = "yarn"

    if confirmed.get("python_cmd"):
        preferred_python = confirmed["python_cmd"]
    if confirmed.get("node_package_manager"):
        node_pm = confirmed["node_package_manager"]

    available = sorted([n for n, t in tools.items() if t.available])
    # Common tools agents reach for that are often missing on minimal distros / Windows
    commonly_assumed = ["rg", "grep", "jq", "curl", "sed", "awk", "wc", "head", "tail", "xargs", "git"]
    if host.get("context") == "windows":
        commonly_assumed = ["rg", "git", "curl", "jq", "wsl", "python", "node", "where"]
    missing_common = [n for n in commonly_assumed if n not in available]

    equivalents: Dict[str, List[str]] = {}
    for name, meta in TOOL_CATALOG.items():
        alts = []
        if host.get("context") == "windows":
            alts = list(meta.get("windows") or [])
        else:
            alts = list(meta.get("unix") or [])
        # Also show cross-map for missing unix tools on windows
        if host.get("context") == "windows" and name not in available:
            win_alts = [a for a in (meta.get("windows") or []) if a.split(".")[0] in available or a in available]
            if win_alts:
                equivalents[name] = win_alts
        elif alts:
            equivalents[name] = alts

    preferred_script = confirmed.get("preferred_script_language") or _preferred_script_language(
        host, tools
    )
    agent_rules = _agent_rules(host, preferred_script, available)

    warnings: List[str] = []
    recommendations: List[str] = []

    if host.get("context") == "windows" and host.get("shell", {}).get("family") == "powershell":
        warnings.append(
            "Host shell looks like PowerShell on Windows. "
            "Do not run bash/POSIX pipelines directly — use available Windows tools "
            "or write a file and invoke WSL."
        )
        if host.get("wsl_exe"):
            recommendations.append(
                "WSL is available. Preferred bridge: write script → "
                "wsl.exe -d <distro> -- python3 /home/.../script.py"
            )
        else:
            warnings.append("wsl.exe not found on PATH — Linux-only commands will fail here.")

    if host.get("context") == "wsl":
        recommendations.append(
            f"Inside WSL distro `{host.get('wsl_distro') or 'unknown'}`. Use POSIX tools from available_tools."
        )

    if "node" in stacks:
        if not (tools.get("node") and tools["node"].available):
            warnings.append(
                "This looks like a Node project, but `node` is not installed or not on PATH."
            )
            recommendations.append("Install Node.js, then: awesomeSearch env tool add node --after-install")
        if tools.get("node") and tools["node"].available and not any(
            tools.get(x) and tools[x].available for x in ("npm", "yarn", "pnpm", "bun")
        ):
            warnings.append("Node is available but no package manager (npm/yarn/pnpm/bun) was found.")

    if "python" in stacks:
        if not any(tools.get(x) and tools[x].available for x in ("python3", "python", "py")):
            warnings.append(
                "This looks like a Python project, but neither `python3` nor `python`/`py` is on PATH."
            )
        elif tools.get("python") and tools["python"].available and not (
            tools.get("python3") and tools["python3"].available
        ):
            warnings.append(
                "`python` exists but `python3` does not — confirm the interpreter command."
            )
        if not os.environ.get("VIRTUAL_ENV") and not local_venvs:
            recommendations.append(
                "No active virtualenv and no `.venv`/`venv` directory found."
            )
        elif not os.environ.get("VIRTUAL_ENV") and local_venvs:
            recommendations.append(
                f"Found local venv dir(s) {local_venvs} but VIRTUAL_ENV is not set — activate it."
            )

    if "rust" in stacks and not (tools.get("cargo") and tools["cargo"].available):
        warnings.append("Cargo.toml present but `cargo` is not on PATH.")
    if "go" in stacks and not (tools.get("go") and tools["go"].available):
        warnings.append("go.mod present but `go` is not on PATH.")

    if "jq" not in available and ("node" in stacks or "python" in stacks):
        recommendations.append(
            "jq missing — use python/node for JSON, or install jq and "
            "`awesomeSearch env tool add jq --after-install`."
        )
    if "rg" not in available:
        recommendations.append(
            "ripgrep (`rg`) missing — agents often assume it. Install and "
            "`awesomeSearch env tool add rg --after-install`."
        )

    report = EnvReport(
        project_root=str(root),
        host=host,
        stacks=stacks or ["unknown"],
        signals=signals,
        tools={k: asdict(v) for k, v in tools.items()},
        available_tools=available,
        missing_common_tools=missing_common,
        tool_equivalents=equivalents,
        python={
            "preferred_cmd": preferred_python,
            "virtual_env_active": bool(os.environ.get("VIRTUAL_ENV")),
            "virtual_env_path": os.environ.get("VIRTUAL_ENV"),
            "local_venv_dirs": local_venvs,
            "python3": asdict(tools["python3"]) if "python3" in tools else None,
            "python": asdict(tools["python"]) if "python" in tools else None,
            "py": asdict(tools["py"]) if "py" in tools else None,
        },
        node={
            "package_manager": node_pm,
            "node": asdict(tools["node"]) if "node" in tools else None,
            "npm": asdict(tools["npm"]) if "npm" in tools else None,
            "yarn": asdict(tools["yarn"]) if "yarn" in tools else None,
            "pnpm": asdict(tools["pnpm"]) if "pnpm" in tools else None,
        },
        preferred_script_language=preferred_script,
        agent_rules=agent_rules,
        warnings=warnings,
        recommendations=recommendations,
        confirmed=confirmed,
    )
    return report


def confirm_from_probe(project_root: Path, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Persist detection (plus optional overrides) as the confirmed environment."""
    report = probe_environment(project_root)
    # Snapshot currently available tools into confirmed allowlist
    tools_snap = {}
    for name in report.available_tools:
        info = report.tools.get(name) or {}
        tools_snap[name] = {
            **info,
            "source": (report.confirmed.get("tools") or {}).get(name, {}).get("source")
            or info.get("source")
            or "probe",
        }
    data = {
        "stacks": report.stacks,
        "host": {
            "context": report.host.get("context"),
            "os": report.host.get("os"),
            "shell": report.host.get("shell"),
            "wsl_distro": report.host.get("wsl_distro"),
            "invoke": report.host.get("invoke"),
        },
        "python_cmd": report.python.get("preferred_cmd"),
        "node_package_manager": report.node.get("package_manager"),
        "preferred_script_language": report.preferred_script_language,
        "virtual_env_path": report.python.get("virtual_env_path")
        or (
            str(Path(project_root) / report.python["local_venv_dirs"][0])
            if report.python.get("local_venv_dirs")
            else None
        ),
        "tools": tools_snap,
        "available_tools": report.available_tools,
        "agent_rules": report.agent_rules,
        "warnings_at_confirm": report.warnings,
    }
    if overrides:
        data.update({k: v for k, v in overrides.items() if v is not None})
    store = EnvStore(project_root)
    # Preserve tools previously added with source installed/confirmed if still present
    existing = store.list_tools()
    for name, entry in existing.items():
        if entry.get("source") in {"installed", "confirmed"}:
            data.setdefault("tools", {})[name] = entry
            if name not in data.get("available_tools", []):
                data.setdefault("available_tools", []).append(name)
    store.save(data)
    return store.load()

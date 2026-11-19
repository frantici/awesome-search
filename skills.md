# AwesomeSearch — agent skills

Use this file as the default playbook when working in a repo that has AwesomeSearch installed (`awesomeSearch` on PATH).

## 1. First action every session

```bash
awesomeSearch env detect --json
# or human-readable:
awesomeSearch env
```

Read and obey, in order:

1. **`host.context`** — `windows` | `wsl` | `linux` | `macos`
2. **`host.shell`** — e.g. `powershell`, `pwsh`, `bash`, `zsh`
3. **`host.invoke`** — how to run commands (native vs WSL bridge)
4. **`available_tools`** — only use these CLIs by default
5. **`preferred_script_language`** — `py` | `ps1` | `js` | `sh` for small helpers
6. **`agent_rules`** — hard constraints for this machine

If there are warnings (exit code 2), fix or acknowledge them before install/build loops.

Confirm once per machine/project when the detection looks right:

```bash
awesomeSearch env confirm --yes
awesomeSearch env confirm --yes --python-cmd python3 --script-language py
```

## 2. Windows + WSL (common failure mode)

If `host.context` is `windows` and the shell family is PowerShell:

- **Do not** paste bash pipelines into `powershell -c` or `wsl bash -c '…huge…'`.
- **Do** write a file, then invoke WSL/Python with a path:

```powershell
# after writing C:\Users\...\tmp\task.py or \\wsl$\Distro\home\...\task.py
wsl.exe -d Ubuntu -- python3 /home/you/proj/tmp/task.py
```

If `host.context` is `wsl`, you are already in Linux — run POSIX tools directly; do not wrap everything in `wsl.exe`.

Rough equivalents when a Unix tool is missing on Windows:

| Need | Prefer if available | Windows-leaning fallback |
|------|---------------------|--------------------------|
| search | `rg` | `findstr`, `Select-String` |
| which | `which` / `command -v` | `where.exe`, `Get-Command` |
| line count | `wc -l` | `Measure-Object -Line`, `rg -c` |
| head/tail | `head` / `tail` | `Select-Object -First/-Last` |
| download | `curl` | `curl.exe`, `Invoke-WebRequest` |
| JSON | `jq` | `python -c`, `node -e` |

## 3. Tool allowlist — grow it after install

Default `available_tools` is a **probed subset**. If you install something the job needs:

```bash
# verify + register
awesomeSearch env tool probe rg
# or after you installed it:
awesomeSearch env tool add jq --after-install
awesomeSearch env tool add rg --path /usr/bin/rg --after-install
awesomeSearch env tool list
awesomeSearch env tool remove oldtool
```

Do **not** invent pipelines that use tools absent from `available_tools` / confirmed tools.

## 4. Code index (ownership-aware)

```bash
awesomeSearch code index
awesomeSearch code search "create user" --mine --group backend
awesomeSearch code watch
# during npm update / mass installs:
awesomeSearch code watch stop --minutes 10
awesomeSearch code watch resume
awesomeSearch code purge          # wipe vectors; keeps path rules
```

Path rules:

```bash
awesomeSearch code rule set "vendor/**" library
awesomeSearch code rule set "vendor/mycompany/**" mine --group backend
```

## 5. Doc search (`llms/`)

```bash
awesomeIndexer
awesomeSearch "rate limiting" --context
```

## 6. Minimal command subset (default agent surface)

Prefer this short set unless the user asks for more:

| Command | Why |
|---------|-----|
| `awesomeSearch env` | host/shell/tools — run first |
| `awesomeSearch env tool add/probe` | extend allowlist after install |
| `awesomeSearch code search … --mine` | find *our* symbols |
| `awesomeSearch code watch stop --minutes N` | pause during dependency storms |
| `awesomeIndexer` / `awesomeSearch <query>` | doc/RAG when `llms/` exists |

Deeper detail (`code rule`, `env set`, purge, TUI, prep) is available on request — do not dump every subcommand into the first plan.

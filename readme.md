# AwesomeSearch - LLM Instruction Search Tool

A local FAISS IVF + e-means search tool for quickly finding relevant information in LLM instruction documents.

## Installation

### Package Structure
Create this directory structure:

```
awesome-search/
├── setup.py
├── README.md
└── awesome_search/
    ├── __init__.py
    └── cli.py
```

### Files

1. **setup.py** - Use the setup.py artifact provided
2. **awesome_search/cli.py** - Use the main CLI tool code provided
3. **awesome_search/__init__.py** - Create empty file or add: `__version__ = "0.1.0"`

### Install Globally

```bash
cd awesome-search
pip install -e .
```

This installs these global commands:
- `awesomeIndexer` - Build/update the LLM-doc search index
- `awesomeSearch` - Search docs, code (`awesomeSearch code ...`), or probe env (`awesomeSearch env ...`)
- `awesomeTUI` - Terminal UI for indexing and searching docs
- `awesomePrep` - Convert PDF/DOCX/web/markdown into chunk-ready markdown (optional deps: `pip install 'awesome-search[prep]'`)

## Usage

### Project Structure Required

Your project must have an `llms` folder containing your instruction documents:

```
my-project/
├── llms/
│   ├── azure_python_llm.md
│   ├── openai_guidelines.md
│   └── prompt_engineering.txt
├── src/
└── other files...
```

The `.awesome_index/` folder will be created automatically to store the index.

### Indexing Documents

**Index all documents in llms folder:**
```bash
cd my-project
awesomeIndexer
```

**Index a specific file:**
```bash
awesomeIndexer "azure_python_llm.md"
```

**Force re-index (ignore hash checks):**
```bash
awesomeIndexer --force
```

### Searching

**Basic search:**
```bash
awesomeSearch "Azure functions and python"
```

**Search with source filter:**
```bash
awesomeSearch "Azure functions and python" "azure_python_llm.md"
```

**Search with custom number of results:**
```bash
awesomeSearch "Azure functions and python" "" 20
```

**Search with relevancy threshold:**
```bash
awesomeSearch "Azure functions and python" "azure_python_llm.md" 20 0.8
```

**Search with context mode (shows surrounding lines):**
```bash
awesomeSearch "Azure functions and python" --context
```

**Search with custom context window:**
```bash
awesomeSearch "Azure functions and python" --context --context-lines 100
```

### Arguments

**awesomeSearch:**
1. `query` (required) - What to search for
2. `source` (optional) - Filter by specific source filename
3. `num_results` (optional, default: 5) - Number of results to return
4. `threshold` (optional, default: 0.0) - Maximum distance threshold (lower = more relevant)
5. `--context` (flag) - Enable context mode: fetch surrounding lines from source file
6. `--context-lines N` (optional, default: 50) - Number of lines before/after to show in context mode

**awesomeIndexer:**
1. `file` (optional) - Specific file to index, or omit to index all
2. `--force` (flag) - Force re-indexing

## How It Works

1. **Automatic Change Detection**: The indexer maintains a hash log of all files in `.awesome_index/file_hashes.json`
2. **Smart Updates**: Only re-indexes files that have changed
3. **Local Storage**: Index stored in `.awesome_index/` within your project (add to .gitignore)
4. **Semantic Search**: Uses sentence embeddings for meaning-based search, not just keyword matching
5. **FAISS IVF e-means engine**: Primary vector index is `IndexIVFFlat` with coarse centroids trained by FAISS e-means (`faiss.contrib.e_means`) instead of Lloyd k-means (falls back to flat L2 only for very small corpora)
6. **Context Mode**: Tracks character positions to fetch surrounding lines from original files, giving you complete context around matches

## Context Mode Features

When you use `--context`, the search tool:
- Identifies the exact line numbers where the chunk appears in the source file
- Fetches N lines before and after (default: 50)
- Displays with line numbers for easy reference
- Highlights the matched chunk with `>>>` markers
- Shows total lines in file and context range

**Example output:**
```
[Result 1] Score: 0.8532
Source: llms/python-jira.llms.md
Chunk: 5/56
Lines: 45-48 (of 1250)
Context: Lines 1-98
--------------------------------------------------------------------------------

--- CONTEXT BEFORE ---
   1 | # JIRA Python Integration Guide
   2 | 
   3 | ## Authentication Setup
...
  44 | 
--- MATCHED CHUNK ---
  45 | >>> jira = JIRA(
  46 | >>>     server='https://your-jira-instance.com',
  47 | >>>     oauth={
  48 | >>>         'access_token': 'access_token',

--- CONTEXT AFTER ---
  49 |         'access_token_secret': 'secret',
  50 |         'consumer_key': 'key',
...
```

## Example Agent Usage

Your agent can call these commands from the project root:

```bash
# Agent indexes new document
awesomeIndexer "new_guidelines.md"

# Agent searches for information
awesomeSearch "rate limiting best practices" "" 10 0.5
```

## File Hash Log

The hash log (`.awesome_index/file_hashes.json`) tracks file changes:

```json
{
  "llms/azure_python_llm.md": "a1b2c3d4...",
  "llms/openai_guidelines.md": "e5f6g7h8..."
}
```

## Code symbol index

Index project source as functions/classes with ownership (`mine` / `library`) and group tags (`frontend`, `backend`, `frontend_library`, `backend_library`, `shared`, `tooling`). Defaults treat `node_modules/`, `vendor/`, `.venv/`, etc. as library, and also honor directory hints from `.gitignore`. Explicit path rules override defaults.

```bash
# Build / refresh the code index (stores under .awesome_index/code/)
awesomeSearch code index
awesomeSearch code index --force
awesomeSearch code purge          # wipe index; keeps path rules
awesomeSearch code purge --all    # wipe index + rules

# Path ownership rules
awesomeSearch code rule set "vendor/**" library
awesomeSearch code rule set "vendor/mycompany/**" mine --group backend
awesomeSearch code rule list
awesomeSearch code rule remove "vendor/**"

# Search with filters (agent-friendly)
awesomeSearch code search "create user" --mine
awesomeSearch code search "button component" --group frontend --exclude-group frontend_library
awesomeSearch code search "helper" --language python --path-prefix backend/

# Watch writes and refresh symbol line maps (polling)
awesomeSearch code watch --poll 1.0
awesomeSearch code watch --for 30          # auto-stop after 30 minutes
awesomeSearch code watch stop              # stop permanently
awesomeSearch code watch stop --minutes 10 # pause 10 min (e.g. during npm update), then resume
awesomeSearch code watch resume
awesomeSearch code watch status
```

Code mode does not require an `llms/` folder; it uses the nearest `.git` root or the current directory.

## Environment probe

Agents often guess wrong about **Windows vs WSL vs bash**, `python` vs `python3`, venvs, or missing CLI tools. Probe first:

```bash
awesomeSearch env                 # host/shell/tools + agent rules; exit 2 on warnings
awesomeSearch env detect --json
awesomeSearch env confirm --yes   # snapshot allowlist into .awesome_index/env.json
awesomeSearch env confirm --yes --python-cmd python3 --script-language py
awesomeSearch env set preferred_script_language py

# Grow the tool allowlist after you verify/install something
awesomeSearch env tool probe rg
awesomeSearch env tool add jq --after-install
awesomeSearch env tool list
awesomeSearch env tool remove jq

awesomeSearch env show
awesomeSearch env clear
```

Detection covers:

- **Host context**: `windows` / `wsl` / `linux` / `macos`, shell family (`powershell`, `bash`, …)
- **Invoke rules**: e.g. on Windows+WSL — write a file, then `wsl.exe -d Distro -- python3 /path` (not nested bash in PowerShell)
- **Runtimes**: python/python3/py, node package managers, venv
- **CLI allowlist**: probed common tools (`rg`, `grep`, `curl`, `jq`, `wc`, `sed`, …) with Windows equivalents where relevant

See **`skills.md`** for the default agent playbook (run env first, respect `available_tools`, register tools after install).

## Tips

- **Supported formats**: .md, .txt, .mdx files in the llms folder
- **Code formats**: `.py`, `.ts`/`.tsx`, `.js`/`.jsx`, `.rs`, `.go`, `.java`, and related source extensions
- **Chunk size**: Documents are split into 500-character chunks with 50-character overlap
- **Embeddings**: Uses `all-MiniLM-L6-v2` (fast, 384 dimensions)
- **Vector engine**: FAISS IVF Flat with e-means training (`nlist ≈ 4√N`, 5 epochs); `nprobe` defaults to 10
- **Distance threshold**: Lower scores = more relevant. Typical good matches are < 1.0
- **Add to .gitignore**: `.awesome_index/`

## Troubleshooting

**"No 'llms' folder found"**
- Make sure you're in a project directory that has an `llms` folder
- The tool searches up to 5 parent directories

**"Index not found"**
- Run `awesomeIndexer` first to create the index

**Poor search results**
- Try adjusting the threshold parameter
- Increase num_results to see more options
- Make sure your documents are indexed with `awesomeIndexer`
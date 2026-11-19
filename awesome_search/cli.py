"""
AwesomeSearch - CLI tool for searching LLM instruction documents
Install globally: pip install -e .
"""

import argparse
import faiss
import numpy as np
import pickle
import json
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys

class DocumentSearchEngine:
    # Primary engine: FAISS IVF Flat with e-means clustering (Meta FAIR).
    # Flat L2 is only a fallback when the corpus is too small to train IVF.
    PRIMARY_ENGINE = "ivf_emeans"
    FALLBACK_ENGINE = "flat_l2"
    MIN_VECTORS_FOR_IVF = 39  # Need enough points relative to nlist
    EMEANS_NEPOCHS = 5

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.llms_folder = self.project_root / "llms"
        self.index_folder = self.project_root / ".awesome_index"
        self.index_path = self.index_folder / "index.faiss"
        self.data_path = self.index_folder / "data.pkl"
        self.hash_log_path = self.index_folder / "file_hashes.json"
        self.embedding_dim = 384
        self.default_nprobe = 10
        
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using sentence-transformers."""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings

    def _choose_nlist(self, num_vectors: int) -> int:
        """
        Choose IVF nlist (e-means centroids) for the corpus size.

        Rule of thumb: ~4 * sqrt(N), clamped so each list has enough vectors
        for stable e-means training.
        """
        if num_vectors < self.MIN_VECTORS_FOR_IVF:
            return 1
        nlist = int(4 * np.sqrt(num_vectors))
        # Keep at least ~39 training vectors per centroid when possible
        max_by_density = max(1, num_vectors // 39)
        nlist = max(1, min(nlist, max_by_density, num_vectors))
        return nlist

    def _train_emeans_centroids(self, vectors: np.ndarray, nlist: int) -> np.ndarray:
        """Train IVF coarse quantizer centroids with FAISS e-means."""
        from faiss.contrib.e_means import NumpyEmeans

        dim = vectors.shape[1]
        em = NumpyEmeans(dim, nlist, nepochs=self.EMEANS_NEPOCHS)
        centroids, _labels = em.train(vectors)
        return np.ascontiguousarray(centroids.astype('float32'))

    def _build_faiss_index(self, embeddings: np.ndarray) -> Tuple[faiss.Index, Dict]:
        """
        Build the primary FAISS vector engine: IVF Flat trained with e-means.

        Falls back to IndexFlatL2 only when there are too few vectors to train
        a meaningful inverted-file / e-means partition.
        """
        vectors = np.ascontiguousarray(embeddings.astype('float32'))
        num_vectors, dim = vectors.shape
        if dim != self.embedding_dim:
            raise ValueError(
                f"Embedding dim mismatch: got {dim}, expected {self.embedding_dim}"
            )

        if num_vectors < self.MIN_VECTORS_FOR_IVF:
            index = faiss.IndexFlatL2(dim)
            index.add(vectors)
            return index, {
                'engine': self.FALLBACK_ENGINE,
                'trainer': None,
                'nlist': 0,
                'nprobe': 0,
                'metric': 'l2',
                'num_vectors': num_vectors,
            }

        nlist = self._choose_nlist(num_vectors)
        centroids = self._train_emeans_centroids(vectors, nlist)

        quantizer = faiss.IndexFlatL2(dim)
        quantizer.add(centroids)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
        # Centroids were trained by e-means; skip FAISS Lloyd index.train().
        if not index.is_trained:
            index.is_trained = True
        index.add(vectors)
        nprobe = min(self.default_nprobe, nlist)
        index.nprobe = nprobe

        return index, {
            'engine': self.PRIMARY_ENGINE,
            'trainer': 'e_means',
            'nlist': nlist,
            'nprobe': nprobe,
            'nepochs': self.EMEANS_NEPOCHS,
            'metric': 'l2',
            'num_vectors': num_vectors,
        }

    def _configure_index_for_search(self, index: faiss.Index, engine_meta: Optional[Dict] = None):
        """Apply search-time settings (nprobe) for IVF / e-means indexes."""
        if hasattr(index, 'nprobe'):
            nlist = getattr(index, 'nlist', None)
            preferred = (engine_meta or {}).get('nprobe', self.default_nprobe)
            if nlist:
                index.nprobe = min(max(1, preferred), nlist)
            else:
                index.nprobe = max(1, preferred)
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file content."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _load_hash_log(self) -> Dict[str, str]:
        """Load file hash log."""
        if self.hash_log_path.exists():
            with open(self.hash_log_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_hash_log(self, hash_log: Dict[str, str]):
        """Save file hash log."""
        self.index_folder.mkdir(parents=True, exist_ok=True)
        with open(self.hash_log_path, 'w') as f:
            json.dump(hash_log, f, indent=2)
    
    def _chunk_document(self, text: str, chunk_size: int = 500, 
                       overlap: int = 50) -> List[Tuple[str, int, List[str]]]:
        '''
        Split document into overlapping chunks with character positions,
        trying to break at natural boundaries (markdown headers, code blocks, paragraphs).
        
        Returns:
            List of tuples (chunk_text, start_char_position, languages)
        '''
        chunks = []
        start = 0
        text_len = len(text)

        def extract_languages(chunk_text: str) -> List[str]:
            languages = set()
            for match in re.finditer(r'^[ \t]*```([a-zA-Z0-9_+-]+)(?:[ \t,].*)?$', chunk_text, re.MULTILINE):
                languages.add(match.group(1).lower())
            return list(languages)

        while start < text_len:
            if start + chunk_size >= text_len:
                chunk = text[start:]
                if chunk.strip():
                    chunks.append((chunk, start, extract_languages(chunk)))
                break

            end = start + chunk_size
            search_start = max(start, end - int(chunk_size * 0.5))

            break_pos = -1

            num_markers = text.count('```', 0, end)
            if num_markers % 2 != 0:
                code_end = text.find('```', end)
                if code_end != -1 and code_end - start <= chunk_size * 1.5:
                    end = code_end + 3
                    search_start = end - 1

            if break_pos == -1:
                break_pos = text.rfind('\n## ', search_start, end)
                if break_pos != -1: end = break_pos + 1
            if break_pos == -1:
                break_pos = text.rfind('\n# ', search_start, end)
                if break_pos != -1: end = break_pos + 1
            if break_pos == -1:
                break_pos = text.rfind('\n```', search_start, end)
                if break_pos != -1: end = break_pos + 4
            if break_pos == -1:
                break_pos = text.rfind('\n\n', search_start, end)
                if break_pos != -1: end = break_pos + 2
            if break_pos == -1:
                break_pos = text.rfind('\n', search_start, end)
                if break_pos != -1: end = break_pos + 1
            if break_pos == -1:
                matches = list(re.finditer(r'[.!?]\s', text[search_start:end]))
                if matches: end = search_start + matches[-1].end()
            if break_pos == -1:
                break_pos = text.rfind(' ', search_start, end)
                if break_pos != -1: end = break_pos + 1

            chunk = text[start:end]
            if chunk.strip():
                chunks.append((chunk, start, extract_languages(chunk)))

            target_start = end - overlap

            if target_start <= start:
                start = end
                continue

            search_window_start = max(start + 1, target_start - int(overlap * 0.5))
            search_window_end = min(end, target_start + int(overlap * 0.5))

            start_pos = -1

            if start_pos == -1:
                start_pos = text.find('\n## ', search_window_start, search_window_end)
                if start_pos != -1: start = start_pos + 1
            if start_pos == -1:
                start_pos = text.find('\n# ', search_window_start, search_window_end)
                if start_pos != -1: start = start_pos + 1
            if start_pos == -1:
                start_pos = text.find('\n```', search_window_start, search_window_end)
                if start_pos != -1: start = start_pos + 1
            if start_pos == -1:
                start_pos = text.find('\n\n', search_window_start, search_window_end)
                if start_pos != -1: start = start_pos + 2
            if start_pos == -1:
                start_pos = text.find('\n', search_window_start, search_window_end)
                if start_pos != -1: start = start_pos + 1
            if start_pos == -1:
                match = re.search(r'[.!?]\s', text[search_window_start:search_window_end])
                if match: start = search_window_start + match.end()
            if start_pos == -1:
                start_pos = text.find(' ', search_window_start, search_window_end)
                if start_pos != -1: start = start_pos + 1

            if start_pos == -1:
                start = target_start

        return chunks

    def _ensure_gitignore(self):
        """Ensure .awesome_index is in .gitignore if .git exists."""
        git_dir = self.project_root / ".git"
        if not git_dir.exists() or not git_dir.is_dir():
            return

        gitignore_path = self.project_root / ".gitignore"
        index_folder_name = self.index_folder.name

        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                content = f.read()
                if index_folder_name in content:
                    return

            with open(gitignore_path, 'a') as f:
                if not content.endswith('\n'):
                    f.write('\n')
                f.write(f"\n# AwesomeSearch index\n{index_folder_name}/\n")
        else:
            with open(gitignore_path, 'w') as f:
                f.write(f"# AwesomeSearch index\n{index_folder_name}/\n")

    def _get_files_to_index(self, specific_file: Optional[str] = None) -> List[Path]:
        """Get list of files to index."""
        if not self.llms_folder.exists():
            raise FileNotFoundError(f"llms folder not found at {self.llms_folder}")
        
        if specific_file:
            file_path = self.llms_folder / specific_file
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            return [file_path]
        
        # Get all markdown and text files
        files = []
        for ext in ['*.md', '*.txt', '*.mdx']:
            files.extend(self.llms_folder.glob(ext))
        
        return sorted(files)
    
    def _files_need_update(self, files: List[Path]) -> bool:
        """Check if any files have changed since last index."""
        hash_log = self._load_hash_log()
        
        for file_path in files:
            relative_path = str(file_path.relative_to(self.project_root))
            current_hash = self._compute_file_hash(file_path)
            
            if relative_path not in hash_log or hash_log[relative_path] != current_hash:
                return True
        
        return False
    
    def index_documents(self, specific_file: Optional[str] = None, force: bool = False):
        """
        Build or update the search index.
        
        Args:
            specific_file: Index only this file (relative to llms folder)
            force: Force re-indexing even if files haven't changed
        """
        files = self._get_files_to_index(specific_file)
        
        if not files:
            print("No files found to index")
            return
        
        # Check if update needed
        if not force and self.index_path.exists():
            if not self._files_need_update(files):
                print("All files up to date, no indexing needed")
                return
        
        self._ensure_gitignore()
        print(f"Indexing {len(files)} file(s)...")
        
        # Process documents
        all_chunks = []
        all_metadata = []
        hash_log = {}
        
        for file_path in files:
            relative_path = str(file_path.relative_to(self.project_root))
            print(f"  Processing: {relative_path}")
            
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Compute hash
            file_hash = self._compute_file_hash(file_path)
            hash_log[relative_path] = file_hash
            
            # Chunk document with positions
            chunks_with_pos = self._chunk_document(text)
            
            for i, (chunk, char_pos, languages) in enumerate(chunks_with_pos):
                all_chunks.append(chunk)
                all_metadata.append({
                    'source': file_path.name,
                    'relative_path': relative_path,
                    'chunk_id': i,
                    'total_chunks': len(chunks_with_pos),
                    'char_position': char_pos,
                    'languages': languages
                })
        
        print(f"Generating embeddings for {len(all_chunks)} chunks...")
        embeddings = self._get_embeddings(all_chunks)
        
        # Primary vector engine: FAISS IVF + e-means
        index, engine_meta = self._build_faiss_index(embeddings)
        
        # Save everything
        self.index_folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))
        
        with open(self.data_path, 'wb') as f:
            pickle.dump({
                'chunks': all_chunks,
                'metadata': all_metadata,
                'embedding_dim': self.embedding_dim,
                'engine': engine_meta,
            }, f)
        
        self._save_hash_log(hash_log)
        
        engine_name = engine_meta['engine']
        print(f"✓ Indexed {len(all_chunks)} chunks from {len(files)} file(s)")
        print(f"  Vector engine: {engine_name}", end="")
        if engine_name == self.PRIMARY_ENGINE:
            print(
                f" (e-means nlist={engine_meta['nlist']}, "
                f"nprobe={engine_meta['nprobe']}, nepochs={engine_meta['nepochs']})"
            )
        else:
            print(f" (fallback; need >={self.MIN_VECTORS_FOR_IVF} vectors for IVF e-means)")
        print(f"  Index saved to: {self.index_folder}")
    
    def _get_context_around_chunk(self, file_path: Path, chunk_text: str, 
                                   char_position: int, lines_before: int = 50,
                                   lines_after: int = 50) -> Dict:
        """
        Get context lines around a chunk in the original file.
        
        Args:
            file_path: Path to the source file
            chunk_text: The chunk text to find
            char_position: Character position where chunk starts
            lines_before: Number of lines to include before
            lines_after: Number of lines to include after
            
        Returns:
            Dict with context information
        """
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
        
        # Split into lines but keep track of positions
        lines = full_text.split('\n')
        
        # Find which line the chunk starts on
        current_pos = 0
        chunk_start_line = 0
        chunk_end_line = 0
        
        for i, line in enumerate(lines):
            line_length = len(line) + 1  # +1 for newline
            
            # Check if chunk starts in this line
            if current_pos <= char_position < current_pos + line_length:
                chunk_start_line = i
            
            # Check if chunk ends in this line
            chunk_end = char_position + len(chunk_text)
            if current_pos <= chunk_end < current_pos + line_length:
                chunk_end_line = i
                break
            
            current_pos += line_length
        
        # Calculate context boundaries
        context_start = max(0, chunk_start_line - lines_before)
        context_end = min(len(lines), chunk_end_line + lines_after + 1)
        
        # Extract context lines
        before_lines = lines[context_start:chunk_start_line]
        chunk_lines = lines[chunk_start_line:chunk_end_line + 1]
        after_lines = lines[chunk_end_line + 1:context_end]
        
        return {
            'chunk_start_line': chunk_start_line + 1,  # 1-indexed for display
            'chunk_end_line': chunk_end_line + 1,
            'context_start_line': context_start + 1,
            'context_end_line': context_end,
            'lines_before': before_lines,
            'chunk_lines': chunk_lines,
            'lines_after': after_lines,
            'total_lines': len(lines)
        }
    
    def search(self, query: str, source_filter: Optional[str] = None, 
               k: int = 5, threshold: float = 0.0,
               context_mode: bool = False, context_lines: int = 50,
               exclude_languages: Optional[List[str]] = None) -> List[Dict]:
        """
        Search the index.
        
        Args:
            query: Search query
            source_filter: Filter by source filename (optional)
            k: Number of results
            threshold: Maximum distance threshold (0.0 = no threshold)
            context_mode: If True, fetch surrounding lines from source file
            context_lines: Number of lines before/after to include in context
            
        Returns:
            List of results with text, score, metadata, and optionally context
        """
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"Index not found. Run 'awesomeIndexer' first to create the index."
            )
        
        # Load index
        index = faiss.read_index(str(self.index_path))
        
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
            chunks = data['chunks']
            metadata = data['metadata']
            engine_meta = data.get('engine')
        
        self._configure_index_for_search(index, engine_meta)
        
        # Generate query embedding
        query_embedding = self._get_embeddings([query])
        
        # Search - get more results if we're filtering by source/language
        search_k = k * 10 if (source_filter or exclude_languages) else k
        # IVF search_k cannot exceed ntotal; Flat tolerates larger k but clamp anyway
        search_k = min(search_k, max(1, index.ntotal))
        distances, indices = index.search(query_embedding.astype('float32'), search_k)
        
        # Collect results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= len(chunks):
                continue
            
            # Apply threshold
            if threshold > 0.0 and dist > threshold:
                continue
            
            meta = metadata[idx]
            
            # Apply source filter
            if source_filter and meta['source'] != source_filter:
                continue

            # Apply exclude languages filter
            if exclude_languages and 'languages' in meta:
                if any(lang.lower() in exclude_languages for lang in meta['languages']):
                    continue
            
            result = {
                'text': chunks[idx],
                'score': float(dist),
                'metadata': meta
            }
            
            # Add context if requested
            if context_mode:
                file_path = self.project_root / meta['relative_path']
                context = self._get_context_around_chunk(
                    file_path,
                    chunks[idx],
                    meta['char_position'],
                    lines_before=context_lines,
                    lines_after=context_lines
                )
                result['context'] = context
            
            results.append(result)
            
            if len(results) >= k:
                break
        
        return results


def find_project_root(require_llms: bool = True) -> Optional[Path]:
    """
    Find project root.

    Doc mode prefers an `llms/` folder. Code mode also accepts `.git` or CWD.
    """
    current = Path.cwd()

    for _ in range(6):
        if (current / "llms").exists():
            return current
        if not require_llms and (current / ".git").exists():
            return current
        if current.parent == current:
            break
        current = current.parent

    if not require_llms:
        return Path.cwd()
    return None


def cmd_indexer():
    """Entry point for awesomeIndexer command."""
    parser = argparse.ArgumentParser(
        description='Index LLM instruction documents for searching'
    )
    parser.add_argument(
        'file',
        nargs='?',
        help='Specific file to index (relative to llms folder), or omit to index all'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-indexing even if files haven\'t changed'
    )
    
    args = parser.parse_args()
    
    # Find project root
    project_root = find_project_root(require_llms=True)
    if not project_root:
        print("Error: No 'llms' folder found in current directory or parent directories")
        print("Make sure you're running this from within a project with an 'llms' folder")
        sys.exit(1)
    
    print(f"Project root: {project_root}")
    
    try:
        engine = DocumentSearchEngine(project_root)
        engine.index_documents(specific_file=args.file, force=args.force)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def _print_env_report(report, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(report.to_dict(), indent=2))
        return

    host = report.host or {}
    shell = host.get("shell") or {}
    invoke = host.get("invoke") or {}

    print(f"Project root: {report.project_root}")
    print(f"Detected stacks: {', '.join(report.stacks)}")
    if report.signals:
        print(f"Signals: {', '.join(report.signals)}")
    print()
    print("Host")
    print(f"  context       : {host.get('context')}  (os={host.get('os')}, arch={host.get('arch')})")
    print(f"  shell         : {shell.get('name')}  (family={shell.get('family')})")
    if host.get("is_wsl") or host.get("wsl_distro"):
        print(f"  wsl distro    : {host.get('wsl_distro')}")
    if host.get("wsl_exe"):
        print(f"  wsl.exe       : {host.get('wsl_exe')}")
    print(f"  invoke mode   : {invoke.get('mode')}")
    if invoke.get("bridge"):
        print(f"  bridge        : {invoke['bridge'].get('pattern')}")
        print(f"  anti-pattern  : {invoke['bridge'].get('anti_pattern')}")
    print(f"  script lang   : {report.preferred_script_language}")
    print()
    print("Python")
    print(f"  preferred cmd : {report.python.get('preferred_cmd')}")
    print(f"  python3       : {report.python.get('python3')}")
    print(f"  python        : {report.python.get('python')}")
    print(f"  VIRTUAL_ENV   : {report.python.get('virtual_env_path') or '(inactive)'}")
    print(f"  local venvs   : {report.python.get('local_venv_dirs') or '(none)'}")
    print()
    print("Node")
    print(f"  package mgr   : {report.node.get('package_manager')}")
    print(f"  node          : {report.node.get('node')}")
    print(f"  npm/yarn/pnpm : {[report.node.get(k) for k in ('npm', 'yarn', 'pnpm')]}")
    print()
    print(f"Available tools ({len(report.available_tools)}):")
    print("  " + ", ".join(report.available_tools) if report.available_tools else "  (none)")
    if report.missing_common_tools:
        print(f"Missing common: {', '.join(report.missing_common_tools)}")
    if report.tool_equivalents:
        # Show a short subset of useful maps for missing tools
        shown = 0
        for unix_name in report.missing_common_tools:
            alts = report.tool_equivalents.get(unix_name)
            if alts:
                print(f"  equiv {unix_name} → {', '.join(alts)}")
                shown += 1
                if shown >= 8:
                    break
    print()
    if report.agent_rules:
        print("Agent rules:")
        for r in report.agent_rules:
            print(f"  • {r}")
        print()
    if report.warnings:
        print("Warnings:")
        for w in report.warnings:
            print(f"  ! {w}")
        print()
    if report.recommendations:
        print("Recommendations:")
        for r in report.recommendations:
            print(f"  - {r}")
        print()
    if report.confirmed:
        print("Confirmed overrides (.awesome_index/env.json):")
        for k, v in report.confirmed.items():
            if k in {"project_root", "tools", "agent_rules", "available_tools", "warnings_at_confirm"}:
                continue
            print(f"  {k}: {v}")
        conf_tools = (report.confirmed.get("tools") or {})
        if conf_tools:
            print(f"  tools ({len(conf_tools)}): {', '.join(sorted(conf_tools))}")
    else:
        print("No confirmed env yet. Run: awesomeSearch env confirm --yes")


def _cmd_env(argv: List[str]) -> None:
    """Handle `awesomeSearch env ...`."""
    from awesome_search.env_probe import EnvStore, confirm_from_probe, probe_environment

    parser = argparse.ArgumentParser(
        prog="awesomeSearch env",
        description=(
            "Detect host/shell (Windows/WSL/Linux), runtimes, and confirmed CLI tools for agents"
        ),
    )
    sub = parser.add_subparsers(dest="env_cmd")

    p_detect = sub.add_parser("detect", help="Probe environment (default)")
    p_detect.add_argument("--json", action="store_true", help="Machine-readable output")

    p_confirm = sub.add_parser("confirm", help="Save current detection as confirmed")
    p_confirm.add_argument("--yes", action="store_true", help="Accept detection without prompts")
    p_confirm.add_argument("--python-cmd", help="Override interpreter, e.g. python3 or .venv/bin/python")
    p_confirm.add_argument(
        "--node-package-manager",
        choices=["npm", "yarn", "pnpm", "bun"],
        help="Override node package manager",
    )
    p_confirm.add_argument(
        "--script-language",
        choices=["py", "ps1", "js", "sh"],
        help="Preferred language for small helper scripts",
    )

    p_set = sub.add_parser("set", help="Set one confirmed value")
    p_set.add_argument(
        "key",
        choices=[
            "python_cmd",
            "node_package_manager",
            "virtual_env_path",
            "preferred_script_language",
        ],
        help="Config key",
    )
    p_set.add_argument("value", help="Value to store")

    p_tool = sub.add_parser("tool", help="Manage confirmed CLI tool allowlist")
    tool_sub = p_tool.add_subparsers(dest="tool_cmd", required=True)
    tool_sub.add_parser("list", help="List confirmed/registered tools")
    p_tool_add = tool_sub.add_parser(
        "add",
        help="Register a tool after verifying it exists (or after install)",
    )
    p_tool_add.add_argument("name", help="Tool name, e.g. rg, jq, curl")
    p_tool_add.add_argument("--path", help="Explicit path to binary")
    p_tool_add.add_argument(
        "--after-install",
        action="store_true",
        help="Mark source=installed (still requires the binary to resolve)",
    )
    p_tool_add.add_argument("--notes", help="Optional note for agents")
    p_tool_rm = tool_sub.add_parser("remove", help="Remove a registered tool")
    p_tool_rm.add_argument("name", help="Tool name")
    p_tool_probe = tool_sub.add_parser(
        "probe",
        help="Check if a tool exists; register it if found",
    )
    p_tool_probe.add_argument("name", help="Tool name")

    sub.add_parser("show", help="Show confirmed env + fresh detection summary")
    sub.add_parser("clear", help="Clear confirmed env overrides")

    args = parser.parse_args(argv)
    project_root = find_project_root(require_llms=False)
    cmd = args.env_cmd or "detect"

    try:
        if cmd in {"detect", "show"}:
            report = probe_environment(project_root)
            _print_env_report(report, as_json=getattr(args, "json", False))
            if report.warnings:
                sys.exit(2)
            return

        if cmd == "confirm":
            if not args.yes and sys.stdin.isatty():
                report = probe_environment(project_root)
                _print_env_report(report)
                answer = input("Confirm this environment for agents? [y/N] ").strip().lower()
                if answer not in {"y", "yes"}:
                    print("Aborted")
                    return
            overrides = {
                "python_cmd": args.python_cmd,
                "node_package_manager": args.node_package_manager,
                "preferred_script_language": args.script_language,
            }
            data = confirm_from_probe(project_root, overrides=overrides)
            print("✓ Confirmed environment saved to .awesome_index/env.json")
            printable = {
                k: v
                for k, v in data.items()
                if k not in {"warnings_at_confirm", "agent_rules"}
            }
            if "tools" in printable:
                printable["tools"] = sorted(printable["tools"].keys())
            print(json.dumps(printable, indent=2))
            if data.get("warnings_at_confirm"):
                print("\nStill noted at confirm time:")
                for w in data["warnings_at_confirm"]:
                    print(f"  ! {w}")
                sys.exit(2)
            return

        if cmd == "set":
            store = EnvStore(project_root)
            data = store.set_value(args.key, args.value)
            print(f"✓ Set {args.key}={args.value}")
            print(json.dumps({k: v for k, v in data.items() if k != "tools"}, indent=2))
            return

        if cmd == "tool":
            store = EnvStore(project_root)
            if args.tool_cmd == "list":
                tools = store.list_tools()
                if not tools:
                    print("(no confirmed tools yet — run env confirm, or env tool add <name>)")
                    return
                for name in sorted(tools):
                    t = tools[name]
                    print(
                        f"{name}: path={t.get('path')} source={t.get('source')} "
                        f"version={t.get('version')}"
                    )
                return
            if args.tool_cmd == "add":
                source = "installed" if args.after_install else "confirmed"
                entry = store.add_tool(
                    args.name,
                    path=args.path,
                    source=source,
                    notes=args.notes,
                    require_exists=True,
                )
                print(f"✓ Registered tool `{args.name}` ({source})")
                print(json.dumps(entry, indent=2))
                return
            if args.tool_cmd == "remove":
                ok = store.remove_tool(args.name)
                print("✓ Removed" if ok else "Tool not in allowlist")
                return
            if args.tool_cmd == "probe":
                from awesome_search.env_probe import _tool

                info = _tool(args.name)
                if not info.available:
                    print(f"✗ `{args.name}` not found on PATH")
                    sys.exit(1)
                entry = store.add_tool(
                    args.name,
                    path=info.path,
                    version=info.version,
                    source="confirmed",
                )
                print(f"✓ Found and registered `{args.name}`")
                print(json.dumps(entry, indent=2))
                return

        if cmd == "clear":
            ok = EnvStore(project_root).clear()
            print("✓ Cleared" if ok else "Nothing to clear")
            return
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def _cmd_code(argv: List[str]) -> None:
    """Handle `awesomeSearch code ...` subcommands."""
    from awesome_search.code_engine import CodeSearchEngine
    from awesome_search.code_classify import VALID_GROUPS

    parser = argparse.ArgumentParser(
        prog="awesomeSearch code",
        description="Code symbol index: ownership/group-aware search and watch",
    )
    sub = parser.add_subparsers(dest="code_cmd", required=True)

    p_index = sub.add_parser("index", help="Index project source symbols")
    p_index.add_argument("--force", action="store_true", help="Rebuild even if unchanged")
    p_index.add_argument("--path", default=".", help="Subfolder to index (default: project root)")

    p_purge = sub.add_parser("purge", help="Delete code index artifacts")
    p_purge.add_argument(
        "--all",
        action="store_true",
        help="Also delete path rules (default keeps code_rules.json)",
    )

    p_watch = sub.add_parser("watch", help="Watch / stop / pause / resume code index updates")
    p_watch.add_argument("--poll", type=float, default=1.0, help="Poll interval seconds (start mode)")
    p_watch.add_argument(
        "--for",
        dest="for_minutes",
        type=float,
        help="Auto-stop after N minutes (start mode)",
    )
    watch_sub = p_watch.add_subparsers(dest="watch_cmd")
    p_watch_stop = watch_sub.add_parser(
        "stop",
        help="Stop watcher, or pause for N minutes (useful during npm update)",
    )
    p_watch_stop.add_argument(
        "--minutes",
        type=float,
        help="Pause for N minutes then auto-resume (omit to stop permanently)",
    )
    watch_sub.add_parser("resume", help="Resume a paused watcher")
    watch_sub.add_parser("status", help="Show watcher status")

    p_rule = sub.add_parser("rule", help="Manage path ownership/group rules")
    rule_sub = p_rule.add_subparsers(dest="rule_cmd", required=True)
    p_rule_set = rule_sub.add_parser("set", help="Set a path rule")
    p_rule_set.add_argument("pattern", help='Path pattern, e.g. "vendor/**" or "vendor/mycompany/**"')
    p_rule_set.add_argument(
        "ownership",
        choices=["mine", "library", "unknown"],
        help="Ownership label",
    )
    p_rule_set.add_argument(
        "--group",
        choices=sorted(VALID_GROUPS),
        help="Optional group override",
    )
    p_rule_rm = rule_sub.add_parser("remove", help="Remove a path rule")
    p_rule_rm.add_argument("pattern", help="Exact pattern to remove")
    rule_sub.add_parser("list", help="List path rules")

    p_search = sub.add_parser("search", help="Search the code symbol index")
    p_search.add_argument("query", help="Natural language / keyword query")
    p_search.add_argument("-k", "--num-results", type=int, default=10)
    p_search.add_argument("--mine", action="store_true", help="Only symbols owned by 'mine'")
    p_search.add_argument("--ownership", choices=["mine", "library", "unknown"])
    p_search.add_argument("--group", choices=sorted(VALID_GROUPS))
    p_search.add_argument(
        "--exclude-group",
        action="append",
        default=[],
        help="Exclude group(s), e.g. frontend_library",
    )
    p_search.add_argument("--language", help="Filter by language, e.g. typescript")
    p_search.add_argument("--path-prefix", help="Only paths under this prefix")

    args = parser.parse_args(argv)
    project_root = find_project_root(require_llms=False)
    engine = CodeSearchEngine(project_root)
    print(f"Project root: {project_root}")

    try:
        if args.code_cmd == "index":
            root = (project_root / args.path).resolve()
            stats = engine.index_codebase(root=root, force=args.force)
            print(
                f"✓ Code index: {stats['symbols']} symbols from {stats['files']} files "
                f"(changed={stats['changed']}, reused={stats['reused']})"
            )
            if stats.get("engine"):
                print(f"  Vector engine: {stats['engine']}")
            return

        if args.code_cmd == "purge":
            stats = engine.purge(keep_rules=not args.all)
            print(
                f"✓ Purged code index ({stats['removed']} items removed; "
                f"rules kept={bool(stats['kept_rules'])})"
            )
            return

        if args.code_cmd == "watch":
            watch_cmd = getattr(args, "watch_cmd", None)
            if watch_cmd is None:
                if not engine.faiss_path.exists():
                    print("No code index yet; building initial index...")
                    engine.index_codebase(force=True)
                engine.watch(
                    poll_seconds=args.poll,
                    stop_after_minutes=args.for_minutes,
                )
                return
            if watch_cmd == "stop":
                result = engine.stop_watch(minutes=args.minutes)
                if not result.get("ok"):
                    print("No running code watch found")
                    sys.exit(1)
                if result["action"] == "pause":
                    print(
                        f"✓ Watch paused for {result['minutes']:g} minute(s); "
                        "it will resume automatically (re-baselines hashes after pause)"
                    )
                else:
                    print("✓ Stop signal sent to code watch")
                return
            if watch_cmd == "resume":
                result = engine.resume_watch()
                if not result.get("ok"):
                    print("No running code watch found")
                    sys.exit(1)
                print("✓ Resume signal sent")
                return
            if watch_cmd == "status":
                status = engine._read_watch_status()
                running = engine.is_watch_running()
                if not status:
                    print("Watch: not running")
                    return
                print(f"Watch running: {running}")
                print(json.dumps(status, indent=2))
                return

        if args.code_cmd == "rule":
            if args.rule_cmd == "set":
                rule = engine.rules.set_rule(args.pattern, args.ownership, args.group)
                print(f"✓ Rule set: {rule.pattern} -> ownership={rule.ownership}", end="")
                if rule.group:
                    print(f", group={rule.group}")
                else:
                    print()
                return
            if args.rule_cmd == "remove":
                ok = engine.rules.remove_rule(args.pattern)
                print("✓ Removed" if ok else "No matching rule")
                return
            if args.rule_cmd == "list":
                rules = engine.rules.list_rules()
                if not rules:
                    print("(no explicit rules; defaults + .gitignore hints apply)")
                    return
                for rule in rules:
                    extra = f", group={rule.group}" if rule.group else ""
                    print(f"{rule.pattern} -> {rule.ownership}{extra}")
                return

        if args.code_cmd == "search":
            results = engine.search(
                query=args.query,
                k=args.num_results,
                ownership=args.ownership,
                group=args.group,
                exclude_groups=args.exclude_group,
                language=args.language,
                path_prefix=args.path_prefix,
                mine_only=args.mine,
            )
            if not results:
                print("No results found")
                return
            print(f"\nFound {len(results)} code symbol(s):\n")
            print("=" * 80)
            for i, r in enumerate(results, 1):
                print(f"\n[Result {i}] Score: {r['score']:.4f}")
                print(
                    f"{r['kind']} {r['name']}  ({r['language']})  "
                    f"ownership={r['ownership']}  group={r['group']}"
                )
                print(f"Path: {r['path']}:{r['start_line']}-{r['end_line']}")
                if r.get("signature"):
                    print(f"Sig:  {r['signature']}")
                print("-" * 80)
                print(r.get("context") or "")
                print("=" * 80)
            return
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_search():
    """Entry point for awesomeSearch command."""
    # Code-index mode: awesomeSearch code ...
    if len(sys.argv) > 1 and sys.argv[1] == "code":
        _cmd_code(sys.argv[2:])
        return

    # Environment probe: awesomeSearch env ...
    if len(sys.argv) > 1 and sys.argv[1] == "env":
        _cmd_env(sys.argv[2:])
        return

    parser = argparse.ArgumentParser(
        description='Search LLM docs, or use: awesomeSearch code ... / awesomeSearch env ...'
    )
    parser.add_argument(
        'query',
        help='Search query'
    )
    parser.add_argument(
        'source',
        nargs='?',
        help='Filter by source filename (optional)'
    )
    parser.add_argument(
        'num_results',
        nargs='?',
        type=int,
        default=5,
        help='Number of results (default: 5)'
    )
    parser.add_argument(
        'threshold',
        nargs='?',
        type=float,
        default=0.0,
        help='Relevancy threshold - max distance (default: 0.0 = no threshold)'
    )
    parser.add_argument(
        '--context',
        action='store_true',
        help='Enable context mode: fetch surrounding lines from source file'
    )
    parser.add_argument(
        '--context-lines',
        type=int,
        default=50,
        help='Number of lines before/after chunk in context mode (default: 50)'
    )
    parser.add_argument(
        '--exclude-lang',
        action='append',
        help='Exclude chunks containing these programming languages'
    )
    
    args = parser.parse_args()
    
    # Find project root
    project_root = find_project_root(require_llms=True)
    if not project_root:
        print("Error: No 'llms' folder found in current directory or parent directories")
        sys.exit(1)
    
    try:
        engine = DocumentSearchEngine(project_root)
        exclude_langs = [lang.lower() for lang in args.exclude_lang] if args.exclude_lang else None
        results = engine.search(
            query=args.query,
            source_filter=args.source,
            k=args.num_results,
            threshold=args.threshold,
            context_mode=args.context,
            context_lines=args.context_lines,
            exclude_languages=exclude_langs
        )
        
        if not results:
            print("No results found")
            sys.exit(0)
        
        print(f"\nFound {len(results)} result(s):\n")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"\n[Result {i}] Score: {result['score']:.4f}")
            print(f"Source: {result['metadata']['relative_path']}")
            print(f"Chunk: {result['metadata']['chunk_id'] + 1}/{result['metadata']['total_chunks']}")
            
            if 'context' in result:
                ctx = result['context']
                print(f"Lines: {ctx['chunk_start_line']}-{ctx['chunk_end_line']} (of {ctx['total_lines']})")
                print(f"Context: Lines {ctx['context_start_line']}-{ctx['context_end_line']}")
                print("-" * 80)
                
                # Print lines before with line numbers
                if ctx['lines_before']:
                    print("\n--- CONTEXT BEFORE ---")
                    start_line = ctx['context_start_line']
                    for j, line in enumerate(ctx['lines_before']):
                        print(f"{start_line + j:4d} | {line}")
                
                # Print chunk lines highlighted
                print("\n--- MATCHED CHUNK ---")
                for j, line in enumerate(ctx['chunk_lines']):
                    line_num = ctx['chunk_start_line'] + j
                    print(f"{line_num:4d} | >>> {line}")
                
                # Print lines after
                if ctx['lines_after']:
                    print("\n--- CONTEXT AFTER ---")
                    start_line = ctx['chunk_end_line'] + 1
                    for j, line in enumerate(ctx['lines_after']):
                        print(f"{start_line + j:4d} | {line}")
            else:
                print("-" * 80)
                print(result['text'])
            
            print("=" * 80)
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # For testing
    cmd_indexer()
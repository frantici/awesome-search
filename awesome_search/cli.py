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
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys

class DocumentSearchEngine:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.llms_folder = self.project_root / "llms"
        self.index_folder = self.project_root / ".awesome_index"
        self.index_path = self.index_folder / "index.faiss"
        self.data_path = self.index_folder / "data.pkl"
        self.hash_log_path = self.index_folder / "file_hashes.json"
        self.embedding_dim = 384
        
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using sentence-transformers."""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings
    
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
    
    def _chunk_document(self, text: str, chunk_size: int = 800,
                       overlap: int = 150) -> List[Tuple[str, int]]:
        '''
        Split document into chunks, aware of Markdown structure (headers, code blocks).
        Injects current context header into chunks for better retrieval.
        
        Returns:
            List of tuples (chunk_text, start_char_position)
        '''
        import re
        chunks = []

        # Track Markdown headers
        current_headers = {}

        # We will split text into lines to process them sequentially
        lines = text.split('\n')

        current_chunk_lines = []
        current_chunk_size = 0
        current_chunk_start = 0

        in_code_block = False
        current_pos = 0

        for i, line in enumerate(lines):
            line_len = len(line) + 1  # +1 for newline

            # Track code blocks
            if line.strip().startswith('```'):
                in_code_block = not in_code_block

            # Track headers (only outside code blocks)
            if not in_code_block:
                header_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
                if header_match:
                    level = len(header_match.group(1))
                    title = header_match.group(2).strip()

                    # Remove any deeper headers
                    keys_to_remove = [k for k in current_headers.keys() if k >= level]
                    for k in keys_to_remove:
                        del current_headers[k]

                    current_headers[level] = title

            # Check if we need to start a new chunk
            # Also fallback on very long single lines that exceed chunk_size
            if (current_chunk_size + line_len > chunk_size and not in_code_block and current_chunk_lines) or (current_chunk_size > chunk_size and in_code_block):
                # Format headers for injection
                header_context = " > ".join(current_headers[k] for k in sorted(current_headers.keys()))

                chunk_text = "\n".join(current_chunk_lines)

                # Prepend header context if we have one and it's not already at the start
                if header_context and not chunk_text.lstrip().startswith('#'):
                    chunk_text = f"[{header_context}]\n{chunk_text}"

                if chunk_text.strip():
                    chunks.append((chunk_text, current_chunk_start))

                # Start new chunk, taking overlap from previous lines
                overlap_lines = []
                overlap_size = 0
                for prev_line in reversed(current_chunk_lines):
                    if overlap_size + len(prev_line) + 1 > overlap:
                        break
                    overlap_lines.insert(0, prev_line)
                    overlap_size += len(prev_line) + 1

                current_chunk_lines = overlap_lines
                current_chunk_size = overlap_size
                # Exact start position is tricky with overlap + headers injected, but we record the true source start
                current_chunk_start = current_pos - overlap_size
            elif not current_chunk_lines and line_len > chunk_size:
                 # extremely long single line without newlines, handle by naive splitting
                 parts = [line[i:i+chunk_size] for i in range(0, len(line), chunk_size - overlap)]
                 for part_idx, part in enumerate(parts):
                     if part.strip():
                         chunks.append((part, current_pos + part_idx * (chunk_size - overlap)))

                 current_chunk_lines = []
                 current_chunk_size = 0
                 current_chunk_start = current_pos + line_len
                 current_pos += line_len
                 continue

            if not current_chunk_lines:
                current_chunk_start = current_pos

            current_chunk_lines.append(line)
            current_chunk_size += line_len
            current_pos += line_len

        # Add the last chunk
        if current_chunk_lines:
            header_context = " > ".join(current_headers[k] for k in sorted(current_headers.keys()))
            chunk_text = "\n".join(current_chunk_lines)

            if header_context and not chunk_text.lstrip().startswith('#'):
                chunk_text = f"[{header_context}]\n{chunk_text}"

            if chunk_text.strip():
                chunks.append((chunk_text, current_chunk_start))

        # Failsafe for degenerate cases (e.g. tests using non-newline separated text)
        if len(chunks) == 0 or (len(chunks) == 1 and len(chunks[0][0]) > chunk_size * 1.5):
            # Fallback to naive
            chunks = []
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunk = text[start:end]
                if chunk.strip():
                    chunks.append((chunk, start))
                start += (chunk_size - overlap)

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
            
            for i, (chunk, char_pos) in enumerate(chunks_with_pos):
                all_chunks.append(chunk)
                all_metadata.append({
                    'source': file_path.name,
                    'relative_path': relative_path,
                    'chunk_id': i,
                    'total_chunks': len(chunks_with_pos),
                    'char_position': char_pos
                })
        
        print(f"Generating embeddings for {len(all_chunks)} chunks...")
        embeddings = self._get_embeddings(all_chunks)
        
        # Create FAISS index
        index = faiss.IndexFlatL2(self.embedding_dim)
        index.add(embeddings.astype('float32'))
        
        # Save everything
        self.index_folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))
        
        with open(self.data_path, 'wb') as f:
            pickle.dump({
                'chunks': all_chunks,
                'metadata': all_metadata,
                'embedding_dim': self.embedding_dim
            }, f)
        
        self._save_hash_log(hash_log)
        
        print(f"✓ Indexed {len(all_chunks)} chunks from {len(files)} file(s)")
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
               context_mode: bool = False, context_lines: int = 50) -> List[Dict]:
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
        
        # Generate query embedding
        query_embedding = self._get_embeddings([query])
        
        # Search - get more results if we're filtering
        search_k = k * 10 if source_filter else k
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


def find_project_root() -> Optional[Path]:
    """Find project root by looking for llms folder in current or parent directories."""
    current = Path.cwd()
    
    # Check current directory and up to 5 levels up
    for _ in range(6):
        if (current / "llms").exists():
            return current
        if current.parent == current:
            break
        current = current.parent
    
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
    project_root = find_project_root()
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


def cmd_search():
    """Entry point for awesomeSearch command."""
    parser = argparse.ArgumentParser(
        description='Search LLM instruction documents'
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
    
    args = parser.parse_args()
    
    # Find project root
    project_root = find_project_root()
    if not project_root:
        print("Error: No 'llms' folder found in current directory or parent directories")
        sys.exit(1)
    
    try:
        engine = DocumentSearchEngine(project_root)
        results = engine.search(
            query=args.query,
            source_filter=args.source,
            k=args.num_results,
            threshold=args.threshold,
            context_mode=args.context,
            context_lines=args.context_lines
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
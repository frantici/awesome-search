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
from typing import List, Dict, Optional
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
    
    def _chunk_document(self, text: str, chunk_size: int = 500, 
                       overlap: int = 50) -> List[str]:
        """Split document into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start += (chunk_size - overlap)
        return chunks
    
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
            
            # Chunk document
            chunks = self._chunk_document(text)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    'source': file_path.name,
                    'relative_path': relative_path,
                    'chunk_id': i,
                    'total_chunks': len(chunks)
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
    
    def search(self, query: str, source_filter: Optional[str] = None, 
               k: int = 5, threshold: float = 0.0) -> List[Dict]:
        """
        Search the index.
        
        Args:
            query: Search query
            source_filter: Filter by source filename (optional)
            k: Number of results
            threshold: Maximum distance threshold (0.0 = no threshold)
            
        Returns:
            List of results with text, score, and metadata
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
            
            results.append({
                'text': chunks[idx],
                'score': float(dist),
                'metadata': meta
            })
            
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
            threshold=args.threshold
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
            print("-" * 80)
            print(result['text'])
            print("=" * 80)
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # For testing
    cmd_indexer()
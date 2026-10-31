"""
Tests for awesome-search library
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import json
import numpy as np
from awesome_search.cli import DocumentSearchEngine, find_project_root


@pytest.fixture
def temp_project():
    """Create a temporary project structure with llms folder and sample documents."""
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create llms folder
    llms_folder = temp_dir / "llms"
    llms_folder.mkdir()
    
    # Create sample documents
    sample_docs = {
        "python_guide.md": """# Python Best Practices
        
Python is a versatile programming language. Here are some best practices:

1. Use virtual environments
2. Follow PEP 8 style guide
3. Write docstrings for functions
4. Use type hints for better code clarity
5. Write unit tests for your code

## Testing with pytest
Pytest is the most popular testing framework for Python. It's simple and powerful.
""",
        "javascript_guide.md": """# JavaScript Fundamentals

JavaScript is the language of the web. Key concepts:

- Variables: let, const, var
- Functions: arrow functions, async/await
- DOM manipulation
- ES6+ features
- Testing with Jest

## Modern JavaScript
Use modern ES6+ syntax for cleaner code.
""",
        "testing.txt": """Unit Testing Guide

Testing is crucial for software quality.

Types of tests:
- Unit tests: Test individual functions
- Integration tests: Test component interactions
- End-to-end tests: Test full workflows

Best practices:
- Test edge cases
- Use descriptive test names
- Keep tests independent
""",
    }
    
    for filename, content in sample_docs.items():
        (llms_folder / filename).write_text(content)
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def search_engine(temp_project):
    """Create a DocumentSearchEngine instance."""
    return DocumentSearchEngine(temp_project)


class TestDocumentSearchEngine:
    """Test suite for DocumentSearchEngine class."""
    
    def test_initialization(self, temp_project):
        """Test that DocumentSearchEngine initializes correctly."""
        engine = DocumentSearchEngine(temp_project)
        
        assert engine.project_root == temp_project
        assert engine.llms_folder == temp_project / "llms"
        assert engine.index_folder == temp_project / ".awesome_index"
        assert engine.embedding_dim == 384
    
    def test_get_files_to_index(self, search_engine, temp_project):
        """Test that all markdown and text files are discovered."""
        files = search_engine._get_files_to_index()
        
        assert len(files) == 3
        filenames = {f.name for f in files}
        assert "python_guide.md" in filenames
        assert "javascript_guide.md" in filenames
        assert "testing.txt" in filenames
    
    def test_get_specific_file_to_index(self, search_engine, temp_project):
        """Test indexing a specific file."""
        files = search_engine._get_files_to_index(specific_file="python_guide.md")
        
        assert len(files) == 1
        assert files[0].name == "python_guide.md"
    
    def test_get_files_to_index_nonexistent(self, search_engine):
        """Test that nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            search_engine._get_files_to_index(specific_file="nonexistent.md")
    
    def test_compute_file_hash(self, search_engine, temp_project):
        """Test file hash computation."""
        test_file = temp_project / "llms" / "python_guide.md"
        hash1 = search_engine._compute_file_hash(test_file)
        
        # Hash should be consistent
        hash2 = search_engine._compute_file_hash(test_file)
        assert hash1 == hash2
        
        # Hash should be hex string
        assert len(hash1) == 64
        assert all(c in "0123456789abcdef" for c in hash1)
    
    def test_chunk_document(self, search_engine):
        """Test document chunking with positions."""
        text = "A" * 1000 + "B" * 500
        chunks = search_engine._chunk_document(text, chunk_size=500, overlap=50)
        
        # Should have multiple chunks
        assert len(chunks) > 1
        
        # Each chunk should be a tuple (text, position)
        for chunk_text, position, langs in chunks:
            assert isinstance(chunk_text, str)
            assert isinstance(position, int)
            assert position >= 0
        
        # First chunk should start at position 0
        assert chunks[0][1] == 0
        
        # Chunks should overlap
        if len(chunks) > 1:
            assert chunks[1][1] < chunks[0][1] + len(chunks[0][0])
    
    def test_chunk_document_small_text(self, search_engine):
        """Test chunking with text smaller than chunk size."""
        text = "Short text"
        chunks = search_engine._chunk_document(text, chunk_size=500)
        
        assert len(chunks) == 1
        assert chunks[0][0] == text
        assert chunks[0][1] == 0
    
    def test_hash_log_operations(self, search_engine, temp_project):
        """Test saving and loading hash log."""
        hash_log = {
            "llms/file1.md": "hash1",
            "llms/file2.md": "hash2"
        }
        
        search_engine._save_hash_log(hash_log)
        assert search_engine.hash_log_path.exists()
        
        loaded_log = search_engine._load_hash_log()
        assert loaded_log == hash_log
    
    def test_index_documents(self, search_engine, temp_project):
        """Test full indexing process."""
        search_engine.index_documents()
        
        # Check that index files were created
        assert search_engine.index_path.exists()
        assert search_engine.data_path.exists()
        assert search_engine.hash_log_path.exists()
        
        # Check hash log contains all files
        hash_log = search_engine._load_hash_log()
        assert len(hash_log) == 3
    
    def test_index_specific_file(self, search_engine, temp_project):
        """Test indexing a specific file."""
        search_engine.index_documents(specific_file="python_guide.md")
        
        assert search_engine.index_path.exists()
        assert search_engine.data_path.exists()
    
    def test_files_need_update(self, search_engine, temp_project):
        """Test change detection for files."""
        files = search_engine._get_files_to_index()
        
        # Initially should need update (no hash log)
        assert search_engine._files_need_update(files) is True
        
        # After indexing, should not need update
        search_engine.index_documents()
        assert search_engine._files_need_update(files) is False
        
        # Modify a file
        test_file = temp_project / "llms" / "python_guide.md"
        test_file.write_text("Modified content")
        
        # Should now need update
        assert search_engine._files_need_update(files) is True
    
    def test_search_basic(self, search_engine):
        """Test basic search functionality."""
        # Index documents first
        search_engine.index_documents()
        
        # Search for Python-related content
        results = search_engine.search("Python testing pytest", k=3)
        
        assert len(results) > 0
        assert all("score" in r for r in results)
        assert all("text" in r for r in results)
        assert all("metadata" in r for r in results)
        
        # Results should be sorted by score (distance)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores)
    
    def test_search_with_source_filter(self, search_engine):
        """Test search with source file filter."""
        search_engine.index_documents()
        
        results = search_engine.search(
            "programming language",
            source_filter="python_guide.md",
            k=5
        )
        
        # All results should be from the filtered source
        for result in results:
            assert result["metadata"]["source"] == "python_guide.md"
    
    def test_search_with_threshold(self, search_engine):
        """Test search with distance threshold."""
        search_engine.index_documents()
        
        # Search with very low threshold (only very similar results)
        results = search_engine.search("Python", k=10, threshold=0.5)
        
        # All results should have score below threshold
        assert all(r["score"] <= 0.5 for r in results)
    
    def test_search_without_index(self, search_engine):
        """Test that search raises error when index doesn't exist."""
        with pytest.raises(FileNotFoundError):
            search_engine.search("test query")
    
    def test_search_metadata(self, search_engine):
        """Test that search results contain proper metadata."""
        search_engine.index_documents()
        results = search_engine.search("test", k=1)
        
        if results:
            metadata = results[0]["metadata"]
            assert "source" in metadata
            assert "relative_path" in metadata
            assert "chunk_id" in metadata
            assert "total_chunks" in metadata
            assert "char_position" in metadata
            
            assert isinstance(metadata["chunk_id"], int)
            assert isinstance(metadata["total_chunks"], int)
            assert metadata["chunk_id"] >= 0
            assert metadata["chunk_id"] < metadata["total_chunks"]
    
    def test_get_context_around_chunk(self, search_engine, temp_project):
        """Test context extraction around chunks."""
        test_file = temp_project / "llms" / "python_guide.md"
        content = test_file.read_text()
        
        # Get a chunk from the middle
        chunk_text = content[50:150]
        
        context = search_engine._get_context_around_chunk(
            test_file,
            chunk_text,
            char_position=50,
            lines_before=2,
            lines_after=2
        )
        
        assert "chunk_start_line" in context
        assert "chunk_end_line" in context
        assert "lines_before" in context
        assert "chunk_lines" in context
        assert "lines_after" in context
        assert "total_lines" in context
        
        assert isinstance(context["lines_before"], list)
        assert isinstance(context["chunk_lines"], list)
        assert isinstance(context["lines_after"], list)
    
    def test_search_with_context_mode(self, search_engine):
        """Test search with context mode enabled."""
        search_engine.index_documents()
        
        results = search_engine.search(
            "Python",
            k=2,
            context_mode=True,
            context_lines=3
        )
        
        assert len(results) > 0
        
        for result in results:
            assert "context" in result
            ctx = result["context"]
            assert "chunk_start_line" in ctx
            assert "lines_before" in ctx
            assert "chunk_lines" in ctx
            assert "lines_after" in ctx


class TestProjectRootDetection:
    """Test suite for find_project_root function."""
    
    def test_find_project_root_current_dir(self, temp_project):
        """Test finding project root in current directory."""
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_project)
            root = find_project_root()
            # Resolve both paths to handle symlinks (macOS /var -> /private/var)
            assert root.resolve() == temp_project.resolve()
        finally:
            os.chdir(original_cwd)
    
    def test_find_project_root_parent_dir(self, temp_project):
        """Test finding project root in parent directory."""
        import os
        original_cwd = os.getcwd()
        
        # Create a subdirectory
        sub_dir = temp_project / "subdir"
        sub_dir.mkdir()
        
        try:
            os.chdir(sub_dir)
            root = find_project_root()
            # Resolve both paths to handle symlinks (macOS /var -> /private/var)
            assert root.resolve() == temp_project.resolve()
        finally:
            os.chdir(original_cwd)
    
    def test_find_project_root_not_found(self):
        """Test that None is returned when no project root found."""
        import os
        original_cwd = os.getcwd()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                os.chdir(temp_dir)
                root = find_project_root()
                assert root is None
            finally:
                os.chdir(original_cwd)


class TestEmbeddings:
    """Test suite for embedding generation."""
    
    def test_get_embeddings_single(self, search_engine):
        """Test generating embeddings for single text."""
        texts = ["This is a test sentence."]
        embeddings = search_engine._get_embeddings(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (1, 384)
    
    def test_get_embeddings_multiple(self, search_engine):
        """Test generating embeddings for multiple texts."""
        texts = [
            "First sentence.",
            "Second sentence.",
            "Third sentence."
        ]
        embeddings = search_engine._get_embeddings(texts)
        
        assert embeddings.shape == (3, 384)
    
    def test_embeddings_similarity(self, search_engine):
        """Test that similar texts have similar embeddings."""
        texts = [
            "Python is a programming language",
            "Python is used for programming",
            "The sky is blue"
        ]
        embeddings = search_engine._get_embeddings(texts)
        
        # Calculate cosine similarity between embeddings
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        # Python-related texts should be more similar to each other
        sim_python = cosine_similarity(embeddings[0], embeddings[1])
        sim_different = cosine_similarity(embeddings[0], embeddings[2])
        
        assert sim_python > sim_different


class TestVersioning:
    """Test version information."""
    
    def test_version_exists(self):
        """Test that version is defined."""
        from awesome_search import __version__
        assert __version__ is not None
        assert isinstance(__version__, str)
        assert len(__version__) > 0
    
    def test_version_format(self):
        """Test that version follows semantic versioning."""
        from awesome_search import __version__
        parts = __version__.split(".")
        assert len(parts) >= 2  # At least major.minor
        assert all(part.isdigit() for part in parts)

    def test_extract_languages_from_rust_book(self, search_engine):
        """Test language extraction specifically from rust book files containing code blocks like ```rust,ignore"""
        # Create a mock file with the content similar to the Rust book
        chunk_text = """
        Some markdown text before the block.

        ```rust,ignore
        let apples = 5; // immutable
        let mut bananas = 5; // mutable
        ```

        Some text after the block.
        """

        chunks = search_engine._chunk_document(chunk_text)

        rust_found = False
        for chunk, pos, languages in chunks:
            if 'rust' in languages:
                rust_found = True
                break

        assert rust_found, "Failed to extract 'rust' from rust,ignore code block"

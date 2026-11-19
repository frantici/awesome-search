"""
Setup file for AwesomeSearch CLI tools
"""

from setuptools import setup, find_packages

setup(
    name="awesome-search",
    version="0.9.0",
    description="Local FAISS IVF e-means search for LLM docs and code symbols",
    author="Trym K. Bjerkan",
    packages=find_packages(),
    install_requires=[
        "faiss-cpu>=1.16.0",
        "sentence-transformers>=2.2.0",
        "numpy>=1.21.0",
        "textual>=0.52.1",
    ],
    extras_require={
        "prep": [
            "openai>=1.0.0",
            "PyMuPDF>=1.23.0",
            "python-docx>=1.0.0",
            "beautifulsoup4>=4.12.0",
            "requests>=2.31.0",
            "markdownify>=0.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "awesomeIndexer=awesome_search.cli:cmd_indexer",
            "awesomeSearch=awesome_search.cli:cmd_search",
            "awesomeTUI=awesome_search.tui:main",
            "awesomePrep=awesome_search.prep:cmd_prep",
        ],
    },
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)

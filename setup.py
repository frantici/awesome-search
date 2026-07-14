"""
Setup file for AwesomeSearch CLI tools
"""

from setuptools import setup, find_packages

setup(
    name="awesome-search",
    version="0.6.0",
    description="Local FAISS-based search tool for LLM instruction documents",
    author="Trym K. Bjerkan",
    packages=find_packages(),
    install_requires=[
        "faiss-cpu>=1.7.4",
        "sentence-transformers>=2.7.0",
        "numpy>=1.26.0",
        "textual>=0.52.1"
    ],
    entry_points={
        "console_scripts": [
            "awesomeIndexer=awesome_search.cli:cmd_indexer",
            "awesomeSearch=awesome_search.cli:cmd_search",
            "awesomeTUI=awesome_search.tui:main"
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

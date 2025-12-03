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
        "faiss-cpu>=1.7.0",
        "sentence-transformers>=2.2.0",
        "numpy>=1.21.0",
    ],
    entry_points={
        "console_scripts": [
            "awesomeIndexer=awesome_search.cli:cmd_indexer",
            "awesomeSearch=awesome_search.cli:cmd_search",
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
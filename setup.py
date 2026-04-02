"""Functionality: Declare this repo as an installable Python package (setuptools)

Run from repo root: pip install -e .
That registers packages (agent, llm_clients, prompt_builder, etc) and installs
the listed third-party dependencies so imports work from any working directory
"""

from setuptools import setup, find_packages

setup(
    name="adaptive-memento",
    version="0.1.0",
    # Subpackages with __init__.py under this directory
    packages=find_packages(),
    python_requires=">=3.9",
    # Runtime deps for gym / BabyAI and project scripts; add more if you use them in code
    install_requires=[
        "gymnasium",
        "minigrid",
        "pyyaml",
        "tqdm",
        "sentence-transformers",
    ],
)

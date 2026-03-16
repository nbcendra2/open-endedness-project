from setuptools import setup, find_packages

setup(
    name="adaptive-memento",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "gymnasium",
        "minigrid",
        "pyyaml",
        "tqdm",
        "sentence-transformers",
    ],
)
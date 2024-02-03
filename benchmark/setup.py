import os
from pathlib import Path
from setuptools import setup

with open("README.md", "r") as readme:
    long_description = readme.read()

cwd = Path(__file__).resolve().parent

setup(
    name="divr-benchmark",
    packages=["divr_benchmark"],
    version=os.environ["RELEASE_VERSION"],
    license="MIT",
    description="Toolkit to extract features from disordered voice databases",
    author="Computational Audio Research Lab",
    url="https://github.com/ComputationalAudioResearchLab/divr-benchmark",
    keywords=[
        "ML Audio Features",
        "ML",
        "Disordered Voice Features",
        "Research",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "aiofiles>=23.1.0",
        "aiohttp>=3.8.4",
        "librosa>=0.10.0.post2",
        "matplotlib>=3.7.1",
        "nspfile>=0.1.4",
        "openpyxl>=3.1.2",
        "pandas>=2.0.1",
        "PyYAML>=6.0.1",
        "class-argparse>=0.1.3",
        "svd-downloader>=0.1.1",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Researchers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Utilities",
    ],
)

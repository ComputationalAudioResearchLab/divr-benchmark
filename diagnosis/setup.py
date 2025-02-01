import os
from setuptools import setup

with open("README.md", "r") as readme:
    long_description = readme.read()

setup(
    name="divr-diagnosis",
    packages=["divr_diagnosis"],
    version=os.environ["RELEASE_VERSION"],
    license="MIT",
    description="Toolkit to standardize voice disorder diagnostic labels",
    author="Computational Audio Research Lab",
    url="https://github.com/ComputationalAudioResearchLab/divr-benchmark/diagnosis",
    keywords=[
        "ML Audio Features",
        "ML",
        "Disordered Voice Features",
        "Research",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["PyYAML>=6.0.1"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Researchers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Utilities",
    ],
)

#!/bin/bash

set -e

cd /home/workspace/benchmark;
export PIPENV_CUSTOM_VENV_NAME=divr-benchmark;
pipenv install --dev;

cd /home/workspace/baselines;
export PIPENV_CUSTOM_VENV_NAME=divr-baselines;
pipenv install --site-packages --dev;

cd /home/workspace/icassp_2025;
export PIPENV_CUSTOM_VENV_NAME=icassp-2025;
pipenv install --site-packages --dev;
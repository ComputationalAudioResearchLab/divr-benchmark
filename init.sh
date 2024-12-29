#!/bin/bash

set -e

cd /home/workspace/benchmark;
cp -n .env.default .env;
pipenv install --dev;

cd /home/workspace/baselines;
cp -n .env.default .env;
pipenv install --site-packages --dev;

cd /home/workspace/acm_transactions_2025;
cp -n .env.default .env;
pipenv install --site-packages --dev;

cd /home/workspace/icassp_2025;
cp -n .env.default .env;
pipenv install --site-packages --dev;
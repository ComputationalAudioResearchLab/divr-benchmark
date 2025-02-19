#!/bin/bash

set -e

cd /home/workspace/diagnosis;
cp -n .env.default .env;
pipenv install --dev;

cd /home/workspace/benchmark;
cp -n .env.default .env;
pipenv install --dev;

cd /home/workspace/acm_transactions_2025;
cp -n .env.default .env;
pipenv install --site-packages --dev;

cd /home/workspace/thesis_work;
cp -n .env.default .env;
pipenv install --site-packages --dev;

#!/bin/bash

set -e

cp -n .env.default .env;
pipenv install --dev;
# DiVR (Disordered Voice Recognition) - Benchmark

This repository contains the work that establishes a benchmark to test voice disorder recognition systems.

## Getting Started with Development workflow

### Requirements

1. Docker
2. VSCode
   1. VSCode Devcontainer extension
3. Databases
   - Open Access
     - SVD
     - TORGO
     - Voiced
   - Closed Access
     - AVFD
     - MEEI
     - UncommonVoice

### Setup

1. Copy `.devcontainer/.env.default` file to `.devcontainer/.env`
2. Replace the environment configuration variable `RESEARCH_DATA_PATH` with the path on your host machine where you wish to store data
3. Open this repository in VSCode
4. Open the command pallette (View > Command Pallette) or (Ctrl + Shift + P)
   1. Select `Dev Containers: Open Workspace in Container...`
   2. Select `divr-benchmark.code-workspace` within this folder

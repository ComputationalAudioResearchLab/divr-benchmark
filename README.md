# DiVR (Disordered Voice Recognition) - Benchmark

This repository contains the work that establishes a benchmark to test voice disorder recognition systems as well as some of our baseline implementations.

## Links

- [Benchmark](./benchmark/README.md)
- [Baselines](./baselines/README.md)

## Working on this repository directly

If you need to work on this repository directly then follow the following instructions:

### Requirements

1. Docker
2. VSCode
   1. VSCode Devcontainer extension
3. Databases
   - Open Access (these can be auto-downloaded by the benchmark as well)
     - SVD
     - TORGO
     - Voiced
   - Restricted Access
     - AVFD
     - MEEI
     - UASpeech
     - UncommonVoice

### Setup

1. Copy `.devcontainer/.env.default` file to `.devcontainer/.env`
2. Replace the environment configuration variable `RESEARCH_DATA_PATH` with the path on your host machine where you store research data related to this project
3. Open this repository in VSCode
4. Open the command pallette (View > Command Pallette) or (Ctrl + Shift + P)
   1. Select `Dev Containers: Open Workspace in Container...`
   2. Select `divr-benchmark.code-workspace` within this folder

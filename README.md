# DiVR (Disordered Voice Recognition) - Benchmark

This repository contains the work that establishes a benchmark to test voice disorder recognition systems.

## Getting Started

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

1. Add all the databases in a sibling directory, such that the structure looks like:
   ```
       - vdml-benchmarks
           - ...
           - README.md
           - workspace.code-workspace
       - databases
           - SVD
           - Voice
           - ...
   ```
1. Open this repository in VSCode
1. Open the command pallette (View > Command Pallette) or (Ctrl + Shift + P)
   1. Select `Dev Containers: Open Workspace in Container...`
   2. Select `workspace.code-workspace` within this folder

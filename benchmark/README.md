# DiVR (Disordered Voice Recognition) - Benchmark

This repository contains the work that establishes a benchmark to test voice disorder recognition systems.

## Installation

```sh
pip install divr-benchmark
```

## How to use

```python
from divr_benchmark import Benchmark, Diagnosis

# Get a specific task from the benchmark
# Storage path is needed to store the public datasets and generated files
benchmark = Benchmark(
    storage_path="/home/user/divr_benchmark/storage",
    version="v1",
)
task = benchmark.task(stream=1, task=0)

# Train ONLY with train data, and optionally validation data
for data_point in task.train:
    audio: np.ndarray = data_point.audio
    label: Diagnosis = data_point.label

# Validate with validation data
# or you can combine this with train data if you don't want to validate
# although we highly recommend that you perform validation
for data_point in task.val:
    audio: np.ndarray = data_point.audio
    label: Diagnosis = data_point.label

# Test with test data
# Do NOT use the test data for training or validation as this would invalidate your experiment
# Only run this when you are finished with a set of experiments
predictions = {}
for data_point in task.test:
    id: str = data_point.id
    audio: np.ndarray = data_point.audio
    label: str = model.classify(audio)
    predictions[id] = label

# Score the predictions
# optionally share back in this repo with a Pull-Request
scores = task.score(predictions)
```

## Tasks

[v1: First Version](./divr_benchmark/tasks/v1/README.md)

## How to cite

Coming soon

# DiVR (Disordered Voice Recognition) - Benchmark

This repository contains the work that enables working with various disordered voice databases using the divr-diagnosis label standardization toolkit.

## Installation

```sh
pip install divr-benchmark
```

## How to use

While you can generate your own tasks, we provide a battery of tasks that we have used across a wide range of experiments. You can read more about them in [Tasks](./divr_benchmark/tasks/README.md).

### Generating tasks

You can generate new tasks from the databases (AVFAD, MEEI, SVD, Torgo, UASpeech, UncommonVoice, VOICED). Of these SVD, Torgo and VOICED as publicly accessible and scripts can download the data automatically provided the database is still available on the expected URLs.

```python
from divr_diagnosis import diagnosis_maps
from divr_benchmark import Benchmark, Diagnosis

benchmark = Benchmark(
    storage_path="/home/user/divr_benchmark/storage",
    version="v1",
    sample_rate=16000,
)
diag_map = diagnosis_maps.CaRLab_2025()


async def filter_func(database_func: DatabaseFunc):
    # You can filter the data by min_tasks, so thate every speaker has at least N audios
    # this is called 'task' because in most datasets the audios represent different vocal tasks
    db = await database_func(name="svd", min_tasks=None)
    diag_level = diag_map.max_diag_level

    def filter_unclassified(tasks): # example of filtering tasks by label
        # You can also get task.speaker_id which can be used to count
        # number of diag/speaker and restrict which diags are used for the dataset
        return [task for task in tasks if not task.label.incompletely_classified]

    return Dataset(
        train=filter_unclassified(db.all_train(level=diag_level)),
        val=filter_unclassified(db.all_val(level=diag_level)),
        test=filter_unclassified(db.all_test(level=diag_level)),
    )

benchmark.generate_task(
    filter_func=filter_func,
    task_path="/home/user/divr_benchmark/tasks/all",
    diagnosis_map=diag_level,
    allow_incomplete_classification=False,
)
```

### Using existing tasks

Almost all functions of the library accept a `level` parameter which decides which level of diagnosis is the operation performed on. These parameters default to the maximum diagnostic level if left as None, i.e. the narrowest diagnosis furthest away from the binary detection.

```python
from divr_diagnosis import diagnosis_maps
from divr_benchmark import Benchmark, Diagnosis

benchmark = Benchmark(
    storage_path="/home/user/divr_benchmark/storage",
    version="v1",
    sample_rate=16000,
)
# The diagnosis map here can be different from the one used for generating the tasks
# the library will automatically map diagnosis which can be mapped to the new map
# automatically, and unmapped items will be left as unclassified
diag_map = diagnosis_maps.CaRLab_2025()

task = benchmark.load_task(
    task_path="/home/user/divr_benchmark/tasks/all",
    diag_level=None,
    diagnosis_map=diag_map,
    load_audios=True,
)

# Training at default level of diagnosis
for train_point in task.train:
    point_id = train_point.id
    audio = train_point.audio
    label = task.diag_to_index(
        diag=train_point.label,
        level=None,
    )

# Training at root/0th level of diagnosis. Equivalent to binary detection
for train_point in task.train:
    point_id = train_point.id
    audio = train_point.audio
    label = task.diag_to_index(
        diag=train_point.label,
        level=0,
    )

# Validating
for val_point in task.val:
    point_id = val_point.id
    audio = val_point.audio
    label = task.diag_to_index(
        diag=val_point.label,
        level=None,
    )

# Testing
for test_point in task.test:
    point_id = test_point.id
    audio = test_point.audio
    label = task.diag_to_index(
        diag=test_point.label,
        level=None,
    )

# Class weights for cross entropy loss
class_weights = task.train_class_weights(level=None) # level defaults to max level of label
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))

# Convert predicted index to diagnosis
diagnosis = task.index_to_diag(
    index=index,
    level=None,
)
print(diagnosis.name)

# Get all unique diagnosis in the data
diagnosis_names = task.unique_diagnosis(level=None)
```

## How to cite

Coming soon

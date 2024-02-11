import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Type, Callable


@dataclass
class HParams:
    # task choice
    benchmark_version = "v1"
    stream: int
    task: int

    # storage path config
    experiment_key: str
    base_path = Path("/home/divr_benchmark")

    # model and features
    DataLoaderClass: Type
    ModelClass: Type
    criterion: Callable
    OptimClass: Type
    lr: float

    # execution
    batch_size: int
    device: torch.device
    num_epochs = 1000
    save_epochs = list(range(0, num_epochs + 1, num_epochs // 10))
    confusion_epochs = list(range(0, num_epochs + 1, 10))
    random_seed = 42
    shuffle_train = True
    save_enabled = True

import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Type, Callable

project_root = Path(__file__).parent.parent.parent.resolve()


@dataclass
class HParams:
    # task choice
    benchmark_version = "v1"
    stream: int
    task: int

    # storage path config
    experiment_key: str
    base_path = Path(f"{project_root}/data/divr_benchmark")
    cache_base_path = Path("/home/storage/divr_benchmark")
    benchmark_path = Path(f"{base_path}/storage")

    # model and features
    DataLoaderClass: Type
    ModelClass: Type
    criterion: Callable
    OptimClass: Type
    lr: float

    # execution
    batch_size: int
    device: torch.device
    cache_key: str
    cache_enabled = True
    num_epochs = 1000
    save_epochs = list(range(0, num_epochs + 1, num_epochs // 10))
    confusion_epochs = list(range(0, num_epochs + 1, 10))
    random_seed = 42
    shuffle_train = True
    save_enabled = False
    tboard_enabled = False

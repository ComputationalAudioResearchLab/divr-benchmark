import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Type, Callable
from ..data_loader import LoaderTypes

project_root = Path(__file__).parent.parent.parent.resolve()


@dataclass
class HParams:
    # task choice
    benchmark_version = "v1"
    stream: int
    task: int

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

    # storage path config
    results_key: str
    checkpoint_key: str
    tensorboard_key: str
    base_path: Path = Path(f"{project_root}/data/divr_benchmark")
    cache_base_path: Path = Path("/home/storage/PRJ-VDML/divr_benchmark")
    # cache_base_path: Path = base_path  # Path("/home/storage/PRJ-VDML/divr_benchmark")
    benchmark_path: Path = Path(f"{base_path}/storage")

    loader_type: LoaderTypes = LoaderTypes.NORMAL
    best_checkpoint_epoch: int | None = None
    num_epochs = 202
    save_epochs = list(range(0, num_epochs + 1, num_epochs // 10))
    confusion_epochs = list(range(0, num_epochs + 1, 10))
    random_seed = 42
    shuffle_train = True
    save_enabled = True
    tboard_enabled = True

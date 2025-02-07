import torch
from torch import nn
from typing import List
from pathlib import Path
from dataclasses import dataclass
from ..model import SavableModule
from ..data_loader import DataLoader
from torch.optim.optimizer import Optimizer

project_root = Path(__file__).parent.parent.parent.resolve()


@dataclass
class HParams:
    confusion_epochs: List[int]
    criterion: nn.Module
    model_name: str
    task_key: str
    model: SavableModule
    data_loader: DataLoader
    num_classes: int
    num_epochs: int
    optimizer: Optimizer
    save_epochs: List[int]

    cache_path: Path = Path(f"{project_root}/.cache")
    device: torch.device = torch.device("cuda")
    sample_rate: int = 16000
    random_seed: int = 42
    save_enabled: bool = True
    shuffle_train: bool = True
    tboard_enabled: bool = True

from torch import nn
from .model import Simple
from torch.optim import Adam
from typing import Literal
from .data_loader import ModifiedCPC
from .trainer import HParams, Trainer

EXPERIMENTS = Literal["S0/ModifiedCPC/Simple", "S1/T1/ModifiedCPC/Simple"]

experiments = [
    HParams(
        experiment_key="S0/ModifiedCPC/Simple",
        stream=0,
        task=1,
        DataLoaderClass=ModifiedCPC,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=8,
    ),
    HParams(
        experiment_key="S1/T1/ModifiedCPC/Simple",
        stream=1,
        task=1,
        DataLoaderClass=ModifiedCPC,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=32,
    ),
]


def experiment(experiment_key: EXPERIMENTS) -> None:
    hparams = next(filter(lambda x: x.experiment_key == experiment_key, experiments))
    Trainer(hparams=hparams).run()

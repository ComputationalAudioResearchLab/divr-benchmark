import torch
from torch import nn
from .model import Simple
from torch.optim import Adam
from typing import Literal
from .data_loader import Data2Vec, MeanMfcc, ModifiedCPC, UnispeechSAT, Wav2Vec
from .trainer import HParams, Trainer

EXPERIMENTS = Literal[
    # S0
    "S0/ModifiedCPC/Simple",
    # S1/T1
    "S1/T1/Data2Vec/Simple",
    "S1/T1/MeanMfcc/Simple",
    "S1/T1/ModifiedCPC/Simple",
    "S1/T1/UnispeechSAT/Simple",
    "S1/T1/Wav2Vec/Simple",
    # S1/T9
    "S1/T9/Data2Vec/Simple",
]
device = torch.device("cuda")
## class weights are derived from train set as that's what is used for training
S0_class_weights = 22805 / torch.LongTensor([9489, 13316]).to(device)
"""
 [normal, pathological]
"""
S1T1_class_weights = 1172 / torch.LongTensor([132, 129, 494, 417]).to(device)
"""
 [functional, muscle_tension, normal, organic] = [132, 129, 494, 417]
"""
S1T9_class_weights = 1172 / torch.LongTensor([132, 129, 494, 68, 208, 137, 4]).to(
    device
)
"""
 [functional, muscle_tension, normal, organic_inflammatory, organic_neuro_muscular, organic_structural, organic_trauma] = [132, 129, 494, 68, 211, 137, 4]
"""

experiments = [
    ## S0
    HParams(
        experiment_key="S0/ModifiedCPC/Simple",
        cache_key="S0",
        stream=0,
        task=1,
        DataLoaderClass=ModifiedCPC,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S0_class_weights),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=8,
        device=device,
    ),
    ## S1/T1
    HParams(
        experiment_key="S1/T1/Data2Vec/Simple",
        cache_key="S1/T1",
        stream=1,
        task=1,
        DataLoaderClass=Data2Vec,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T1_class_weights),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=32,
        device=device,
    ),
    HParams(
        experiment_key="S1/T1/MeanMfcc/Simple",
        cache_key="S1/T1",
        stream=1,
        task=1,
        DataLoaderClass=MeanMfcc,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T1_class_weights),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=32,
        device=device,
    ),
    HParams(
        experiment_key="S1/T1/ModifiedCPC/Simple",
        cache_key="S1/T1",
        stream=1,
        task=1,
        DataLoaderClass=ModifiedCPC,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T1_class_weights),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=32,
        device=device,
    ),
    HParams(
        experiment_key="S1/T1/UnispeechSAT/Simple",
        cache_key="S1/T1",
        stream=1,
        task=1,
        DataLoaderClass=UnispeechSAT,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T1_class_weights),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=32,
        device=device,
    ),
    HParams(
        experiment_key="S1/T1/Wav2Vec/Simple",
        cache_key="S1/T1",
        stream=1,
        task=1,
        DataLoaderClass=Wav2Vec,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T1_class_weights),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=32,
        device=device,
    ),
    ## S1/T9
    HParams(
        experiment_key="S1/T9/Data2Vec/Simple",
        cache_key="S1/T9",
        stream=1,
        task=9,
        DataLoaderClass=Data2Vec,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T9_class_weights),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=32,
        device=device,
    ),
]


def experiment(experiment_key: EXPERIMENTS) -> None:
    hparams = next(filter(lambda x: x.experiment_key == experiment_key, experiments))
    Trainer(hparams=hparams).run()

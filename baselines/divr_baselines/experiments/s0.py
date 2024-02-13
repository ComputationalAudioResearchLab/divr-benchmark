import torch
from torch import nn
from ..model import Simple
from torch.optim import Adam
from ..data_loader import (
    Data2Vec,
    MeanMfcc,
    ModifiedCPC,
    UnispeechSAT,
    Wav2Vec,
    LoaderTypes,
)
from ..trainer import HParams
from .device import device

## class weights are derived from train set as that's what is used for training
S0_class_weights = 22805 / torch.LongTensor([9489, 13316])
"""
 [normal, pathological]
"""

s0_experiments = [
    HParams(
        experiment_key="S0/Data2Vec/Simple",
        cache_key="S0/Data2Vec",
        stream=0,
        task=1,
        DataLoaderClass=Data2Vec,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S0_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
    ),
    HParams(
        experiment_key="S0/MeanMfcc/Simple",
        cache_key="S0/MeanMfcc",
        stream=0,
        task=1,
        DataLoaderClass=MeanMfcc,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S0_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=8,
        device=device,
    ),
    HParams(
        experiment_key="S0/ModifiedCPC/Simple",
        cache_key="S0/ModifiedCPC",
        stream=0,
        task=1,
        DataLoaderClass=ModifiedCPC,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S0_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=8,
        device=device,
    ),
    HParams(
        experiment_key="S0/UnispeechSAT/Simple",
        cache_key="S0/UnispeechSAT",
        stream=0,
        task=1,
        DataLoaderClass=UnispeechSAT,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S0_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=8,
        device=device,
        loader_type=LoaderTypes.NORMAL,
    ),
    HParams(
        experiment_key="S0/Wav2Vec/Simple",
        cache_key="S0/Wav2Vec",
        stream=0,
        task=1,
        DataLoaderClass=Wav2Vec,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S0_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=8,
        device=device,
        loader_type=LoaderTypes.BATCH_AHEAD,
    ),
]

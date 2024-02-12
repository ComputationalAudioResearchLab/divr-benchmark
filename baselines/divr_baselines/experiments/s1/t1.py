import torch
from torch import nn
from ...model import Simple
from torch.optim import Adam
from ...data_loader import Data2Vec, MeanMfcc, ModifiedCPC, UnispeechSAT, Wav2Vec
from ...trainer import HParams
from ..device import device

## class weights are derived from train set as that's what is used for training
S1T1_class_weights = 1172 / torch.LongTensor([132, 129, 494, 417])
"""
 [functional, muscle_tension, normal, organic] = [132, 129, 494, 417]
"""

t1_experiments = [
    HParams(
        experiment_key="S1/T1/Data2Vec/Simple",
        cache_key="S1/T1/Data2Vec",
        stream=1,
        task=1,
        DataLoaderClass=Data2Vec,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T1_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=32,
        device=device,
    ),
    HParams(
        experiment_key="S1/T1/MeanMfcc/Simple",
        cache_key="S1/T1/MeanMfcc",
        stream=1,
        task=1,
        DataLoaderClass=MeanMfcc,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T1_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=32,
        device=device,
    ),
    HParams(
        experiment_key="S1/T1/ModifiedCPC/Simple",
        cache_key="S1/T1/ModifiedCPC",
        stream=1,
        task=1,
        DataLoaderClass=ModifiedCPC,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T1_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=32,
        device=device,
    ),
    HParams(
        experiment_key="S1/T1/UnispeechSAT/Simple",
        cache_key="S1/T1/UnispeechSAT",
        stream=1,
        task=1,
        DataLoaderClass=UnispeechSAT,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T1_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=32,
        device=device,
    ),
    HParams(
        experiment_key="S1/T1/Wav2Vec/Simple",
        cache_key="S1/T1/Wav2Vec",
        stream=1,
        task=1,
        DataLoaderClass=Wav2Vec,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T1_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=32,
        device=device,
    ),
]

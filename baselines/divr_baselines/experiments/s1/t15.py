import torch
from torch import nn
from ...model import Simple
from torch.optim import Adam
from ...data_loader import (
    Data2Vec,
    Mfcc,
    MfccWithDeltas,
    ModifiedCPC,
    UnispeechSAT,
    Wav2Vec,
    LoaderTypes,
)
from ...trainer import HParams
from ..device import device

## class weights are derived from train set as that's what is used for training
S1T15_class_weights = (
    (88 + 44) + (85 + 43) + (257 + 210) + (30 + 38) + (125 + 83) + (66 + 71) + (2 + 2)
) / torch.LongTensor(
    [(88 + 44), (85 + 43), (257 + 210), (30 + 38), (125 + 83), (66 + 71), (2 + 2)]
)
"""
 [functional, muscle_tension, normal, organic_inflammatory, organic_neuro_muscular, organic_structural, organic_trauma] = [132, 129, 494, 68, 211, 137, 4]
"""

t15_experiments = [
    HParams(
        results_key="S1/T15/Data2Vec/Simple4",
        checkpoint_key="S1/T15/Data2Vec/Simple4",
        tensorboard_key="S1/T15/Data2Vec/Simple4",
        cache_key="S1/T15/Data2Vec",
        stream=1,
        task=15,
        DataLoaderClass=Data2Vec,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T15_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=163,
    ),
    HParams(
        results_key="S1/T15/Mfcc/Simple4",
        checkpoint_key="S1/T15/Mfcc/Simple4",
        tensorboard_key="S1/T15/Mfcc/Simple4",
        cache_key="S1/T15/Mfcc",
        stream=1,
        task=15,
        DataLoaderClass=Mfcc,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T15_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=113,
    ),
    HParams(
        results_key="S1/T15/MfccWithDeltas/Simple4",
        checkpoint_key="S1/T15/MfccWithDeltas/Simple4",
        tensorboard_key="S1/T15/MfccWithDeltas/Simple4",
        cache_key="S1/T15/MfccWithDeltas",
        stream=1,
        task=15,
        DataLoaderClass=MfccWithDeltas,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T15_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=183,
    ),
    HParams(
        results_key="S1/T15/ModifiedCPC/Simple4",
        checkpoint_key="S1/T15/ModifiedCPC/Simple4",
        tensorboard_key="S1/T15/ModifiedCPC/Simple4",
        cache_key="S1/T15/ModifiedCPC",
        stream=1,
        task=15,
        DataLoaderClass=ModifiedCPC,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T15_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=183,
    ),
    HParams(
        results_key="S1/T15/UnispeechSAT/Simple4",
        checkpoint_key="S1/T15/UnispeechSAT/Simple4",
        tensorboard_key="S1/T15/UnispeechSAT/Simple4",
        cache_key="S1/T15/UnispeechSAT",
        stream=1,
        task=15,
        DataLoaderClass=UnispeechSAT,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T15_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=171,
    ),
    HParams(
        results_key="S1/T15/Wav2Vec/Simple4",
        checkpoint_key="S1/T15/Wav2Vec/Simple4",
        tensorboard_key="S1/T15/Wav2Vec/Simple4",
        cache_key="S1/T15/Wav2Vec",
        stream=1,
        task=15,
        DataLoaderClass=Wav2Vec,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T15_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=193,
    ),
]

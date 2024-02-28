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
S1T13_class_weights = (
    (88 + 44) + (86 + 43) + (279 + 215) + (30 + 38) + (125 + 83) + (66 + 71) + (2 + 2)
) / torch.LongTensor(
    [(88 + 44), (86 + 43), (279 + 215), (30 + 38), (125 + 83), (66 + 71), (2 + 2)]
)
"""
 [functional, muscle_tension, normal, organic_inflammatory, organic_neuro_muscular, organic_structural, organic_trauma] = [132, 129, 494, 68, 211, 137, 4]
"""

t13_experiments = [
    HParams(
        results_key="S1/T13/Data2Vec/Simple4",
        checkpoint_key="S1/T13/Data2Vec/Simple4",
        tensorboard_key="S1/T13/Data2Vec/Simple4",
        cache_key="S1/T13/Data2Vec",
        stream=1,
        task=13,
        DataLoaderClass=Data2Vec,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T13_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=141,
    ),
    HParams(
        results_key="S1/T13/Mfcc/Simple4",
        checkpoint_key="S1/T13/Mfcc/Simple4",
        tensorboard_key="S1/T13/Mfcc/Simple4",
        cache_key="S1/T13/Mfcc",
        stream=1,
        task=13,
        DataLoaderClass=Mfcc,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T13_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=172,
    ),
    HParams(
        results_key="S1/T13/MfccWithDeltas/Simple4",
        checkpoint_key="S1/T13/MfccWithDeltas/Simple4",
        tensorboard_key="S1/T13/MfccWithDeltas/Simple4",
        cache_key="S1/T13/MfccWithDeltas",
        stream=1,
        task=13,
        DataLoaderClass=MfccWithDeltas,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T13_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=172,
    ),
    HParams(
        results_key="S1/T13/ModifiedCPC/Simple4",
        checkpoint_key="S1/T13/ModifiedCPC/Simple4",
        tensorboard_key="S1/T13/ModifiedCPC/Simple4",
        cache_key="S1/T13/ModifiedCPC",
        stream=1,
        task=13,
        DataLoaderClass=ModifiedCPC,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T13_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=172,
    ),
    HParams(
        results_key="S1/T13/UnispeechSAT/Simple4",
        checkpoint_key="S1/T13/UnispeechSAT/Simple4",
        tensorboard_key="S1/T13/UnispeechSAT/Simple4",
        cache_key="S1/T13/UnispeechSAT",
        stream=1,
        task=13,
        DataLoaderClass=UnispeechSAT,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T13_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=191,
    ),
    HParams(
        results_key="S1/T13/Wav2Vec/Simple4",
        checkpoint_key="S1/T13/Wav2Vec/Simple4",
        tensorboard_key="S1/T13/Wav2Vec/Simple4",
        cache_key="S1/T13/Wav2Vec",
        stream=1,
        task=13,
        DataLoaderClass=Wav2Vec,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T13_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=174,
    ),
]

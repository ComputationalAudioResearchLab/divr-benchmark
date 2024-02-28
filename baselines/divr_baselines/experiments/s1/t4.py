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
S1T4_class_weights = (
    (88 + 44) + (86 + 43) + (279 + 215) + (223 + 194)
) / torch.LongTensor([(88 + 44), (86 + 43), (279 + 215), (223 + 194)])
"""
 [functional, muscle_tension, normal, organic] = [132, 129, 494, 417]
"""

t4_experiments = [
    #### Batch size = 4
    HParams(
        results_key="S1/T4/Data2Vec/Simple4",
        checkpoint_key="S1/T4/Data2Vec/Simple4",
        tensorboard_key="S1/T4/Data2Vec/Simple4",
        cache_key="S1/T4/Data2Vec",
        stream=1,
        task=4,
        DataLoaderClass=Data2Vec,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T4_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=116,
    ),
    HParams(
        results_key="S1/T4/Mfcc/Simple4",
        checkpoint_key="S1/T4/Mfcc/Simple4",
        tensorboard_key="S1/T4/Mfcc/Simple4",
        cache_key="S1/T4/Mfcc",
        stream=1,
        task=4,
        DataLoaderClass=Mfcc,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T4_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=183,
    ),
    HParams(
        results_key="S1/T4/MfccWithDeltas/Simple4",
        checkpoint_key="S1/T4/MfccWithDeltas/Simple4",
        tensorboard_key="S1/T4/MfccWithDeltas/Simple4",
        cache_key="S1/T4/MfccWithDeltas",
        stream=1,
        task=4,
        DataLoaderClass=MfccWithDeltas,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T4_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=129,
    ),
    HParams(
        results_key="S1/T4/ModifiedCPC/Simple4",
        checkpoint_key="S1/T4/ModifiedCPC/Simple4",
        tensorboard_key="S1/T4/ModifiedCPC/Simple4",
        cache_key="S1/T4/ModifiedCPC",
        stream=1,
        task=4,
        DataLoaderClass=ModifiedCPC,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T4_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=186,
    ),
    HParams(
        results_key="S1/T4/UnispeechSAT/Simple4",
        checkpoint_key="S1/T4/UnispeechSAT/Simple4",
        tensorboard_key="S1/T4/UnispeechSAT/Simple4",
        cache_key="S1/T4/UnispeechSAT",
        stream=1,
        task=4,
        DataLoaderClass=UnispeechSAT,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T4_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=39,
    ),
    HParams(
        results_key="S1/T4/Wav2Vec/Simple4",
        checkpoint_key="S1/T4/Wav2Vec/Simple4",
        tensorboard_key="S1/T4/Wav2Vec/Simple4",
        cache_key="S1/T4/Wav2Vec",
        stream=1,
        task=4,
        DataLoaderClass=Wav2Vec,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T4_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=162,
    ),
]

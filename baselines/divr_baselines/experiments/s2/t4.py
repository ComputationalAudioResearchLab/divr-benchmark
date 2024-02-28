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
S2T4_class_weights = ((381 + 254) + (530 + 343)) / torch.LongTensor(
    [(381 + 254), (530 + 343)]
)
"""
 [normal, pathological] = [(381+254), (530+343)]
"""

t4_experiments = [
    #### Batch size = 4
    HParams(
        results_key="S2/T4/Data2Vec/Simple4",
        checkpoint_key="S2/T4/Data2Vec/Simple4",
        tensorboard_key="S2/T4/Data2Vec/Simple4",
        cache_key="S2/T4/Data2Vec",
        stream=2,
        task=4,
        DataLoaderClass=Data2Vec,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S2T4_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=26,
    ),
    HParams(
        results_key="S2/T4/Mfcc/Simple4",
        checkpoint_key="S2/T4/Mfcc/Simple4",
        tensorboard_key="S2/T4/Mfcc/Simple4",
        cache_key="S2/T4/Mfcc",
        stream=2,
        task=4,
        DataLoaderClass=Mfcc,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S2T4_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=128,
    ),
    HParams(
        results_key="S2/T4/MfccWithDeltas/Simple4",
        checkpoint_key="S2/T4/MfccWithDeltas/Simple4",
        tensorboard_key="S2/T4/MfccWithDeltas/Simple4",
        cache_key="S2/T4/MfccWithDeltas",
        stream=2,
        task=4,
        DataLoaderClass=MfccWithDeltas,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S2T4_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=164,
    ),
    HParams(
        results_key="S2/T4/ModifiedCPC/Simple4",
        checkpoint_key="S2/T4/ModifiedCPC/Simple4",
        tensorboard_key="S2/T4/ModifiedCPC/Simple4",
        cache_key="S2/T4/ModifiedCPC",
        stream=2,
        task=4,
        DataLoaderClass=ModifiedCPC,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S2T4_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=107,
    ),
    HParams(
        results_key="S2/T4/UnispeechSAT/Simple4",
        checkpoint_key="S2/T4/UnispeechSAT/Simple4",
        tensorboard_key="S2/T4/UnispeechSAT/Simple4",
        cache_key="S2/T4/UnispeechSAT",
        stream=2,
        task=4,
        DataLoaderClass=UnispeechSAT,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S2T4_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=200,
    ),
    HParams(
        results_key="S2/T4/Wav2Vec/Simple4",
        checkpoint_key="S2/T4/Wav2Vec/Simple4",
        tensorboard_key="S2/T4/Wav2Vec/Simple4",
        cache_key="S2/T4/Wav2Vec",
        stream=2,
        task=4,
        DataLoaderClass=Wav2Vec,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S2T4_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=197,
    ),
]

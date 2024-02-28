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
S2T2_class_weights = (
    (119 + 51) + (121 + 44) + (381 + 254) + (290 + 248)
) / torch.LongTensor([(119 + 51), (121 + 44), (381 + 254), (290 + 248)])
"""
 [functional, muscle_tension, normal, organic]
"""

t2_experiments = [
    #### Batch size = 4
    HParams(
        results_key="S2/T2/Data2Vec/Simple4",
        checkpoint_key="S2/T2/Data2Vec/Simple4",
        tensorboard_key="S2/T2/Data2Vec/Simple4",
        cache_key="S2/T2/Data2Vec",
        stream=2,
        task=2,
        DataLoaderClass=Data2Vec,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S2T2_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=165,
    ),
    HParams(
        results_key="S2/T2/Mfcc/Simple4",
        checkpoint_key="S2/T2/Mfcc/Simple4",
        tensorboard_key="S2/T2/Mfcc/Simple4",
        cache_key="S2/T2/Mfcc",
        stream=2,
        task=2,
        DataLoaderClass=Mfcc,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S2T2_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=79,
    ),
    HParams(
        results_key="S2/T2/MfccWithDeltas/Simple4",
        checkpoint_key="S2/T2/MfccWithDeltas/Simple4",
        tensorboard_key="S2/T2/MfccWithDeltas/Simple4",
        cache_key="S2/T2/MfccWithDeltas",
        stream=2,
        task=2,
        DataLoaderClass=MfccWithDeltas,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S2T2_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=47,
    ),
    HParams(
        results_key="S2/T2/ModifiedCPC/Simple4",
        checkpoint_key="S2/T2/ModifiedCPC/Simple4",
        tensorboard_key="S2/T2/ModifiedCPC/Simple4",
        cache_key="S2/T2/ModifiedCPC",
        stream=2,
        task=2,
        DataLoaderClass=ModifiedCPC,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S2T2_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=198,
    ),
    HParams(
        results_key="S2/T2/UnispeechSAT/Simple4",
        checkpoint_key="S2/T2/UnispeechSAT/Simple4",
        tensorboard_key="S2/T2/UnispeechSAT/Simple4",
        cache_key="S2/T2/UnispeechSAT",
        stream=2,
        task=2,
        DataLoaderClass=UnispeechSAT,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S2T2_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=200,
    ),
    HParams(
        results_key="S2/T2/Wav2Vec/Simple4",
        checkpoint_key="S2/T2/Wav2Vec/Simple4",
        tensorboard_key="S2/T2/Wav2Vec/Simple4",
        cache_key="S2/T2/Wav2Vec",
        stream=2,
        task=2,
        DataLoaderClass=Wav2Vec,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S2T2_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=126,
    ),
]

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
S1T2_class_weights = (
    (88 + 44) + (86 + 43) + (279 + 215) + (223 + 194)
) / torch.LongTensor([(88 + 44), (86 + 43), (279 + 215), (223 + 194)])
"""
 [functional, muscle_tension, normal, organic] = [132, 129, 494, 417]
"""

t2_experiments = [
    #### Batch size = 4
    HParams(
        results_key="S1/T2/Data2Vec/Simple4",
        checkpoint_key="S1/T2/Data2Vec/Simple4",
        tensorboard_key="S1/T2/Data2Vec/Simple4",
        cache_key="S1/T2/Data2Vec",
        stream=1,
        task=2,
        DataLoaderClass=Data2Vec,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T2_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=68,
    ),
    HParams(
        results_key="S1/T2/Mfcc/Simple4",
        checkpoint_key="S1/T2/Mfcc/Simple4",
        tensorboard_key="S1/T2/Mfcc/Simple4",
        cache_key="S1/T2/Mfcc",
        stream=1,
        task=2,
        DataLoaderClass=Mfcc,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T2_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=47,
    ),
    HParams(
        results_key="S1/T2/MfccWithDeltas/Simple4",
        checkpoint_key="S1/T2/MfccWithDeltas/Simple4",
        tensorboard_key="S1/T2/MfccWithDeltas/Simple4",
        cache_key="S1/T2/MfccWithDeltas",
        stream=1,
        task=2,
        DataLoaderClass=MfccWithDeltas,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T2_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=195,
    ),
    HParams(
        results_key="S1/T2/ModifiedCPC/Simple4",
        checkpoint_key="S1/T2/ModifiedCPC/Simple4",
        tensorboard_key="S1/T2/ModifiedCPC/Simple4",
        cache_key="S1/T2/ModifiedCPC",
        stream=1,
        task=2,
        DataLoaderClass=ModifiedCPC,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T2_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=18,
    ),
    HParams(
        results_key="S1/T2/UnispeechSAT/Simple4",
        checkpoint_key="S1/T2/UnispeechSAT/Simple4",
        tensorboard_key="S1/T2/UnispeechSAT/Simple4",
        cache_key="S1/T2/UnispeechSAT",
        stream=1,
        task=2,
        DataLoaderClass=UnispeechSAT,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T2_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=77,
    ),
    HParams(
        results_key="S1/T2/Wav2Vec/Simple4",
        checkpoint_key="S1/T2/Wav2Vec/Simple4",
        tensorboard_key="S1/T2/Wav2Vec/Simple4",
        cache_key="S1/T2/Wav2Vec",
        stream=1,
        task=2,
        DataLoaderClass=Wav2Vec,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T2_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=156,
    ),
]

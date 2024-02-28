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
S1T3_class_weights = (
    (88 + 44) + (86 + 43) + (279 + 215) + (223 + 194)
) / torch.LongTensor([(88 + 44), (86 + 43), (279 + 215), (223 + 194)])
"""
 [functional, muscle_tension, normal, organic] = [132, 129, 494, 417]
"""

t3_experiments = [
    #### Batch size = 4
    HParams(
        results_key="S1/T3/Data2Vec/Simple4",
        checkpoint_key="S1/T3/Data2Vec/Simple4",
        tensorboard_key="S1/T3/Data2Vec/Simple4",
        cache_key="S1/T3/Data2Vec",
        stream=1,
        task=3,
        DataLoaderClass=Data2Vec,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T3_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=24,
    ),
    HParams(
        results_key="S1/T3/Mfcc/Simple4",
        checkpoint_key="S1/T3/Mfcc/Simple4",
        tensorboard_key="S1/T3/Mfcc/Simple4",
        cache_key="S1/T3/Mfcc",
        stream=1,
        task=3,
        DataLoaderClass=Mfcc,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T3_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=200,
    ),
    HParams(
        results_key="S1/T3/MfccWithDeltas/Simple4",
        checkpoint_key="S1/T3/MfccWithDeltas/Simple4",
        tensorboard_key="S1/T3/MfccWithDeltas/Simple4",
        cache_key="S1/T3/MfccWithDeltas",
        stream=1,
        task=3,
        DataLoaderClass=MfccWithDeltas,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T3_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=71,
    ),
    HParams(
        results_key="S1/T3/ModifiedCPC/Simple4",
        checkpoint_key="S1/T3/ModifiedCPC/Simple4",
        tensorboard_key="S1/T3/ModifiedCPC/Simple4",
        cache_key="S1/T3/ModifiedCPC",
        stream=1,
        task=3,
        DataLoaderClass=ModifiedCPC,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T3_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=184,
    ),
    HParams(
        results_key="S1/T3/UnispeechSAT/Simple4",
        checkpoint_key="S1/T3/UnispeechSAT/Simple4",
        tensorboard_key="S1/T3/UnispeechSAT/Simple4",
        cache_key="S1/T3/UnispeechSAT",
        stream=1,
        task=3,
        DataLoaderClass=UnispeechSAT,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T3_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=149,
    ),
    HParams(
        results_key="S1/T3/Wav2Vec/Simple4",
        checkpoint_key="S1/T3/Wav2Vec/Simple4",
        tensorboard_key="S1/T3/Wav2Vec/Simple4",
        cache_key="S1/T3/Wav2Vec",
        stream=1,
        task=3,
        DataLoaderClass=Wav2Vec,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T3_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=98,
    ),
]

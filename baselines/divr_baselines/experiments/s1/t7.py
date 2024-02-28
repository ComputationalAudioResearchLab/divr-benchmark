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
S1T7_class_weights = (
    (88 + 44) + (85 + 43) + (257 + 210) + (223 + 194)
) / torch.LongTensor([(88 + 44), (85 + 43), (257 + 210), (223 + 194)])
"""
 [functional, muscle_tension, normal, organic] = [132, 128, 467, 417]
"""

t7_experiments = [
    #### Batch size = 4
    HParams(
        results_key="S1/T7/Data2Vec/Simple4",
        checkpoint_key="S1/T7/Data2Vec/Simple4",
        tensorboard_key="S1/T7/Data2Vec/Simple4",
        cache_key="S1/T7/Data2Vec",
        stream=1,
        task=7,
        DataLoaderClass=Data2Vec,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T7_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=87,
    ),
    HParams(
        results_key="S1/T7/Mfcc/Simple4",
        checkpoint_key="S1/T7/Mfcc/Simple4",
        tensorboard_key="S1/T7/Mfcc/Simple4",
        cache_key="S1/T7/Mfcc",
        stream=1,
        task=7,
        DataLoaderClass=Mfcc,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T7_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=133,
    ),
    HParams(
        results_key="S1/T7/MfccWithDeltas/Simple4",
        checkpoint_key="S1/T7/MfccWithDeltas/Simple4",
        tensorboard_key="S1/T7/MfccWithDeltas/Simple4",
        cache_key="S1/T7/MfccWithDeltas",
        stream=1,
        task=7,
        DataLoaderClass=MfccWithDeltas,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T7_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=113,
    ),
    HParams(
        results_key="S1/T7/ModifiedCPC/Simple4",
        checkpoint_key="S1/T7/ModifiedCPC/Simple4",
        tensorboard_key="S1/T7/ModifiedCPC/Simple4",
        cache_key="S1/T7/ModifiedCPC",
        stream=1,
        task=7,
        DataLoaderClass=ModifiedCPC,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T7_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=169,
    ),
    HParams(
        results_key="S1/T7/UnispeechSAT/Simple4",
        checkpoint_key="S1/T7/UnispeechSAT/Simple4",
        tensorboard_key="S1/T7/UnispeechSAT/Simple4",
        cache_key="S1/T7/UnispeechSAT",
        stream=1,
        task=7,
        DataLoaderClass=UnispeechSAT,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T7_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=98,
    ),
    HParams(
        results_key="S1/T7/Wav2Vec/Simple4",
        checkpoint_key="S1/T7/Wav2Vec/Simple4",
        tensorboard_key="S1/T7/Wav2Vec/Simple4",
        cache_key="S1/T7/Wav2Vec",
        stream=1,
        task=7,
        DataLoaderClass=Wav2Vec,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T7_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=61,
    ),
]

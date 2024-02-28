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
S1T12_class_weights = (
    (88 + 44) + (86 + 43) + (279 + 215) + (30 + 38) + (125 + 83) + (66 + 71) + (2 + 2)
) / torch.LongTensor(
    [(88 + 44), (86 + 43), (279 + 215), (30 + 38), (125 + 83), (66 + 71), (2 + 2)]
)
"""
 [functional, muscle_tension, normal, organic_inflammatory, organic_neuro_muscular, organic_structural, organic_trauma] = [132, 129, 494, 68, 211, 137, 4]
"""

t12_experiments = [
    HParams(
        results_key="S1/T12/Data2Vec/Simple4",
        checkpoint_key="S1/T12/Data2Vec/Simple4",
        tensorboard_key="S1/T12/Data2Vec/Simple4",
        cache_key="S1/T12/Data2Vec",
        stream=1,
        task=12,
        DataLoaderClass=Data2Vec,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T12_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=87,
    ),
    HParams(
        results_key="S1/T12/Mfcc/Simple4",
        checkpoint_key="S1/T12/Mfcc/Simple4",
        tensorboard_key="S1/T12/Mfcc/Simple4",
        cache_key="S1/T12/Mfcc",
        stream=1,
        task=12,
        DataLoaderClass=Mfcc,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T12_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=141,
    ),
    HParams(
        results_key="S1/T12/MfccWithDeltas/Simple4",
        checkpoint_key="S1/T12/MfccWithDeltas/Simple4",
        tensorboard_key="S1/T12/MfccWithDeltas/Simple4",
        cache_key="S1/T12/MfccWithDeltas",
        stream=1,
        task=12,
        DataLoaderClass=MfccWithDeltas,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T12_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=142,
    ),
    HParams(
        results_key="S1/T12/ModifiedCPC/Simple4",
        checkpoint_key="S1/T12/ModifiedCPC/Simple4",
        tensorboard_key="S1/T12/ModifiedCPC/Simple4",
        cache_key="S1/T12/ModifiedCPC",
        stream=1,
        task=12,
        DataLoaderClass=ModifiedCPC,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T12_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=172,
    ),
    HParams(
        results_key="S1/T12/UnispeechSAT/Simple4",
        checkpoint_key="S1/T12/UnispeechSAT/Simple4",
        tensorboard_key="S1/T12/UnispeechSAT/Simple4",
        cache_key="S1/T12/UnispeechSAT",
        stream=1,
        task=12,
        DataLoaderClass=UnispeechSAT,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T12_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=164,
    ),
    HParams(
        results_key="S1/T12/Wav2Vec/Simple4",
        checkpoint_key="S1/T12/Wav2Vec/Simple4",
        tensorboard_key="S1/T12/Wav2Vec/Simple4",
        cache_key="S1/T12/Wav2Vec",
        stream=1,
        task=12,
        DataLoaderClass=Wav2Vec,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T12_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=58,
    ),
]

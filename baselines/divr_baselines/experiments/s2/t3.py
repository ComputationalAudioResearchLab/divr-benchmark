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
S2T3_class_weights = (
    (119 + 51)
    + (121 + 44)
    + (381 + 254)
    + (34 + 55)
    + (163 + 101)
    + (91 + 89)
    + (2 + 3)
) / torch.LongTensor(
    [(119 + 51), (121 + 44), (381 + 254), (34 + 55), (163 + 101), (91 + 89), (2 + 3)]
)
"""
 [functional, muscle_tension, normal, organic_inflammatory, organic_neuro_muscular, organic_structural, organic_trauma]
"""

t3_experiments = [
    #### Batch size = 4
    HParams(
        results_key="S2/T3/Data2Vec/Simple4",
        checkpoint_key="S2/T3/Data2Vec/Simple4",
        tensorboard_key="S2/T3/Data2Vec/Simple4",
        cache_key="S2/T3/Data2Vec",
        stream=2,
        task=3,
        DataLoaderClass=Data2Vec,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S2T3_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=163,
    ),
    HParams(
        results_key="S2/T3/Mfcc/Simple4",
        checkpoint_key="S2/T3/Mfcc/Simple4",
        tensorboard_key="S2/T3/Mfcc/Simple4",
        cache_key="S2/T3/Mfcc",
        stream=2,
        task=3,
        DataLoaderClass=Mfcc,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S2T3_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=69,
    ),
    HParams(
        results_key="S2/T3/MfccWithDeltas/Simple4",
        checkpoint_key="S2/T3/MfccWithDeltas/Simple4",
        tensorboard_key="S2/T3/MfccWithDeltas/Simple4",
        cache_key="S2/T3/MfccWithDeltas",
        stream=2,
        task=3,
        DataLoaderClass=MfccWithDeltas,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S2T3_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=178,
    ),
    HParams(
        results_key="S2/T3/ModifiedCPC/Simple4",
        checkpoint_key="S2/T3/ModifiedCPC/Simple4",
        tensorboard_key="S2/T3/ModifiedCPC/Simple4",
        cache_key="S2/T3/ModifiedCPC",
        stream=2,
        task=3,
        DataLoaderClass=ModifiedCPC,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S2T3_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=185,
    ),
    HParams(
        results_key="S2/T3/UnispeechSAT/Simple4",
        checkpoint_key="S2/T3/UnispeechSAT/Simple4",
        tensorboard_key="S2/T3/UnispeechSAT/Simple4",
        cache_key="S2/T3/UnispeechSAT",
        stream=2,
        task=3,
        DataLoaderClass=UnispeechSAT,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S2T3_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=200,
    ),
    HParams(
        results_key="S2/T3/Wav2Vec/Simple4",
        checkpoint_key="S2/T3/Wav2Vec/Simple4",
        tensorboard_key="S2/T3/Wav2Vec/Simple4",
        cache_key="S2/T3/Wav2Vec",
        stream=2,
        task=3,
        DataLoaderClass=Wav2Vec,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S2T3_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=169,
    ),
]

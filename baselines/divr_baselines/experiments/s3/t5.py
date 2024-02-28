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
S3T5_class_weights = (
    (0) + (19 + 17) + (59 + 474) + (13 + 15) + (9 + 2) + (2) + (0) + (361 + 468)
) / torch.LongTensor(
    [(1), (19 + 17), (59 + 474), (13 + 15), (9 + 2), (2), (1), (361 + 468)]
)
"""
 [functional, muscle_tension, normal, organic_inflammatory, organic_neuro_muscular, organic_trauma, organic_structural, unclassified_pathology]
"""

t5_experiments = [
    #### Batch size = 4
    HParams(
        results_key="S3/T5/Data2Vec/Simple4",
        checkpoint_key="S3/T5/Data2Vec/Simple4",
        tensorboard_key="S3/T5/Data2Vec/Simple4",
        cache_key="S3/T5/Data2Vec",
        stream=3,
        task=5,
        DataLoaderClass=Data2Vec,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S3T5_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=191,
    ),
    HParams(
        results_key="S3/T5/Mfcc/Simple4",
        checkpoint_key="S3/T5/Mfcc/Simple4",
        tensorboard_key="S3/T5/Mfcc/Simple4",
        cache_key="S3/T5/Mfcc",
        stream=3,
        task=5,
        DataLoaderClass=Mfcc,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S3T5_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=116,
    ),
    HParams(
        results_key="S3/T5/MfccWithDeltas/Simple4",
        checkpoint_key="S3/T5/MfccWithDeltas/Simple4",
        tensorboard_key="S3/T5/MfccWithDeltas/Simple4",
        cache_key="S3/T5/MfccWithDeltas",
        stream=3,
        task=5,
        DataLoaderClass=MfccWithDeltas,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S3T5_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=101,
    ),
    HParams(
        results_key="S3/T5/ModifiedCPC/Simple4",
        checkpoint_key="S3/T5/ModifiedCPC/Simple4",
        tensorboard_key="S3/T5/ModifiedCPC/Simple4",
        cache_key="S3/T5/ModifiedCPC",
        stream=3,
        task=5,
        DataLoaderClass=ModifiedCPC,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S3T5_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=195,
    ),
    HParams(
        results_key="S3/T5/UnispeechSAT/Simple4",
        checkpoint_key="S3/T5/UnispeechSAT/Simple4",
        tensorboard_key="S3/T5/UnispeechSAT/Simple4",
        cache_key="S3/T5/UnispeechSAT",
        stream=3,
        task=5,
        DataLoaderClass=UnispeechSAT,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S3T5_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=187,
    ),
    HParams(
        results_key="S3/T5/Wav2Vec/Simple4",
        checkpoint_key="S3/T5/Wav2Vec/Simple4",
        tensorboard_key="S3/T5/Wav2Vec/Simple4",
        cache_key="S3/T5/Wav2Vec",
        stream=3,
        task=5,
        DataLoaderClass=Wav2Vec,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S3T5_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=74,
    ),
]

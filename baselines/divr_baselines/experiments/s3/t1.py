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
S3T1_class_weights = (
    (0) + (19 + 17) + (59 + 474) + (24 + 17) + (361 + 468)
) / torch.LongTensor([1, (19 + 17), (59 + 474), (24 + 17), (361 + 468)])
"""
 [functional, muscle_tension, normal, organic, unclassified_pathology]
"""

t1_experiments = [
    #### Batch size = 4
    HParams(
        results_key="S3/T1/Data2Vec/Simple4",
        checkpoint_key="S3/T1/Data2Vec/Simple4",
        tensorboard_key="S3/T1/Data2Vec/Simple4",
        cache_key="S3/T1/Data2Vec",
        stream=3,
        task=1,
        DataLoaderClass=Data2Vec,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S3T1_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=139,
    ),
    HParams(
        results_key="S3/T1/Mfcc/Simple4",
        checkpoint_key="S3/T1/Mfcc/Simple4",
        tensorboard_key="S3/T1/Mfcc/Simple4",
        cache_key="S3/T1/Mfcc",
        stream=3,
        task=1,
        DataLoaderClass=Mfcc,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S3T1_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=101,
    ),
    HParams(
        results_key="S3/T1/MfccWithDeltas/Simple4",
        checkpoint_key="S3/T1/MfccWithDeltas/Simple4",
        tensorboard_key="S3/T1/MfccWithDeltas/Simple4",
        cache_key="S3/T1/MfccWithDeltas",
        stream=3,
        task=1,
        DataLoaderClass=MfccWithDeltas,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S3T1_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=200,
    ),
    HParams(
        results_key="S3/T1/ModifiedCPC/Simple4",
        checkpoint_key="S3/T1/ModifiedCPC/Simple4",
        tensorboard_key="S3/T1/ModifiedCPC/Simple4",
        cache_key="S3/T1/ModifiedCPC",
        stream=3,
        task=1,
        DataLoaderClass=ModifiedCPC,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S3T1_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=172,
    ),
    HParams(
        results_key="S3/T1/UnispeechSAT/Simple4",
        checkpoint_key="S3/T1/UnispeechSAT/Simple4",
        tensorboard_key="S3/T1/UnispeechSAT/Simple4",
        cache_key="S3/T1/UnispeechSAT",
        stream=3,
        task=1,
        DataLoaderClass=UnispeechSAT,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S3T1_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=200,
    ),
    HParams(
        results_key="S3/T1/Wav2Vec/Simple4",
        checkpoint_key="S3/T1/Wav2Vec/Simple4",
        tensorboard_key="S3/T1/Wav2Vec/Simple4",
        cache_key="S3/T1/Wav2Vec",
        stream=3,
        task=1,
        DataLoaderClass=Wav2Vec,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S3T1_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=14,
    ),
]

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
S3T2_class_weights = ((59 + 474) + (404 + 502)) / torch.LongTensor(
    [(59 + 474), (404 + 502)]
)
"""
 [normal, pathological]
"""

t2_experiments = [
    #### Batch size = 4
    HParams(
        results_key="S3/T2/Data2Vec/Simple4",
        checkpoint_key="S3/T2/Data2Vec/Simple4",
        tensorboard_key="S3/T2/Data2Vec/Simple4",
        cache_key="S3/T2/Data2Vec",
        stream=3,
        task=2,
        DataLoaderClass=Data2Vec,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S3T2_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=19,
    ),
    HParams(
        results_key="S3/T2/Mfcc/Simple4",
        checkpoint_key="S3/T2/Mfcc/Simple4",
        tensorboard_key="S3/T2/Mfcc/Simple4",
        cache_key="S3/T2/Mfcc",
        stream=3,
        task=2,
        DataLoaderClass=Mfcc,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S3T2_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=198,
    ),
    HParams(
        results_key="S3/T2/MfccWithDeltas/Simple4",
        checkpoint_key="S3/T2/MfccWithDeltas/Simple4",
        tensorboard_key="S3/T2/MfccWithDeltas/Simple4",
        cache_key="S3/T2/MfccWithDeltas",
        stream=3,
        task=2,
        DataLoaderClass=MfccWithDeltas,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S3T2_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=191,
    ),
    HParams(
        results_key="S3/T2/ModifiedCPC/Simple4",
        checkpoint_key="S3/T2/ModifiedCPC/Simple4",
        tensorboard_key="S3/T2/ModifiedCPC/Simple4",
        cache_key="S3/T2/ModifiedCPC",
        stream=3,
        task=2,
        DataLoaderClass=ModifiedCPC,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S3T2_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=152,
    ),
    HParams(
        results_key="S3/T2/UnispeechSAT/Simple4",
        checkpoint_key="S3/T2/UnispeechSAT/Simple4",
        tensorboard_key="S3/T2/UnispeechSAT/Simple4",
        cache_key="S3/T2/UnispeechSAT",
        stream=3,
        task=2,
        DataLoaderClass=UnispeechSAT,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S3T2_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=200,
    ),
    HParams(
        results_key="S3/T2/Wav2Vec/Simple4",
        checkpoint_key="S3/T2/Wav2Vec/Simple4",
        tensorboard_key="S3/T2/Wav2Vec/Simple4",
        cache_key="S3/T2/Wav2Vec",
        stream=3,
        task=2,
        DataLoaderClass=Wav2Vec,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S3T2_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=4,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=5,
    ),
]

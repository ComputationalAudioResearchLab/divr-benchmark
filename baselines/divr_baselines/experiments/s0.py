import torch
from torch import nn
from ..model import Simple
from torch.optim import Adam
from typing import Type
from ..data_loader import (
    Data2Vec,
    Mfcc,
    MfccWithDeltas,
    ModifiedCPC,
    UnispeechSAT,
    Wav2Vec,
    LoaderTypes,
)
from ..trainer import HParams
from .device import device

## class weights are derived from train set as that's what is used for training
S0_class_weights = 22805 / torch.LongTensor([9489, 13316])
"""
 [normal, pathological]
"""


def gen_params(
    results_key: str,
    checkpoint_key: str,
    cache_key: str,
    DataLoaderClass: Type,
    task: int,
    batch_size: int,
    best_checkpoint_epoch: int | None = None,
):
    return HParams(
        results_key=results_key,
        checkpoint_key=checkpoint_key,
        tensorboard_key=checkpoint_key,
        cache_key=cache_key,
        stream=0,
        task=task,
        DataLoaderClass=DataLoaderClass,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S0_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=batch_size,
        device=device,
        loader_type=LoaderTypes.NORMAL,
        best_checkpoint_epoch=best_checkpoint_epoch,
    )


s0_experiments = [
    HParams(
        results_key="S0/Mfcc/Simple",
        checkpoint_key="S0/Mfcc/Simple",
        tensorboard_key="S0/Mfcc/Simple",
        cache_key="S0/Mfcc",
        stream=0,
        task=1,
        DataLoaderClass=Mfcc,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S0_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=8,
        device=device,
    ),
    HParams(
        results_key="S0/ModifiedCPC/Simple",
        checkpoint_key="S0/ModifiedCPC/Simple",
        tensorboard_key="S0/ModifiedCPC/Simple",
        cache_key="S0/ModifiedCPC",
        stream=0,
        task=1,
        DataLoaderClass=ModifiedCPC,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S0_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=8,
        device=device,
    ),
    HParams(
        results_key="S0/Wav2Vec/Simple",
        checkpoint_key="S0/Wav2Vec/Simple",
        tensorboard_key="S0/Wav2Vec/Simple",
        cache_key="S0/Wav2Vec",
        stream=0,
        task=1,
        DataLoaderClass=Wav2Vec,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S0_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=2,
        device=device,
        loader_type=LoaderTypes.BATCH_AHEAD,
    ),
    *[
        gen_params(
            results_key=f"S0/T{t}/Data2Vec/Simple",
            checkpoint_key="S0/Data2Vec/Simple",
            cache_key="S0/Data2Vec",
            task=t,
            DataLoaderClass=Data2Vec,
            batch_size=1,
            best_checkpoint_epoch=67,
        )
        for t in range(1, 5)
    ],
    *[
        gen_params(
            results_key=f"S0/T{t}/Mfcc/Simple2",
            checkpoint_key="S0/Mfcc/Simple2",
            cache_key="S0/Mfcc",
            task=t,
            DataLoaderClass=Mfcc,
            batch_size=2,
            best_checkpoint_epoch=107,
        )
        for t in range(1, 5)
    ],
    *[
        gen_params(
            results_key=f"S0/T{t}/MfccWithDeltas/Simple2",
            checkpoint_key="S0/MfccWithDeltas/Simple2",
            cache_key="S0/MfccWithDeltas",
            task=t,
            batch_size=2,
            DataLoaderClass=MfccWithDeltas,
            best_checkpoint_epoch=57,
        )
        for t in range(1, 5)
    ],
    *[
        gen_params(
            results_key=f"S0/T{t}/ModifiedCPC/Simple2",
            checkpoint_key="S0/ModifiedCPC/Simple2",
            cache_key="S0/ModifiedCPC",
            task=t,
            DataLoaderClass=ModifiedCPC,
            batch_size=2,
            best_checkpoint_epoch=122,
        )
        for t in range(1, 5)
    ],
    *[
        gen_params(
            results_key=f"S0/T{t}/UnispeechSAT/Simple",
            checkpoint_key="S0/UnispeechSAT/Simple",
            cache_key="S0/UnispeechSAT",
            task=t,
            DataLoaderClass=UnispeechSAT,
            batch_size=1,
            best_checkpoint_epoch=104,
        )
        for t in range(1, 5)
    ],
    *[
        gen_params(
            results_key=f"S0/T{t}/Wav2Vec/Simple2",
            checkpoint_key="S0/Wav2Vec/Simple2",
            cache_key="S0/Wav2Vec",
            task=t,
            DataLoaderClass=Wav2Vec,
            batch_size=2,
            best_checkpoint_epoch=39,
        )
        for t in range(1, 5)
    ],
]

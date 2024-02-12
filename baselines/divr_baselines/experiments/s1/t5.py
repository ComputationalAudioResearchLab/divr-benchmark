import torch
from torch import nn
from ...model import Simple
from torch.optim import Adam
from ...data_loader import ModifiedCPC
from ...trainer import HParams
from ..device import device

## class weights are derived from train set as that's what is used for training
class_weights = 1172 / torch.LongTensor([132, 129, 494, 417])
"""
 [functional, muscle_tension, normal, organic]
"""
task = 5

t5_experiments = [
    HParams(
        experiment_key=f"S1/T{task}/ModifiedCPC/Simple",
        cache_key=f"S1/T{task}/ModifiedCPC",
        stream=1,
        task=task,
        DataLoaderClass=ModifiedCPC,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=32,
        device=device,
    ),
]

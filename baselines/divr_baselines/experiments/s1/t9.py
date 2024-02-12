import torch
from torch import nn
from ...model import Simple
from torch.optim import Adam
from ...data_loader import Data2Vec
from ...trainer import HParams
from ..device import device

## class weights are derived from train set as that's what is used for training
S1T9_class_weights = 1172 / torch.LongTensor([132, 129, 494, 68, 208, 137, 4])
"""
 [functional, muscle_tension, normal, organic_inflammatory, organic_neuro_muscular, organic_structural, organic_trauma] = [132, 129, 494, 68, 211, 137, 4]
"""

t9_experiments = [
    ## S1/T9
    HParams(
        experiment_key="S1/T9/Data2Vec/Simple",
        cache_key="S1/T9/Data2Vec",
        stream=1,
        task=9,
        DataLoaderClass=Data2Vec,
        ModelClass=Simple,
        criterion=nn.CrossEntropyLoss(weight=S1T9_class_weights.to(device)),
        OptimClass=Adam,
        lr=1e-5,
        batch_size=32,
        device=device,
    ),
]

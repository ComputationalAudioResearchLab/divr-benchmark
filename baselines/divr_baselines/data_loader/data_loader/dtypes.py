from __future__ import annotations
import h5py
import torch
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


AudioBatch = List[List[np.ndarray]]
"""
AudioBatch = collection of audios in a given batch

[
    [
        session_audio_1,
        session_audio_2,
        session_audio_3,
        ... for session
    ]
    ... for batch
]
"""

InputArrays = Tuple[np.ndarray, np.ndarray]
"""
InputArrays[0]
    - contains audio data
    - shape = [B, F, S]
---
InputArrays[1]
    - contains length of each audio in batch
    - shape = [B, F]

---
- B = Batch size
- F = Number of files in a session
- S = Sequence length
---
"""

InputTensors = Tuple[torch.Tensor, torch.Tensor]
"""
InputTensors[0]
    - contains feature data
    - type = torch.FloatTensor
    - shape = [B, F, S, H]
---
InputTensors[1]
    - contains length of each audio in batch
    - type = torch.LongTensor
    - shape = [B, F]

---
- B = Batch size
- F = Number of files in a session
- S = Sequence length
- H = Feature length
---
"""

LabelTensor = torch.Tensor
"""
LabelTensor
    - type = torch.LongTensor
    - shape = [B]
---
 - B = Batch size
"""


@dataclass
class CacheSet:
    inputs: h5py.Dataset
    shapes: h5py.Dataset
    labels: h5py.Dataset

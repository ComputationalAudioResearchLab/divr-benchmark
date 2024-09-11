from __future__ import annotations
import torch
import numpy as np
from typing import Tuple, List


AudioBatch = List[np.ndarray]
"""
AudioBatch = collection of audios in a given batch
[
    audio_1,
    audio_2,
    audio_3,
    ... for batch
]
"""

InputArrays = Tuple[np.ndarray, np.ndarray]
"""
InputArrays[0]
    - contains audio data
    - shape = [B, S]
---
InputArrays[1]
    - contains length of each audio in batch
    - shape = [B]

---
- B = Batch size
- S = Sequence length
---
"""

InputTensors = Tuple[torch.Tensor, torch.Tensor]
"""
InputTensors[0]
    - contains feature data
    - type = torch.FloatTensor
    - shape = [B, S, H]
---
InputTensors[1]
    - contains length of each audio in batch
    - type = torch.LongTensor
    - shape = [B]

---
- B = Batch size
- S = Sequence length
- H = Feature length
---
"""

DataPoint = Tuple[InputTensors, torch.LongTensor]
"""
Single data point consisting of input tensors and labels
"""

Dataset = List[Tuple[int | None, str, str, str]]
"""
Dataset defined as list of rows where each row has the structure:
    1. age = None or int
    2. gender = str
    3. label = str (pathology or normal)
    4. audio file path
"""

Database = Tuple[Dataset, Dataset, Dataset]
"""
Database consisting of training, evaluation and testing dataset
"""

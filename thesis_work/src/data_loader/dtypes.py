from __future__ import annotations
import torch
import numpy as np
from typing import List, Tuple


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
    - shape = [B, L, S]
---
InputArrays[1]
    - contains length of each audio in batch
    - shape = [B, L]

---
- B = Batch size
- L = Number of samples / audio files
- S = Sequence length
---
"""

InputTensors = Tuple[torch.Tensor, torch.Tensor]
"""
InputTensors[0]
    - contains feature data
    - type = torch.FloatTensor
    - shape = [B, L, S, H]
---
InputTensors[1]
    - contains length of each audio in batch
    - type = torch.LongTensor
    - shape = [B, L]

---
- B = Batch size
- L = Number of samples / audio files
- S = Sequence length
- H = Feature length
---
"""

LabelTensor = torch.Tensor
"""
LabelTensor
    - type = torch.LongTensor
    - shape = [B, L]
---
 - B = Batch size
 - L = Number of diagnosis levels
"""

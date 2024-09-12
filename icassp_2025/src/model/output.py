import torch
from torch import nn
from pathlib import Path

from ..data_loader import InputTensors
from .savable_module import SavableModule

class Normalized(SavableModule):
    def __init__(self, input_size: int, num_classes: int, checkpoint_path: Path):
        super().__init__(checkpoint_path)
        hidden_size = 1024
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(True),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_classes),
            nn.Softmax(dim=2),
        )

    def forward(self, inputs: InputTensors) -> InputTensors:
        input_audios, input_lens = inputs
        per_frame_labels = self.model(input_audios)
        per_frame_labels = self.__mask(per_frame_labels, input_lens)
        per_audio_labels = per_frame_labels.sum(dim=1) / input_lens.unsqueeze(1)
        return per_audio_labels, per_frame_labels

    def __mask(
        self, per_frame_labels: torch.Tensor, input_lens: torch.Tensor
    ) -> torch.Tensor:
        max_len = int(input_lens.max().item())
        (batch_size,) = input_lens.shape
        mask = torch.arange(max_len, device=input_lens.device).expand(
            batch_size, max_len
        ) < input_lens.unsqueeze(1)
        return per_frame_labels * mask.unsqueeze(2)

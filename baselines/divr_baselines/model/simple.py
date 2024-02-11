from .savable_module import SavableModule
from ..data_loader import InputTensors
import torch
from torch import nn
from pathlib import Path


class Simple(SavableModule):
    def __init__(self, input_size: int, num_classes: int, checkpoint_path: Path):
        super().__init__(checkpoint_path)
        hidden_size = 1024
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, num_classes),
            nn.Softmax(dim=3),
        )
        self.init_orthogonal_weights()

    def forward(self, inputs: InputTensors) -> torch.Tensor:
        input_audios, input_lens = inputs
        per_frame_predicted_labels = self.model(input_audios)
        predicted_labels = per_frame_predicted_labels.mean(dim=(2, 1))
        return predicted_labels

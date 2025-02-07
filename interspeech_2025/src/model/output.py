import torch
from torch import nn
from pathlib import Path
from typing import Dict, List
import torch.nn.functional as F

from ..data_loader import InputTensors
from .savable_module import SavableModule


class Base(SavableModule):

    def process_per_frame_labels(self, input_lens, per_frame_labels):
        audios_per_session = input_lens.count_nonzero(dim=1)
        per_frame_labels = self.__mask_frames(per_frame_labels, input_lens)
        per_audio_labels = per_frame_labels.sum(dim=2)
        per_audio_labels = self.__masked_divide(per_audio_labels, input_lens)
        per_session_labels = per_audio_labels.sum(dim=1)
        per_session_labels /= audios_per_session.unsqueeze(1)
        return per_session_labels, per_audio_labels, per_frame_labels

    def __masked_divide(
        self, per_audio_labels: torch.Tensor, input_lens: torch.Tensor
    ) -> torch.Tensor:
        audios_per_session = input_lens.count_nonzero(dim=1)
        (batch_size, max_audios, num_classes) = per_audio_labels.shape
        mask = torch.arange(max_audios, device=audios_per_session.device).expand(
            batch_size, max_audios
        ) < audios_per_session.unsqueeze(1)
        # By applying the mask we can ensure that anything with 0 length
        # is already zero
        per_audio_labels = per_audio_labels * mask.unsqueeze(2)
        per_audio_labels /= input_lens.clamp(min=1e-8).unsqueeze(2)
        return per_audio_labels

    def __mask_frames(
        self, per_frame_labels: torch.Tensor, input_lens: torch.Tensor
    ) -> torch.Tensor:
        max_len = int(input_lens.max().item())
        (batch_size, max_audios) = input_lens.shape
        mask = torch.arange(max_len, device=input_lens.device).expand(
            batch_size, max_audios, max_len
        ) < input_lens.unsqueeze(2)
        return per_frame_labels * mask.unsqueeze(3)


class Normalized(Base):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        checkpoint_path: Path,
    ):
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
            nn.Softmax(dim=3),
        )
        self.init_orthogonal_weights()

    def forward(self, inputs: InputTensors):
        input_audios, input_lens = inputs
        per_frame_labels = self.model(input_audios)
        return self.process_per_frame_labels(input_lens, per_frame_labels)


class NormalizedMultitask(Base):

    def __init__(
        self,
        input_size: int,
        num_classes: Dict[int, int],
        checkpoint_path: Path,
    ):
        super().__init__(checkpoint_path)
        hidden_size = 1024
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(True),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.LayerNorm(hidden_size),
        )
        self.readout_layers = nn.ModuleList(
            [nn.Linear(hidden_size, c) for c in num_classes.values()]
        )
        self.init_orthogonal_weights()
        self.levels = len(self.readout_layers)

    def forward(self, inputs: InputTensors):
        input_audios, input_lens = inputs
        per_frame_latents = self.model(input_audios)
        results = []
        for readout_layer in self.readout_layers:
            per_frame_labels = F.softmax(readout_layer(per_frame_latents), dim=3)
            results += [self.process_per_frame_labels(input_lens, per_frame_labels)]
        return results


class NormalizedMultiCrit(Base):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        checkpoint_path: Path,
        levels_map: Dict[int, List[List[int]]],
    ):
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
            nn.Softmax(dim=3),
        )
        self.init_orthogonal_weights()
        self.__levels_map = levels_map

    def forward(self, inputs: InputTensors):
        input_audios, input_lens = inputs
        per_frame_labels = self.model(input_audios)
        return self.process_per_frame_labels(input_lens, per_frame_labels)

    def labels_at_level(self, labels: torch.Tensor, level: int) -> torch.Tensor:
        level_map = self.__levels_map[level]
        new_labels = []
        for combinations in level_map:
            new_labels += [labels[:, combinations].sum(dim=1)]
        new_labels = torch.stack(new_labels, dim=1)
        return new_labels

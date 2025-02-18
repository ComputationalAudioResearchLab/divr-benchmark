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

    def set_levels_map(self, levels_map: Dict[int, List[List[int]]]):
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


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.eps = nn.Parameter(
            torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32)),
            requires_grad=False,
        )

    def forward(self, src, attn_mask, key_padding_mask):
        batch_size = src.size(0)

        # Linear projections
        query = (
            self.query(src).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        )
        key = self.key(src).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        value = (
            self.value(src).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        )

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.eps
        scores = scores.masked_fill(attn_mask, float("-inf"))
        scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        zero_length_mask = key_padding_mask.all(dim=1, keepdim=True)[:, None, None, :]
        scores = scores.masked_fill(zero_length_mask, 0.0)
        attn = F.softmax(scores, dim=-1)
        attn = self.attention_dropout(attn)

        # Combine heads
        context = (
            torch.matmul(attn, value)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.d_model)
        )
        output = self.out(context)

        return output, attn


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, nhead, dropout)
        self.mapper = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, attn = self.self_attn(
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        src = self.norm1(src + self.dropout(src2))
        src = src + self.mapper(src)
        src = self.norm2(src)
        return src, attn


class SimpleTransformer(Base):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        checkpoint_path: Path,
    ):
        super().__init__(checkpoint_path)
        hidden_size = 512
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.out_projection = nn.Linear(hidden_size, num_classes)
        num_layers = 6
        self.encoder = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=4,
                    dropout=0.1,
                )
                for _ in range(num_layers)
            ]
        )
        self.num_classes = num_classes
        self.hidden_size = hidden_size

    def forward(self, inputs: InputTensors):
        input_audios, input_lens = inputs
        batch_size, num_audios, seq_length, feature_size = input_audios.shape
        total_batch_suze = batch_size * num_audios

        per_frame_labels = self.model(
            input_audios=input_audios.view(total_batch_suze, seq_length, feature_size),
            input_lens=input_lens.view(total_batch_suze),
        ).view(
            batch_size,
            num_audios,
            seq_length,
            self.num_classes,
        )

        return self.process_per_frame_labels(
            input_lens=input_lens,
            per_frame_labels=per_frame_labels,
        )

    def model(
        self, input_audios: torch.Tensor, input_lens: torch.Tensor
    ) -> torch.Tensor:
        """
        input_audios: [Batch Size, Seq Len, Feature Size]
        input_lens: [Batch Size]
        """
        batch_size, seq_len, feature_size = input_audios.shape
        src_key_padding_mask = self.src_padding_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            input_lens=input_lens,
        )

        causal_mask = self.causal_mask(seq_len=seq_len)
        pos_enc = self.positional_encoding(seq_len=seq_len)

        X = self.input_projection(input_audios)
        X = X + pos_enc
        for encoder in self.encoder:
            X, attn = encoder(
                X,
                src_mask=causal_mask,
                src_key_padding_mask=src_key_padding_mask,
            )

        Y = self.out_projection(X)
        Y = F.softmax(Y, dim=2)

        return Y

    def src_padding_mask(
        self, batch_size: int, seq_len: int, input_lens: torch.Tensor
    ) -> torch.Tensor:
        mask = torch.arange(seq_len, device=self.device).expand(batch_size, seq_len)
        return mask >= input_lens.unsqueeze(1)

    def causal_mask(self, seq_len: int) -> torch.Tensor:
        return (
            torch.triu(torch.ones(seq_len, seq_len, device=self.device), diagonal=1)
            == 1
        )

    def positional_encoding(self, seq_len: int):
        position = torch.arange(
            seq_len, dtype=torch.float, device=self.device
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.hidden_size, 2, device=self.device).float()
            * (-torch.log(torch.tensor(10000.0)) / self.hidden_size)
        )
        pe = torch.zeros(seq_len, self.hidden_size, device=self.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Add batch dimension

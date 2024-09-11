import torch
import torchaudio
from torch import nn
from s3prl.nn import S3PRLUpstream

from ..data_loader import InputTensors

class Feature(nn.Module):
    model_name: str
    device: torch.device
    feature_size: int

class S3PRLFrozen(Feature):

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.model = S3PRLUpstream(self.model_name).eval().to(self.device)

    @torch.no_grad()
    def forward(self, batch: InputTensors) -> InputTensors:
        batch_inputs, batch_lens = batch
        audios = batch_inputs
        audio_lens = batch_lens
        if not isinstance(audios, torch.Tensor):
            audios = torch.tensor(audios, device=self.device, dtype=torch.float32)
        elif audios.device != self.device:
            audios = audios.to(self.device)
        if not isinstance(audio_lens, torch.Tensor):
            audio_lens = torch.tensor(audio_lens, device=self.device, dtype=torch.long)
        elif audio_lens.device != self.device:
            audio_lens = audio_lens.to(self.device)
        all_hs, all_hs_len = self.model(audios, audio_lens)
        feature = torch.cat(all_hs, dim=2)
        feature_lens = all_hs_len[0]
        return feature, feature_lens
    
    
class Data2Vec(S3PRLFrozen):
    model_name = "data2vec_large_ll60k"
    feature_size = 25600


class Wav2Vec(S3PRLFrozen):
    model_name = "wav2vec_large"
    feature_size = 6656


class UnispeechSAT(S3PRLFrozen):
    model_name = "unispeech_sat_large"
    feature_size = 25600


class ModifiedCPC(S3PRLFrozen):
    model_name = "modified_cpc"
    feature_size = 512

class MFCC(Feature):
    model_name = "mfcc_64"
    feature_size = 64
    sample_rate = 16000
    n_fft = 1024
    win_length = n_fft
    hop_length = win_length // 4
    n_mels = feature_size*4

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.feature_size,
            melkwargs={
                "n_fft": self.n_fft,
                "n_mels": self.n_mels,
                "win_length": self.win_length,
                "hop_length": self.hop_length,
            },
        ).to(device)

    @torch.no_grad()
    def forward(self, batch: InputTensors) -> InputTensors:
        batch_inputs, batch_lens = batch
        audios = batch_inputs
        audio_lens = batch_lens
        if not isinstance(audios, torch.Tensor):
            audios = torch.tensor(audios, device=self.device, dtype=torch.float32)
        elif audios.device != self.device:
            audios = audios.to(self.device)
        if not isinstance(audio_lens, torch.Tensor):
            audio_lens = torch.tensor(audio_lens, device=self.device, dtype=torch.long)
        elif audio_lens.device != self.device:
            audio_lens = audio_lens.to(self.device)
        mfcc = self.mfcc(audios).transpose(1, 2)  # move MFCC to the last dimension
        mfcc_lens = self.calc_lengths(audio_lens)
        return mfcc, mfcc_lens

    def calc_lengths(self, audio_lens: torch.Tensor):
        return (
            audio_lens + self.hop_length - audio_lens % self.hop_length
        ) // self.hop_length
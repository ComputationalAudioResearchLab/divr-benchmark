import torch
import opensmile
import torchaudio
import numpy as np
from torch import nn
from s3prl.nn import S3PRLUpstream
from speechbrain.inference.encoders import MelSpectrogramEncoder
from speechbrain.inference.speaker import EncoderClassifier

from ..data_loader import InputTensors


class Feature(nn.Module):
    model_name: str
    __device: torch.device
    feature_size: int

    def individual_np(self, audio: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class S3PRLFrozen(Feature):

    def __init__(self, device: torch.device, sampling_rate: int) -> None:
        super().__init__()
        self.__device = device
        self.model = S3PRLUpstream(self.model_name).eval().to(self.__device)

    @torch.no_grad()
    def individual_np(self, audio: np.ndarray) -> np.ndarray:
        audio_tensor = torch.tensor([audio], dtype=torch.float, device=self.__device)
        audio_lens = torch.tensor([len(audio)], dtype=torch.long, device=self.__device)
        all_hs, _ = self.model(audio_tensor, audio_lens)
        return torch.cat(all_hs, dim=2)[0].cpu().numpy()

    @torch.no_grad()
    def forward(self, batch: InputTensors) -> InputTensors:
        batch_inputs, batch_lens = batch
        batch_size, max_audios_in_session, max_audio_len = batch_inputs.shape
        audios = batch_inputs.reshape(
            batch_size * max_audios_in_session,
            max_audio_len,
        )
        audio_lens = batch_lens.reshape(batch_size * max_audios_in_session)
        if not isinstance(audios, torch.Tensor):
            audios = torch.tensor(
                audios,
                device=self.__device,
                dtype=torch.float32,
            )
        elif audios.device != self.__device:
            audios = audios.to(self.__device)
        if not isinstance(audio_lens, torch.Tensor):
            audio_lens = torch.tensor(
                audio_lens,
                device=self.__device,
                dtype=torch.long,
            )
        elif audio_lens.device != self.__device:
            audio_lens = audio_lens.to(self.__device)
        all_hs, all_hs_len = self.model(audios, audio_lens)
        feature = torch.cat(all_hs, dim=2)
        _, max_feature_len, feature_hidden_len = feature.shape
        feature = feature.reshape(
            (
                batch_size,
                max_audios_in_session,
                max_feature_len,
                feature_hidden_len,
            )
        )
        feature_lens = all_hs_len[0].reshape(
            (
                batch_size,
                max_audios_in_session,
            )
        )
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


class OpenSmile(Feature):
    feature_set: opensmile.FeatureSet
    feature_level: opensmile.FeatureLevel
    window = 0.06  # 60ms
    hop_size = 0.01  # 10ms

    def __init__(self, device: torch.device, sampling_rate: int) -> None:
        super().__init__()
        self.__device = device
        self.__sampling_rate = sampling_rate
        self.__smile = opensmile.Smile(
            feature_set=self.feature_set,
            feature_level=self.feature_level,
        )
        self.__zero_feature = np.zeros(self.feature_size)
        self.__win = sampling_rate * self.window
        self.__hop = sampling_rate * self.hop_size

    @torch.no_grad()
    def individual_np(self, audio: np.ndarray) -> np.ndarray:
        return self.__smile.process_signal(
            signal=audio,
            sampling_rate=self.__sampling_rate,
        ).to_numpy()

    @torch.no_grad()
    def forward(self, batch: InputTensors) -> InputTensors:
        batch_inputs, batch_lens = batch
        audios = batch_inputs
        audio_lens = batch_lens
        if isinstance(audios, torch.Tensor):
            audios = audios.cpu().numpy()
        batch_size, max_sessions, max_audio_len = audios.shape
        audios = audios.reshape(batch_size * max_sessions, max_audio_len)
        audio_lens = audio_lens.reshape(batch_size * max_sessions)

        # this takes into account both lld and lld deltas with a few zero pads
        # at the end we cut this using the actual feature len later on
        expected_feature_len = int((max_audio_len - self.__win) // self.__hop) + 4
        features = np.zeros(
            (
                batch_size * max_sessions,
                expected_feature_len,
                self.feature_size,
            )
        )
        feature_lens = []
        max_feature_len = 0
        for idx, (audio, audio_len) in enumerate(zip(audios, audio_lens)):
            if audio_len > 0:
                feature = self.__smile.process_signal(
                    signal=audio[:audio_len],
                    sampling_rate=self.__sampling_rate,
                )
                feature_len = feature.shape[0]
            else:
                feature = self.__zero_feature
                feature_len = 0
            features[idx, :feature_len] = feature
            feature_lens += [feature_len]
            if feature_len > max_feature_len:
                max_feature_len = feature_len
        features = features.reshape(
            batch_size,
            max_sessions,
            expected_feature_len,
            self.feature_size,
        )
        feature_lens = np.array(feature_lens).reshape(batch_size, max_sessions)
        features = torch.tensor(
            features[:, :, :max_feature_len],
            device=self.__device,
            dtype=torch.float32,
        )
        feature_lens = torch.tensor(
            feature_lens,
            device=self.__device,
            dtype=torch.long,
        )
        return features, feature_lens


class Compare2016Functional(OpenSmile):
    model_name = "ComParE_2016_func"
    feature_size = 6373
    feature_set = opensmile.FeatureSet.ComParE_2016
    feature_level = opensmile.FeatureLevel.Functionals


class Compare2016LLD(OpenSmile):
    model_name = "ComParE_2016_lld"
    feature_size = 65
    feature_set = opensmile.FeatureSet.ComParE_2016
    feature_level = opensmile.FeatureLevel.LowLevelDescriptors


class Compare2016LLDDE(OpenSmile):
    model_name = "ComParE_2016_lld_de"
    feature_size = 65
    feature_set = opensmile.FeatureSet.ComParE_2016
    feature_level = opensmile.FeatureLevel.LowLevelDescriptors_Deltas


class EGEMapsv2Functional(OpenSmile):
    model_name = "eGeMAPSv02_func"
    feature_size = 88
    feature_set = opensmile.FeatureSet.eGeMAPSv02
    feature_level = opensmile.FeatureLevel.Functionals


class EGEMapsv2LLD(OpenSmile):
    model_name = "eGeMAPSv02_func"
    feature_size = 25
    feature_set = opensmile.FeatureSet.eGeMAPSv02
    feature_level = opensmile.FeatureLevel.LowLevelDescriptors


class MFCCDD(Feature):
    model_name = "mfcc_dd_13"
    n_mfcc = 13
    feature_size = n_mfcc * 3
    n_fft = 1024
    win_length = n_fft
    hop_length = win_length // 4
    n_mels = n_mfcc * 4

    def __init__(self, device: torch.device, sampling_rate: int) -> None:
        super().__init__()
        self.__device = device
        self.__sample_rate = sampling_rate
        self.__mfcc = torchaudio.transforms.MFCC(
            sample_rate=self.__sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                "n_fft": self.n_fft,
                "n_mels": self.n_mels,
                "win_length": self.win_length,
                "hop_length": self.hop_length,
            },
        ).to(device)

    @torch.no_grad()
    def individual_np(self, audio: np.ndarray) -> np.ndarray:
        audio_tensor = torch.tensor([audio], dtype=torch.float, device=self.__device)
        mfcc = self.__mfcc(audio_tensor)
        deltas = torchaudio.functional.compute_deltas(mfcc)
        double_deltas = torchaudio.functional.compute_deltas(mfcc)
        mfccdd = torch.cat([mfcc, deltas, double_deltas], dim=-2)
        return mfccdd[0].T.cpu().numpy()

    @torch.no_grad()
    def forward(self, batch: InputTensors) -> InputTensors:
        batch_inputs, batch_lens = batch
        audios = batch_inputs
        audio_lens = batch_lens
        if not isinstance(audios, torch.Tensor):
            audios = torch.tensor(
                audios,
                device=self.__device,
                dtype=torch.float32,
            )
        elif audios.device != self.__device:
            audios = audios.to(self.__device)
        if not isinstance(audio_lens, torch.Tensor):
            audio_lens = torch.tensor(
                audio_lens,
                device=self.__device,
                dtype=torch.long,
            )
        elif audio_lens.device != self.__device:
            audio_lens = audio_lens.to(self.__device)
        # move MFCC to the last dimension
        mfcc = self.__mfcc(audios)
        deltas = torchaudio.functional.compute_deltas(mfcc)
        double_deltas = torchaudio.functional.compute_deltas(mfcc)
        mfccdd = torch.cat([mfcc, deltas, double_deltas], dim=-2)
        mfccdd = mfccdd.transpose(2, 3)
        mfcc_lens = self.calc_lengths(audio_lens)
        return mfccdd, mfcc_lens

    def calc_lengths(self, audio_lens: torch.Tensor):
        return (
            audio_lens + self.hop_length - audio_lens % self.hop_length
        ) // self.hop_length


class SpeechBrainFrozen(Feature):

    def __init__(self, device: torch.device, sampling_rate: int) -> None:
        super().__init__()
        self.__device = device
        self.model = EncoderClassifier.from_hparams(
            source=self.model_name, run_opts={"device": device}
        ).eval()
        self._device = device

    @torch.no_grad()
    def individual_np(self, audio: np.ndarray) -> np.ndarray:
        audio_tensor = torch.tensor([audio], dtype=torch.float, device=self.__device)
        return self.model.encode_batch(audio_tensor)[0].cpu().numpy()

    @torch.no_grad()
    def forward(self, batch: InputTensors) -> InputTensors:
        raise NotImplementedError()
        # batch_inputs, batch_lens = batch
        # batch_size, max_audios_in_session, max_audio_len = batch_inputs.shape
        # audios = batch_inputs.reshape(
        #     batch_size * max_audios_in_session,
        #     max_audio_len,
        # )
        # audio_lens = batch_lens.reshape(batch_size * max_audios_in_session)
        # if not isinstance(audios, torch.Tensor):
        #     audios = torch.tensor(
        #         audios,
        #         device=self.__device,
        #         dtype=torch.float32,
        #     )
        # elif audios.device != self.__device:
        #     audios = audios.to(self.__device)
        # if not isinstance(audio_lens, torch.Tensor):
        #     audio_lens = torch.tensor(
        #         audio_lens,
        #         device=self.__device,
        #         dtype=torch.long,
        #     )
        # elif audio_lens.device != self.__device:
        #     audio_lens = audio_lens.to(self.__device)
        # all_hs, all_hs_len = self.model(audios, audio_lens)
        # feature = torch.cat(all_hs, dim=2)
        # _, max_feature_len, feature_hidden_len = feature.shape
        # feature = feature.reshape(
        #     (
        #         batch_size,
        #         max_audios_in_session,
        #         max_feature_len,
        #         feature_hidden_len,
        #     )
        # )
        # feature_lens = all_hs_len[0].reshape(
        #     (
        #         batch_size,
        #         max_audios_in_session,
        #     )
        # )
        # return feature, feature_lens


class ECAPA_VoxCeleb(SpeechBrainFrozen):
    model_name = "speechbrain/spkrec-ecapa-voxceleb"
    feature_size = 192


class ECAPA_VoxCeleb_MelSpec(Feature):
    model_name = "speechbrain/spkrec-ecapa-voxceleb-mel-spec"
    feature_size = 192

    def __init__(self, device: torch.device, sampling_rate: int) -> None:
        super().__init__()
        self.__device = device
        self.model = MelSpectrogramEncoder.from_hparams(
            source=self.model_name, run_opts={"device": device}
        ).eval()
        self._device = device

    @torch.no_grad()
    def individual_np(self, audio: np.ndarray) -> np.ndarray:
        audio_tensor = torch.tensor([audio], dtype=torch.float, device=self._device)
        feature = self.model.encode_waveform(audio_tensor)
        return feature[0].cpu().numpy()


class ResNet_VoxCeleb(SpeechBrainFrozen):
    model_name = "speechbrain/spkrec-resnet-voxceleb"
    feature_size = 256

    @torch.no_grad()
    def individual_np(self, audio: np.ndarray) -> np.ndarray:
        audio_tensor = torch.tensor([audio], dtype=torch.float, device=self._device)
        feature = self.model.encode_batch(audio_tensor)
        return feature.cpu().numpy()


class XVect_VoxCeleb(SpeechBrainFrozen):
    model_name = "speechbrain/spkrec-xvect-voxceleb"
    feature_size = 512

import librosa
import numpy as np
import torch
from .data_loader import DataLoader, InputArrays, InputTensors


class Mfcc(DataLoader):
    n_mfcc = 13
    win_length = 1024
    hop_length = 256
    n_fft = 1024
    window = "hann"
    center = False
    power = 2
    feature_size = n_mfcc

    def feature_function(self, batch: InputArrays) -> InputTensors:
        batch_size, max_audios_in_session, max_audio_len = batch[0].shape
        max_mfcc_len = self.__to_mfcc_len(max_audio_len)
        mfccs = np.zeros((batch_size, max_audios_in_session, max_mfcc_len, self.n_mfcc))
        mfcc_lens = np.zeros((batch_size, max_audios_in_session), dtype=int)
        for batch_idx, (audio_files, audio_lens) in enumerate(zip(batch[0], batch[1])):
            for session_idx, (audio, audio_len) in enumerate(
                zip(audio_files, audio_lens)
            ):
                mfcc = librosa.feature.mfcc(
                    y=audio[:audio_len],
                    sr=self.audio_sample_rate,
                    n_mfcc=self.n_mfcc,
                    win_length=self.win_length,
                    hop_length=self.hop_length,
                    n_fft=self.n_fft,
                    window=self.window,
                    center=self.center,
                    power=self.power,
                ).T
                mfcc_len = mfcc.shape[0]
                mfccs[batch_idx, session_idx, :mfcc_len] = mfcc
                mfcc_lens[batch_idx, session_idx] = mfcc_len
        mfccs = torch.FloatTensor(mfccs).to(self.device)
        mfcc_lens = torch.LongTensor(mfcc_lens).to(self.device)
        return (mfccs, mfcc_lens)

    def __to_mfcc_len(self, audio_len: int) -> int:
        return (audio_len - self.win_length) // self.hop_length + 1


class MfccWithDeltas(DataLoader):
    n_mfcc = 13
    win_length = 1024
    hop_length = 256
    n_fft = 1024
    window = "hann"
    center = False
    power = 2
    feature_size = n_mfcc * 3

    def feature_function(self, batch: InputArrays) -> InputTensors:
        batch_size, max_audios_in_session, max_audio_len = batch[0].shape
        max_mfcc_len = self.__to_mfcc_len(max_audio_len)
        mfccs = np.zeros(
            (batch_size, max_audios_in_session, max_mfcc_len, self.feature_size)
        )
        mfcc_lens = np.zeros((batch_size, max_audios_in_session), dtype=int)
        for batch_idx, (audio_files, audio_lens) in enumerate(zip(batch[0], batch[1])):
            for session_idx, (audio, audio_len) in enumerate(
                zip(audio_files, audio_lens)
            ):
                mfcc = librosa.feature.mfcc(
                    y=audio[:audio_len],
                    sr=self.audio_sample_rate,
                    n_mfcc=self.n_mfcc,
                    win_length=self.win_length,
                    hop_length=self.hop_length,
                    n_fft=self.n_fft,
                    window=self.window,
                    center=self.center,
                    power=self.power,
                )
                mfcc_delta = self.__delta(mfcc=mfcc, order=1)
                mfcc_delta_2 = self.__delta(mfcc=mfcc, order=2)
                mfcc_data = np.concatenate([mfcc, mfcc_delta, mfcc_delta_2], axis=0).T
                mfcc_len = mfcc_data.shape[0]
                mfccs[batch_idx, session_idx, :mfcc_len] = mfcc_data
                mfcc_lens[batch_idx, session_idx] = mfcc_len
        mfccs = torch.FloatTensor(mfccs).to(self.device)
        mfcc_lens = torch.LongTensor(mfcc_lens).to(self.device)
        return (mfccs, mfcc_lens)

    def __to_mfcc_len(self, audio_len: int) -> int:
        return (audio_len - self.win_length) // self.hop_length + 1

    def __delta(self, mfcc, order):
        num_frames = mfcc.shape[1]
        if num_frames > 9:
            width = 9
        elif num_frames > 6:
            width = 7
        elif num_frames > 3:
            width = 5
        elif num_frames > 2:
            width = 3
        else:
            raise ValueError("Cannot run delta with 2 or fewer frames")

        return librosa.feature.delta(data=mfcc, width=width, order=order, mode="interp")

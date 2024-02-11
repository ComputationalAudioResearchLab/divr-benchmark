import torch
from .data_loader import DataLoader, InputArrays, InputTensors
from s3prl.nn import S3PRLUpstream


class S3PrlFeature(DataLoader):
    model_name: str

    def feature_init(self) -> None:
        self.model = S3PRLUpstream(self.model_name).eval().to(self.device)

    @torch.no_grad()
    def feature_function(self, batch: InputArrays) -> InputTensors:
        batch_inputs, batch_lens = batch
        batch_size, max_audios_in_session, max_audio_len = batch_inputs.shape
        audios = batch_inputs.reshape(batch_size * max_audios_in_session, max_audio_len)
        audio_lens = batch_lens.reshape(batch_size * max_audios_in_session)
        audios = torch.FloatTensor(audios).to(self.device)
        audio_lens = torch.LongTensor(audio_lens).to(self.device)
        all_hs, all_hs_len = self.model(audios, audio_lens)
        feature = torch.cat(all_hs, dim=2)
        _, max_feature_len, feature_hidden_len = feature.shape
        feature = feature.reshape(
            (batch_size, max_audios_in_session, max_feature_len, feature_hidden_len)
        )
        feature_lens = all_hs_len[0].reshape((batch_size, max_audios_in_session))
        return feature, feature_lens


class Data2Vec(S3PrlFeature):
    model_name = "data2vec_large_ll60k"


class Wav2Vec(S3PrlFeature):
    model_name = "wav2vec_large"


class UnispeechSAT(S3PrlFeature):
    model_name = "unispeech_sat_large"


class ModifiedCPC(S3PrlFeature):
    model_name = "modified_cpc"
    feature_size = 512

import torch
import numpy as np
from tqdm import tqdm
from typing import List
from pathlib import Path
import matplotlib.pyplot as plt
from pyannote.audio import Model, Inference


class Similarity:
    __sr = 16000
    __device = torch.device("cuda")
    __data_path = Path("/home/storage/divr-data/svd")
    __results_path = Path("/home/workspace/baselines/data/similarity")

    def __init__(self) -> None:
        pass

    def run(self):
        audio_files = sorted(list(self.__data_path.rglob("*-phrase.wav")))
        np.random.seed(42)
        sorted_indices = np.arange(len(audio_files))
        random_indices = np.arange(len(audio_files))
        np.random.shuffle(random_indices)
        embeddings = self.__diarizations(audios_files=audio_files)
        torch.save(embeddings, f=f"{self.__results_path}/embeddings.pkl")

        cdist_sorted = self.cosine_dist(embeddings[sorted_indices])
        cdist_random = self.cosine_dist(embeddings[random_indices])

        fig, ax = plt.subplots(2, 1, figsize=(15, 10), constrained_layout=True)
        ax[0].imshow(X=cdist_sorted, cmap="magma", aspect="auto", interpolation=None)
        ax[0].set_title("sorted data")
        ax[1].imshow(X=cdist_random, cmap="magma", aspect="auto", interpolation=None)
        ax[1].set_title("randomized data")
        fig.savefig(f"{self.__results_path}/{__class__.__name__}.png")

    def cosine_dist(self, embeddings):
        eps = 1e-8
        norm = embeddings.norm(dim=1).clamp(min=eps)
        num = (embeddings[:, None, None, :] @ embeddings[None, :, :, None]).squeeze()
        den = norm[:, None] * norm[None, :]
        cdist = num / den
        return cdist

    def __diarizations(self, audios_files: List[Path]) -> torch.Tensor:
        model = Inference(Model.from_pretrained("pyannote/embedding"), window="whole")
        embeddings = []
        for audio_file in tqdm(audios_files):
            embedding = model(audio_file)
            embeddings += [embedding]
        embeddings = torch.tensor(np.stack(embeddings))
        return embeddings

    # def __load_audio_features(self, audios_files: List[Path]):
    #     model = S3PRLUpstream(self.__model_name).eval().to(self.__device)
    #     for audio_file in tqdm(audios_files):
    #         audio, _ = librosa.load(path=audio_file, sr=self.__sr)
    #         audio_len = torch.tensor(
    #             [[len(audio)]], dtype=torch.long, device=self.__device
    #         )
    #         audio = torch.tensor([audio], dtype=torch.float, device=self.__device)
    #         print(audio)
    #         print(audio_len)
    #         all_hs, all_hs_len = model(audio, audio_len)
    #         feature = torch.cat(all_hs, dim=2)[0]
    #         feature_lens = all_hs_len[0]
    #         print(feature.shape, feature_lens)
    #         exit()


if __name__ == "__main__":
    Similarity().run()

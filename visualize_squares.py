import json
import numpy as np
import matplotlib.pyplot as plt
import librosa
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm


def pad_to_hop(audio: np.ndarray, hop_size: int) -> np.ndarray:
    audio_len = audio.shape[0]
    offset = audio_len % hop_size
    pad = hop_size - offset
    padded_audio = np.pad(audio, (0, pad))
    return padded_audio


def main(audio_file: Path, key: str, output_folder: Path, sampling_rate=16000):
    audio, _ = librosa.load(audio_file, sr=sampling_rate)

    S = librosa.stft(y=audio, n_fft=sampling_rate, hop_length=1024, win_length=1024)
    freqs = S.argmax(axis=0)
    mean_f0 = freqs.mean()
    plt.plot(freqs, label="freqs")
    plt.plot([mean_f0] * len(freqs), label="mean_f0")
    plt.legend()
    plt.savefig(f"{output_folder}/{key}_freqs.png")
    plt.close()

    hop_size = int(mean_f0) // 2
    audio = pad_to_hop(audio=audio, hop_size=hop_size)
    square = audio
    # square = (audio > 0).astype(int)
    square_img = square.reshape(-1, hop_size).T
    plt.imshow(square_img, cmap="magma", aspect="auto", interpolation=None)
    plt.savefig(f"{output_folder}/{key}_square.png")
    plt.close()


def get_root_diagnosis(diagnosis: Dict):
    name = diagnosis["name"]
    parent = diagnosis["parent"]
    if parent is None:
        return name
    return get_root_diagnosis(parent)


if __name__ == "__main__":
    output_folder = Path("/home/workspace/visualisations")
    output_folder.mkdir(parents=True, exist_ok=True)

    with open(
        "/home/workspace/data/preprocessed/svd_a_n_train.json", "r"
    ) as config_file:
        config = json.load(config_file)
    for session in tqdm(config):
        root_diagnosis = get_root_diagnosis(session["diagnosis"][0])
        main(
            audio_file=Path(session["files"][0]["path"]),
            key=f"""{root_diagnosis}_{session["gender"]}_{session["id"]}""",
            output_folder=output_folder,
        )

from multiprocessing import Pool
from pathlib import Path
from typing import List
from tqdm import tqdm
from src.preprocess.processed import ProcessedSession
from .config import ExperimentConfig
from .features import load_features


class Data:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        train_data = self.load_data(
            key="train",
            files=self.collect_files(config.train_data),
        )
        val_data = self.load_data(
            key="val",
            files=self.collect_files(config.val_data),
        )
        test_data = self.load_data(
            key="test",
            files=self.collect_files(config.test_data),
        )
        # print(val_data)

    def collect_files(self, data: List[ProcessedSession]) -> List[Path]:
        files = []
        for datum in data:
            files += [file.path for file in datum.files]
        return files

    def load_data(self, key: str, files: List[Path]):
        with Pool(10) as pool:
            input_data = [(file_path, self.config.hyper_params) for file_path in files]
            return list(
                tqdm(
                    pool.imap(load_features, input_data),
                    total=len(files),
                    desc=f"Loading {key}",
                )
            )

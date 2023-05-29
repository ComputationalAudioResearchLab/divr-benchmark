import json
import asyncio
from pathlib import Path
from .svd import SVD
from typing import Dict, List
from .base import BaseProcessor
from .processed import ProcessedDataset
from .uncommon_voice import UncommonVoice
from .uaspeech import UASpeech
from .torgo import Torgo
from .voiced import Voiced
from .avfad import AVFAD
from .meei import MEEI


class Preprocess:
    def __init__(self, lib_path: Path, database_path: Path) -> None:
        self.lib_path = lib_path
        self.database_path = database_path
        self.processors: Dict[str, BaseProcessor] = {
            "AVFAD": AVFAD(),
            "MEEI": MEEI(),
            "svd": SVD(),
            "torgo": Torgo(),
            "voiced": Voiced(),
            "UASpeech": UASpeech(),
            "UncommonVoice": UncommonVoice(),
        }

    def ensure_path(self, preprocessed_data_path: Path) -> Path:
        preprocessed_data_path.mkdir(parents=True, exist_ok=True)
        return preprocessed_data_path

    async def all(self, preprocessed_data_path: Path) -> None:
        self.ensure_path(preprocessed_data_path)
        datasets = [
            db.name
            for db in self.database_path.iterdir()
            if (db.is_dir() and db.name in self.processors)
        ]
        await self.process_datasets(
            datasets=datasets, output_path=preprocessed_data_path
        )

    async def selected(self, preprocessed_data_path: Path, datasets: List[str]) -> None:
        self.ensure_path(preprocessed_data_path)
        await self.process_datasets(
            datasets=datasets, output_path=preprocessed_data_path
        )

    async def process_datasets(self, datasets: List[str], output_path: Path):
        processed_datasets: List[ProcessedDataset] = await asyncio.gather(
            *[
                self.processors[db](
                    source_path=Path(f"{self.database_path}/{db}"),
                )
                for db in datasets
            ]
        )
        for dataset in processed_datasets:
            db_key = dataset.db
            data = dataset.sessions
            with open(f"{output_path}/{db_key}.json", "w") as outfile:
                json.dump(data, outfile, indent=2, default=vars)

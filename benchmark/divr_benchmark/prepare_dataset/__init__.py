import asyncio
from pathlib import Path
from typing import Dict, List
from divr_diagnosis import DiagnosisMap

from .svd import SVD
from .base import BaseProcessor
from .uncommon_voice import UncommonVoice
from .uaspeech import UASpeech
from .torgo import Torgo
from .voiced import Voiced
from .avfad import AVFAD
from .meei import MEEI


class PrepareDataset:
    def __init__(
        self,
        database_path: Path,
        audio_extraction_path: Path,
        diagnosis_map: DiagnosisMap,
    ) -> None:
        self.database_path = database_path
        meei_extraction_path = Path(f"{audio_extraction_path}/meei")
        meei_extraction_path.mkdir(exist_ok=True, parents=True)
        voiced_extraction_path = Path(f"{audio_extraction_path}/voiced")
        voiced_extraction_path.mkdir(exist_ok=True, parents=True)
        self.processors: Dict[str, BaseProcessor] = {
            "AVFAD": AVFAD(diagnosis_map=diagnosis_map),
            "MEEI": MEEI(
                audio_extraction_path=meei_extraction_path,
                diagnosis_map=diagnosis_map,
            ),
            "svd": SVD(diagnosis_map=diagnosis_map),
            "torgo": Torgo(diagnosis_map=diagnosis_map),
            "voiced": Voiced(diagnosis_map=diagnosis_map),
            "UASpeech": UASpeech(diagnosis_map=diagnosis_map),
            "UncommonVoice": UncommonVoice(diagnosis_map=diagnosis_map),
        }

    def ensure_path(self, prepared_data_path: Path) -> Path:
        prepared_data_path.mkdir(parents=True, exist_ok=True)
        return prepared_data_path

    async def all(self, prepared_data_path: Path) -> None:
        self.ensure_path(prepared_data_path)
        datasets = [
            db.name
            for db in self.database_path.iterdir()
            if (db.is_dir() and db.name in self.processors)
        ]
        await self.process_datasets(datasets=datasets, output_path=prepared_data_path)

    async def selected(self, prepared_data_path: Path, datasets: List[str]) -> None:
        self.ensure_path(prepared_data_path)
        await self.process_datasets(datasets=datasets, output_path=prepared_data_path)

    async def process_datasets(self, datasets: List[str], output_path: Path):
        await asyncio.gather(
            *[
                self.processors[db](
                    source_path=Path(f"{self.database_path}/{db}"),
                    output_path=output_path,
                )
                for db in datasets
            ]
        )

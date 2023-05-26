import asyncio
from pathlib import Path
from .svd import SVD
from typing import Dict
from .base import BaseProcessor


class Preprocess:
    def __init__(self, lib_path: Path, database_path: Path) -> None:
        self.lib_path = lib_path
        self.database_path = database_path
        self.processors: Dict[str, BaseProcessor] = {"svd": SVD()}

    def ensure_path(self, preprocessed_data_path: Path) -> Path:
        preprocessed_data_path.mkdir(parents=True, exist_ok=True)
        return preprocessed_data_path

    async def all(self, preprocessed_data_path: Path):
        self.ensure_path(preprocessed_data_path)
        datasets = [
            db.name
            for db in self.database_path.iterdir()
            if (db.is_dir() and db.name in self.processors)
        ]
        await asyncio.gather(
            *[
                self.processors[db](
                    source_path=Path(f"{self.database_path}/{db}"),
                    dest_path=self.ensure_path(Path(f"{preprocessed_data_path}/{db}")),
                )
                for db in datasets
            ]
        )

    async def selected(self, preprocessed_data_path: Path, datasets):
        self.ensure_path(preprocessed_data_path)
        await asyncio.gather(
            *[
                self.processors[db](
                    source_path=Path(f"{self.database_path}/{db}"),
                    dest_path=self.ensure_path(Path(f"{preprocessed_data_path}/{db}")),
                )
                for db in datasets
            ]
        )

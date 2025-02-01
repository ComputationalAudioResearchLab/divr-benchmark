from pathlib import Path
from typing import List
from class_argparse import ClassArgParser
from divr_diagnosis import diagnosis_maps

from .download import Download
from .prepare_dataset import PrepareDataset
from .logger import Logger
from .task_generator import VERSIONS, collect_diagnosis_terms, generate_tasks


class Main(ClassArgParser):
    def __init__(self) -> None:
        super().__init__(name="DiVR Benchmark")
        self.logger = Logger(log_path="/tmp/main.log", key="main")

    async def download_openaccess(
        self,
        database_path: Path,
        all: bool = False,
        datasets: List[str] = [],
    ) -> None:
        downloader = Download(database_path=database_path, logger=self.logger)
        if all:
            await downloader.all()
        elif len(datasets) != 0:
            await downloader.selected(datasets)
        else:
            print("Must specify either --all or --datasets")

    async def prepare_dataset(
        self,
        database_path: Path,
        audio_extraction_path: Path,
        prepared_data_path: Path,
        datasets: List[str] = [],
        all: bool = False,
    ):
        processor = PrepareDataset(
            database_path=database_path,
            audio_extraction_path=audio_extraction_path,
            diagnosis_map=diagnosis_maps.USVAC_2025(),
        )
        if all:
            await processor.all(prepared_data_path)
        elif len(datasets) != 0:
            await processor.selected(prepared_data_path, datasets)
        else:
            print("Must specify either --all or --datasets")

    async def generate_tasks(self, data_store_path: Path, version: VERSIONS):
        await generate_tasks(
            version=version,
            source_path=data_store_path,
            diagnosis_map=diagnosis_maps.USVAC_2025(),
        )

    async def collect_diagnosis_terms(self, version: VERSIONS, data_store_path: Path):
        await collect_diagnosis_terms(version=version, source_path=data_store_path)


if __name__ == "__main__":
    main = Main()
    main()

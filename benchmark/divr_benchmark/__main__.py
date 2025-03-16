from pathlib import Path
from typing import List
from class_argparse import ClassArgParser

from .download import Download
from .logger import Logger
from .task_generator import VERSIONS, collect_diagnosis_terms, generate_tasks
from .data_counter import data_counter


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

    async def collect_diagnosis_terms(self, version: VERSIONS, data_store_path: Path):
        await collect_diagnosis_terms(version=version, source_path=data_store_path)

    async def count_per_db_and_diag_map(
        self, version: VERSIONS, data_store_path: Path, output_path: Path
    ):
        await data_counter(
            version=version,
            data_store_path=data_store_path,
            output_path=output_path,
        )


if __name__ == "__main__":
    main = Main()
    main()

from pathlib import Path
from typing import List
from class_argparse import ClassArgParser
from .download import Download
from .diagnosis import (
    analysis as analyse_diagnosis,
    reclassification_candidates,
    level_3_confusion,
    Processor,
)
from .prepare_dataset import PrepareDataset
from .logger import Logger
from .task_generator import generate_tasks, VERSIONS


class Main(ClassArgParser):
    def __init__(self) -> None:
        super().__init__(name="DiVR Benchmark")
        self.logger = Logger(log_path="/tmp/main.log", key="main")

    def analyse_diagnosis_classifications(
        self,
        source_path: Path,
        output_confusion_path: Path,
    ):
        analyse_diagnosis(
            source_path=source_path,
            output_confusion_path=output_confusion_path,
        )

    def reclassification_candidates(self, output_path: Path):
        reclassification_candidates(output_path=output_path)

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
        )
        if all:
            await processor.all(prepared_data_path)
        elif len(datasets) != 0:
            await processor.selected(prepared_data_path, datasets)
        else:
            print("Must specify either --all or --datasets")

    def level_3_confusion(self):
        level_3_confusion()

    def generate_tasks(self, data_store_path: Path, version: VERSIONS):
        generate_tasks(version=version, source_path=data_store_path)

    def process_diagnosis_list(self):
        Processor().run()


if __name__ == "__main__":
    main = Main()
    main()

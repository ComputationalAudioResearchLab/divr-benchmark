import yaml
import typing
import asyncio
from pathlib import Path
from typing import Literal

from .audio_loader import AudioLoader
from .task import Task
from ..logger import Logger
from ..download import Download
from ..diagnosis import DiagnosisMap


VERSIONS = Literal["v1"]
versions = typing.get_args(VERSIONS)
diagnosis_map_maps = {"v1": DiagnosisMap.v1}


class Benchmark:

    def __init__(self, storage_path: str, version: VERSIONS) -> None:
        if not Path(storage_path).is_dir():
            raise ValueError(
                f"storage_path: ({storage_path}) is not a valid directory."
            )
        if version not in versions:
            raise ValueError(
                f"invalid version ({version}) selected. Choose from: {versions}"
            )
        module_path = Path(__file__).parent.parent.resolve()
        self.__diagnosis_map = diagnosis_map_maps[version]()
        self.__logger = Logger(log_path=f"{storage_path}/logs", key=f"{version}")
        self.__data_path = Path(f"{storage_path}/data")
        self.__audio_loader = AudioLoader(version, self.__data_path)
        self.__downloader = Download(
            database_path=self.__data_path, logger=self.__logger
        )
        self.__tasks_path = f"{module_path}/tasks/{version}"
        self.__ensure_datasets(tasks_path=self.__tasks_path)

    def task(self, stream: int, task: int) -> Task:
        stream_path = Path(f"{self.__tasks_path}/{stream}/")
        if not stream_path.is_dir():
            raise ValueError("Invalid stream selected")
        task_path = Path(f"{stream_path}/{task}")
        if not task_path.is_dir():
            raise ValueError("Invalid task selected")

        train_path = Path(f"{task_path}/train.yml")
        if not train_path.is_file():
            train_path = Path(f"{stream_path}/train.yml")

        val_path = Path(f"{task_path}/val.yml")
        if not val_path.is_file():
            val_path = Path(f"{stream_path}/val.yml")

        test_path = Path(f"{task_path}/test.yml")

        return Task(
            diagnosis_map=self.__diagnosis_map,
            audio_loader=self.__audio_loader,
            train=train_path,
            val=val_path,
            test=test_path,
        )

    def __ensure_datasets(self, tasks_path: str) -> None:
        datasets_file = f"{tasks_path}/datasets.yml"
        with open(datasets_file, "r") as df:
            datasets = yaml.load(df, Loader=yaml.FullLoader)

        to_download = []
        for dataset in datasets:
            dataset_path = Path(f"{self.__data_path}/{dataset}")
            if dataset_path.is_dir():
                self.__logger.info(f"{dataset} already exists at {dataset_path}")
            else:
                self.__logger.info(
                    f"{dataset} does not exist. Will create at {dataset_path}"
                )
                to_download.append(dataset)
        asyncio.run(self.__downloader.selected(datasets=to_download))

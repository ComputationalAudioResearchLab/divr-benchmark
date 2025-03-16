import typing
from pathlib import Path
from typing import Literal
from divr_diagnosis import DiagnosisMap

from .audio_loader import AudioLoader
from .task import Task
from ..task_generator import generator_map

VERSIONS = Literal["v1"]
versions = typing.get_args(VERSIONS)
task_generator_maps = {"v1": generator_map["v1"]}


class Benchmark:
    def __init__(
        self,
        storage_path: str | Path,
        version: VERSIONS,
        quiet: bool = False,
        sample_rate: int = 16000,
    ) -> None:
        self.__quiet = quiet
        if not Path(storage_path).is_dir():
            raise ValueError(
                f"storage_path: ({storage_path}) is not a valid directory."
            )
        if version not in versions:
            raise ValueError(
                f"invalid version ({version}) selected. Choose from: {versions}"
            )
        self.__data_path = Path(f"{storage_path}/data")
        self.__audio_loader = AudioLoader(version, self.__data_path, sample_rate)
        self.__task_generator = task_generator_maps[version]

    async def generate_task(
        self,
        filter_func,
        task_path: Path,
        diagnosis_map: DiagnosisMap,
        allow_incomplete_classification: bool,
    ) -> None:
        await self.__task_generator.generate_task(
            source_path=self.__data_path,
            filter_func=filter_func,
            task_path=task_path,
            diagnosis_map=diagnosis_map,
            allow_incomplete_classification=allow_incomplete_classification,
        )

    def load_task(
        self,
        task_path: Path,
        diag_level: int | None,
        diagnosis_map: DiagnosisMap,
        load_audios: bool = True,
    ) -> Task:
        if not task_path.is_dir():
            raise ValueError("Invalid task selected")
        return Task(
            diagnosis_map=diagnosis_map,
            audio_loader=self.__audio_loader,
            train=Path(f"{task_path}/train.yml"),
            val=Path(f"{task_path}/val.yml"),
            test=Path(f"{task_path}/test.yml"),
            quiet=self.__quiet,
            diag_level=diag_level,
            load_audios=load_audios,
        )

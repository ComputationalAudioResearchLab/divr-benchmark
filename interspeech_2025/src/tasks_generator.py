from __future__ import annotations
from pathlib import Path
from divr_benchmark import Benchmark
from divr_benchmark.task_generator.task import Task
from divr_diagnosis import diagnosis_maps


class TaskGenerator:

    def __init__(self, research_data_path: Path, tasks_path: Path) -> None:
        self.__benchmark = Benchmark(
            storage_path=research_data_path,
            version="v1",
        )
        self.__tasks_path = tasks_path

    def load_task(self, task: str, diag_level: int | None) -> Task:
        diagnosis_map = diagnosis_maps.USVAC_2025()
        task_path = Path(f"{self.__tasks_path}/{task}")
        return self.__benchmark.load_task(
            task_path=task_path,
            diag_level=diag_level,
            diagnosis_map=diagnosis_map,
        )

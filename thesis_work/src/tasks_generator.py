from __future__ import annotations
from pathlib import Path
from divr_benchmark import Benchmark
from divr_benchmark.task_generator.task import Task
from divr_diagnosis import diagnosis_maps, DiagnosisMap


class TaskGenerator:

    __diagnosis_maps = {
        "Compton_2022": diagnosis_maps.Compton_2022,
        "daSilvaMoura_2024": diagnosis_maps.daSilvaMoura_2024,
        "Sztaho_2018": diagnosis_maps.Sztaho_2018,
        "Zaim_2023": diagnosis_maps.Zaim_2023,
    }

    def __init__(self, research_data_path: Path, tasks_path: Path) -> None:
        self.__benchmark = Benchmark(
            storage_path=research_data_path,
            version="v1",
        )
        self.__tasks_path = tasks_path

    def load_task(
        self,
        task: str,
        diag_level: int | None,
        diagnosis_map: DiagnosisMap | None = None,
        load_audios: bool = True,
    ) -> Task:
        if diagnosis_map is None:
            diagnosis_map = self.get_diagnosis_map(task)
        task_path = Path(f"{self.__tasks_path}/{task}")
        return self.__benchmark.load_task(
            task_path=task_path,
            diag_level=diag_level,
            diagnosis_map=diagnosis_map,
        )

    def get_diagnosis_map(self, task, allow_unmapped: bool = False):
        if "-" in task:
            # specified diagnosis map
            diagnosis_map_key = task.split("-", maxsplit=1)[0]
            return self.__diagnosis_maps[diagnosis_map_key](
                allow_unmapped=allow_unmapped
            )
        else:
            return diagnosis_maps.USVAC_2025(allow_unmapped=allow_unmapped)

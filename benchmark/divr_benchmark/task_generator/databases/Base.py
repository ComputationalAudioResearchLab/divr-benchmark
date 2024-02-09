from typing import List, Callable
from pathlib import Path
from ...diagnosis import DiagnosisMap
from ...prepare_dataset.database_generator import DatabaseGenerator
from ...prepare_dataset.processed import (
    ProcessedDataset,
    ProcessedFile,
    ProcessedSession,
)
from ..task import Task

FileFilter = Callable[[List[ProcessedFile]], List[ProcessedFile]]


class Base:
    def __init__(self, source_path: Path) -> None:
        self.diagnosis_map = DiagnosisMap.v1()
        self.database_generator = DatabaseGenerator(
            diagnosis_map=self.diagnosis_map,
            train_split=0.7,
            test_split=0.2,
            random_seed=42,
        )
        self.dataset = self.prepare_dataset(source_path=source_path)
        self.__source_path = str(source_path)

    def prepare_dataset(self, source_path: Path) -> ProcessedDataset:
        raise NotImplementedError()

    def to_audio_key(self, source_path: ProcessedFile) -> str:
        return str(source_path.path).removeprefix(self.__source_path).removeprefix("/")

    def all_train(self, level: int) -> List[Task]:
        return self.to_tasks(self.dataset.train_sessions, level=level, file_filter=None)

    def all_val(self, level: int) -> List[Task]:
        return self.to_tasks(self.dataset.val_sessions, level=level, file_filter=None)

    def all_test(self, level: int) -> List[Task]:
        return self.to_tasks(self.dataset.test_sessions, level=level, file_filter=None)

    def to_tasks(
        self,
        sessions: List[ProcessedSession],
        level: int,
        file_filter: FileFilter | None,
    ) -> List[Task]:
        tasks: List[Task] = []
        for session in sessions:
            root_diagnosis = session.best_diagnosis.at_level(level)
            if file_filter is None:
                files = session.files
            else:
                files = file_filter(session.files)
            for file_idx, file_path in enumerate(files):
                task = Task(
                    id=f"{session.id}_{file_idx}",
                    age=session.age,
                    gender=session.gender,
                    label=root_diagnosis,
                    audio_key=self.to_audio_key(file_path),
                )
                tasks.append(task)
        return tasks

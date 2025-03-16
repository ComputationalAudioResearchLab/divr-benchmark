from pathlib import Path
from typing import Callable, Dict, List, Set
from divr_diagnosis import Diagnosis, DiagnosisMap

from ...prepare_dataset.database_generator import DatabaseGenerator
from ...prepare_dataset.processed import (
    ProcessedFile,
    ProcessedSession,
)
from ..task import Task

FileFilter = Callable[[List[ProcessedFile]], List[ProcessedFile]]


class Base:
    DB_NAME: str

    def __init__(
        self,
        source_path: Path,
    ) -> None:
        self.__source_path = source_path
        self.__db_source_path = Path(f"{source_path}/{self.DB_NAME}")

    async def init(
        self,
        diagnosis_map: DiagnosisMap,
        allow_incomplete_classification: bool,
        min_tasks: int | None,
    ):
        database_generator = DatabaseGenerator(
            train_split=0.7,
            test_split=0.2,
            random_seed=42,
        )
        sessions = await self.prepare_dataset(
            source_path=self.__db_source_path,
            allow_incomplete_classification=allow_incomplete_classification,
            min_tasks=min_tasks,
            diagnosis_map=diagnosis_map,
        )
        self.dataset = database_generator.generate(
            db_name=self.DB_NAME,
            sessions=sessions,
        )

    async def collect_diagnosis_terms(self) -> Set[str]:
        return await self._collect_diagnosis_terms(source_path=self.__db_source_path)

    async def _collect_diagnosis_terms(self, source_path: Path) -> Set[str]:
        raise NotImplementedError()

    async def prepare_dataset(
        self,
        source_path: Path,
        allow_incomplete_classification: bool,
        min_tasks: int | None,
        diagnosis_map: DiagnosisMap,
    ) -> List[ProcessedSession]:
        raise NotImplementedError()

    def to_audio_key(self, source_path: ProcessedFile) -> str:
        return (
            str(source_path.path)
            .removeprefix(str(self.__source_path))
            .removeprefix("/")
        )

    def all_train(self, level: int) -> List[Task]:
        return self.to_individual_file_tasks(
            self.dataset.train_sessions, level=level, file_filter=None
        )

    def all_val(self, level: int) -> List[Task]:
        return self.to_individual_file_tasks(
            self.dataset.val_sessions, level=level, file_filter=None
        )

    def all_test(self, level: int) -> List[Task]:
        return self.to_individual_file_tasks(
            self.dataset.test_sessions, level=level, file_filter=None
        )

    def all(self, level: int) -> List[Task]:
        return self.to_individual_file_tasks(
            sessions=self.dataset.all_sessions, level=level, file_filter=None
        )

    def count_per_diag(self, level: int) -> Dict[Diagnosis, set[str]]:
        counts: dict[Diagnosis, set[str]] = {}
        for session in self.dataset.all_sessions:
            root_diagnosis = session.best_diagnosis.at_level(level)
            if root_diagnosis not in counts:
                counts[root_diagnosis] = set()
            counts[root_diagnosis].add(session.speaker_id)
        return counts

    def to_individual_file_tasks(
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
                    speaker_id=session.speaker_id,
                    age=session.age,
                    gender=session.gender,
                    label=root_diagnosis,
                    audio_keys=[self.to_audio_key(file_path)],
                )
                tasks.append(task)
        return tasks

    def to_multi_file_tasks(
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
            task = Task(
                id=f"{session.id}",
                speaker_id=session.speaker_id,
                age=session.age,
                gender=session.gender,
                label=root_diagnosis,
                audio_keys=list(map(self.to_audio_key, files)),
            )
            tasks.append(task)
        return tasks

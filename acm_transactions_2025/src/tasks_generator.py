from __future__ import annotations
import asyncio
from typing import Dict, List, Literal
from pathlib import Path
from divr_benchmark import Benchmark
from divr_benchmark.task_generator import DatabaseFunc, Dataset
from divr_benchmark.task_generator.task import Task
from divr_benchmark.task_generator.databases import SVD
from divr_diagnosis import diagnosis_maps


class TaskGenerator:

    TASKS = Literal["phrase", "a_n", "i_n", "u_n"]

    def __init__(self, research_data_path: Path) -> None:
        self.__benchmark = Benchmark(
            storage_path=research_data_path,
            version="v1",
        )
        cur_path = Path(__file__).parent.resolve()
        self.__tasks_path = self.__ensure_path(f"{cur_path}/tasks")

    def load_task(self, task: TaskGenerator.TASKS, diag_level: int | None) -> Task:
        task_path = Path(f"{self.__tasks_path}/{task}")
        diagnosis_map = diagnosis_maps.USVAC_2025()
        return self.__benchmark.load_task(
            task_path=task_path,
            diag_level=diag_level,
            diagnosis_map=diagnosis_map,
        )

    async def generate(self) -> None:
        diagnosis_map = diagnosis_maps.USVAC_2025()
        diag_level = 4
        coros = []
        coros += [
            self.__benchmark.generate_task(
                filter_func=self.__filter_func(
                    allowed_suffixes=[
                        "-phrase.wav",
                    ],
                    diag_level=diag_level,
                ),
                task_path=self.__ensure_path(f"{self.__tasks_path}/phrase"),
                diagnosis_map=diagnosis_map,
            )
        ]
        coros += [
            self.__benchmark.generate_task(
                filter_func=self.__filter_func(
                    allowed_suffixes=[
                        "-a_n.wav",
                    ],
                    diag_level=diag_level,
                ),
                task_path=self.__ensure_path(f"{self.__tasks_path}/a_n"),
                diagnosis_map=diagnosis_map,
            )
        ]
        coros += [
            self.__benchmark.generate_task(
                filter_func=self.__filter_func(
                    allowed_suffixes=[
                        "-i_n.wav",
                    ],
                    diag_level=diag_level,
                ),
                task_path=self.__ensure_path(f"{self.__tasks_path}/i_n"),
                diagnosis_map=diagnosis_map,
            )
        ]
        coros += [
            self.__benchmark.generate_task(
                filter_func=self.__filter_func(
                    allowed_suffixes=[
                        "-u_n.wav",
                    ],
                    diag_level=diag_level,
                ),
                task_path=self.__ensure_path(f"{self.__tasks_path}/u_n"),
                diagnosis_map=diagnosis_map,
            )
        ]
        coros += [
            self.__benchmark.generate_task(
                filter_func=self.__filter_func(
                    allowed_suffixes=[
                        "-phrase.wav",
                        "-a_n.wav",
                        "-i_n.wav",
                        "-u_n.wav",
                    ],
                    diag_level=diag_level,
                ),
                task_path=self.__ensure_path(f"{self.__tasks_path}/all"),
                diagnosis_map=diagnosis_map,
            )
        ]
        cross_test_datasets = [
            "avfad",
            "meei",
            "torgo",
            "uaspeech",
            "uncommon_voice",
            "voiced",
        ]
        for db in cross_test_datasets:
            coros += [
                self.__benchmark.generate_task(
                    filter_func=self.__cross_test_filter_func(
                        databse_name=db,
                        diag_level=diag_level,
                    ),
                    task_path=self.__ensure_path(
                        f"{self.__tasks_path}/cross_test_{db}"
                    ),
                    diagnosis_map=diagnosis_map,
                )
            ]
        await asyncio.gather(*coros)

    def __cross_test_filter_func(self, databse_name: str, diag_level: int):
        async def filter_func(database_func: DatabaseFunc) -> DatabaseFunc:
            db = await database_func(name=databse_name)
            return Dataset(
                train=[],
                val=[],
                test=db.all(level=diag_level),
            )

        return filter_func

    def __filter_func(self, allowed_suffixes: List[str], diag_level: int):
        async def filter_func(database_func: DatabaseFunc) -> Dataset:
            svd_data = await self.__svd_data(
                database_func=database_func, diag_level=diag_level
            )
            svd_data.train = self.__filter_audio_files(
                tasks=svd_data.train, allowed_suffixes=allowed_suffixes
            )
            svd_data.val = self.__filter_audio_files(
                tasks=svd_data.val, allowed_suffixes=allowed_suffixes
            )
            svd_data.test = self.__filter_audio_files(
                tasks=svd_data.test, allowed_suffixes=allowed_suffixes
            )
            return svd_data

        return filter_func

    def __filter_audio_files(
        self, tasks: List[Task], allowed_suffixes: List[str]
    ) -> List[Task]:
        def file_filter(file_path: str):
            for suffix in allowed_suffixes:
                if file_path.endswith(suffix):
                    return True
            return False

        for task in tasks:
            task.audio_keys = [x for x in task.audio_keys if file_filter(x)]
        return tasks

    async def __svd_data(self, database_func: DatabaseFunc, diag_level: int) -> Dataset:
        db = await database_func(name="svd", min_tasks=SVD.max_tasks)
        train_data = db.all_train(level=diag_level)
        val_data = db.all_val(level=diag_level)
        test_data = db.all_test(level=diag_level)

        train_data = self.__collate_by_speaker_id(tasks=train_data)
        val_data = self.__collate_by_speaker_id(tasks=val_data)
        test_data = self.__collate_by_speaker_id(tasks=test_data)

        test_totals = self.__count_total_per_label(tasks=test_data)
        min_count = 5
        for key in list(test_totals.keys()):
            if test_totals[key] < min_count:
                del test_totals[key]
        allowed_labels = list(test_totals.keys())

        train_data = self.__filter_by_labels(
            tasks=train_data, allowed_labels=allowed_labels
        )
        val_data = self.__filter_by_labels(
            tasks=val_data, allowed_labels=allowed_labels
        )
        test_data = self.__filter_by_labels(
            tasks=test_data, allowed_labels=allowed_labels
        )

        return Dataset(
            train=train_data,
            val=val_data,
            test=test_data,
        )

    def __collate_by_speaker_id(self, tasks: List[Task]) -> List[Task]:
        speakers = {}
        for task in tasks:
            speaker_id = task.speaker_id
            if speaker_id not in speakers:
                speakers[speaker_id] = task
            else:
                speakers[speaker_id].audio_keys += task.audio_keys
        return list(speakers.values())

    def __filter_by_labels(
        self, tasks: List[Task], allowed_labels: List[str]
    ) -> List[Task]:
        return [x for x in tasks if x.label.name in allowed_labels]

    def __count_total_per_label(self, tasks: List[Task]) -> Dict[str, int]:
        """
        Counts unique speakers per label in the given list of tasks
        """
        labels = {}
        for task in tasks:
            task_label = task.label.name
            if task_label not in labels:
                labels[task_label] = 1
            else:
                labels[task_label] += 1
        return labels

    def __ensure_path(self, path: str) -> Path:
        _path = Path(path)
        _path.mkdir(exist_ok=True, parents=True)
        return _path

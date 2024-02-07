import asyncio
import json
import statistics
from pathlib import Path
from typing import List
from ..diagnosis import DiagnosisMap
from .processed import ProcessedDataset, ProcessedSession
from .database_generator import DatabaseGenerator


class BaseProcessor:
    def __init__(self) -> None:
        self.diagnosis_map = DiagnosisMap()
        self.database_generator = DatabaseGenerator(
            diagnosis_map=self.diagnosis_map,
            train_split=0.7,
            test_split=0.2,
            random_seed=42,
        )

    async def __call__(self, source_path: Path, output_path: Path):
        raise NotImplementedError()

    async def process(
        self,
        output_path: Path,
        db_name: str,
        sessions: List[ProcessedSession],
    ) -> ProcessedDataset:
        dataset = self.database_generator.generate(
            db_name=db_name,
            sessions=sessions,
        )
        db_key = dataset.db_name
        db_path = f"{output_path}/{db_key}"
        Path(db_path).mkdir(exist_ok=True, parents=True)
        await self.generate_diagnosis_set(
            dataset=dataset, file_path=f"{db_path}/diagnosis.json"
        )
        await self.generate_demographics(
            dataset=dataset, file_path_base=f"{db_path}/demographics"
        )
        sessions_db = {
            "train": dataset.train_sessions,
            "val": dataset.val_sessions,
            "test": dataset.test_sessions,
        }
        for session_key, session_data in sessions_db.items():
            await self.__save_json(
                data=session_data,
                file_path=f"{db_path}/{session_key}.json",
            )
        return dataset

    async def generate_diagnosis_set(self, dataset: ProcessedDataset, file_path):
        diagnosis_set = set()
        total_sessions = (
            dataset.train_sessions + dataset.test_sessions + dataset.val_sessions
        )
        for session in total_sessions:
            for diagnosis in session.diagnosis:
                diagnosis_set.add(diagnosis.name)
        await self.__save_json(data=sorted(list(diagnosis_set)), file_path=file_path)

    async def generate_demographics(self, dataset: ProcessedDataset, file_path_base):
        data = {
            "train": dataset.train_sessions,
            "test": dataset.test_sessions,
            "val": dataset.val_sessions,
        }

        coros = []
        for level in range(4):
            for key, val in data.items():
                coros.append(
                    self.generate_demographics_at_level(
                        sessions=val,
                        level=level,
                        file_path_base=f"{file_path_base}_{key}",
                    )
                )
        await asyncio.gather(*coros)

    async def generate_demographics_at_level(
        self, sessions: List[ProcessedSession], level: int, file_path_base
    ):
        demographics = {}
        for session in sessions:
            diagnosis_name = session.best_diagnosis.at_level(level).name
            if diagnosis_name not in demographics:
                demographics[diagnosis_name] = {}
            diagnosis = demographics[diagnosis_name]
            gender = session.gender
            if gender not in diagnosis:
                diagnosis[gender] = {"ages": [], "total": 0}
            diagnosis[gender]["total"] += 1
            if session.age is not None:
                diagnosis[gender]["ages"] += [session.age]
        for diagnosis in demographics:
            for gender in demographics[diagnosis]:
                ages = demographics[diagnosis][gender]["ages"]
                total = demographics[diagnosis][gender]["total"]
                total_ages = len(ages)
                age_stats = None
                if total_ages > 0:
                    age_stats = (
                        {
                            "mean": statistics.mean(ages),
                            "std": statistics.stdev(ages) if total_ages > 1 else None,
                            "min": min(ages),
                            "max": max(ages),
                        },
                    )
                demographics[diagnosis][gender] = {
                    "total": total,
                    "age_stats": age_stats,
                }
        await self.__save_json(
            data=demographics, file_path=f"{file_path_base}_{level}.json"
        )

    async def __save_json(self, data, file_path):
        with open(file_path, "w") as outfile:
            json.dump(data, outfile, indent=2, default=vars)

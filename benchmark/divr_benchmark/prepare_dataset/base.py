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
        await self.generate_diagnosis_set(
            sessions=sessions, file_path=f"{output_path}/{db_key}_diagnosis.json"
        )
        await self.generate_demographics(
            sessions=sessions, file_path=f"{output_path}/{db_key}_demographics.json"
        )
        sessions_db = {
            "train": dataset.train_sessions,
            "val": dataset.val_sessions,
            "test": dataset.test_sessions,
        }
        for session_key, session_data in sessions_db.items():
            await self.__save_json(
                data=session_data,
                file_path=f"{output_path}/{db_key}_{session_key}.json",
            )
        return dataset

    async def generate_diagnosis_set(self, sessions: List[ProcessedSession], file_path):
        diagnosis_set = set()
        for session in sessions:
            for diagnosis in session.diagnosis:
                diagnosis_set.add(diagnosis.name)
        await self.__save_json(data=sorted(list(diagnosis_set)), file_path=file_path)

    async def generate_demographics(self, sessions: List[ProcessedSession], file_path):
        demographics = {}
        for session in sessions:
            root_diagnosis = session.diagnosis[0].root()
            if root_diagnosis not in demographics:
                demographics[root_diagnosis] = {}
            diagnosis = demographics[root_diagnosis]
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
        await self.__save_json(data=demographics, file_path=file_path)

    async def __save_json(self, data, file_path):
        with open(file_path, "w") as outfile:
            json.dump(data, outfile, indent=2, default=vars)

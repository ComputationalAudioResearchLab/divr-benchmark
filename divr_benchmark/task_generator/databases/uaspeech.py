from pathlib import Path
from typing import List, Set
from divr_diagnosis import DiagnosisMap

from .Base import Base
from .gender import Gender
from ...prepare_dataset.processed import (
    ProcessedSession,
    ProcessedFile,
)


class UASpeech(Base):
    DB_NAME = "uaspeech"
    __df = [
        {"id": "CF02", "diagnosis": "normal", "gender": "F", "age": None},
        {"id": "CF03", "diagnosis": "normal", "gender": "F", "age": None},
        {"id": "CF04", "diagnosis": "normal", "gender": "F", "age": None},
        {"id": "CF05", "diagnosis": "normal", "gender": "F", "age": None},
        {"id": "CM01", "diagnosis": "normal", "gender": "M", "age": None},
        {"id": "CM04", "diagnosis": "normal", "gender": "M", "age": None},
        {"id": "CM05", "diagnosis": "normal", "gender": "M", "age": None},
        {"id": "CM06", "diagnosis": "normal", "gender": "M", "age": None},
        {"id": "CM08", "diagnosis": "normal", "gender": "M", "age": None},
        {"id": "CM09", "diagnosis": "normal", "gender": "M", "age": None},
        {"id": "CM10", "diagnosis": "normal", "gender": "M", "age": None},
        {"id": "CM12", "diagnosis": "normal", "gender": "M", "age": None},
        {"id": "CM13", "diagnosis": "normal", "gender": "M", "age": None},
        {"id": "M01", "diagnosis": "Spastic", "gender": "M", "age": "18"},
        {"id": "M04", "diagnosis": "Spastic", "gender": "M", "age": "18"},
        {"id": "M05", "diagnosis": "Spastic", "gender": "M", "age": "21"},
        {"id": "M06", "diagnosis": "Spastic", "gender": "M", "age": "18"},
        {"id": "M07", "diagnosis": "Spastic", "gender": "M", "age": "58"},
        {"id": "M08", "diagnosis": "Spastic", "gender": "M", "age": "28"},
        {"id": "M09", "diagnosis": "Spastic", "gender": "M", "age": "18"},
        {"id": "M10", "diagnosis": "Not sure", "gender": "M", "age": "21"},
        {"id": "F02", "diagnosis": "Spastic", "gender": "F", "age": "30"},
        {"id": "F03", "diagnosis": "Spastic", "gender": "F", "age": "51"},
        {
            "id": "F04",
            "diagnosis": "Athetoid (or mixed)",
            "gender": "F",
            "age": "18",
        },
        {"id": "F05", "diagnosis": "Spastic", "gender": "F", "age": "22"},
        {"id": "M11", "diagnosis": "Athetoid", "gender": "M", "age": "48"},
        {"id": "M12", "diagnosis": "Mixed", "gender": "M", "age": "19"},
        {"id": "M13", "diagnosis": "Spastic", "gender": "M", "age": "44"},
        {"id": "M14", "diagnosis": "Spastic", "gender": "M", "age": "40"},
        {"id": "M16", "diagnosis": "Spastic", "gender": "M", "age": None},
    ]

    async def _collect_diagnosis_terms(self, source_path: Path) -> Set[str]:
        return set([data["diagnosis"].lower() for data in self.__df])

    async def prepare_dataset(
        self,
        source_path: Path,
        allow_incomplete_classification: bool,
        min_tasks: int | None,
        diagnosis_map: DiagnosisMap,
    ) -> List[ProcessedSession]:
        sessions = []
        data_path = f"{source_path}/UASpeech/audio/original/"

        for data in self.__df:
            speaker_id = data["id"]
            diagnosis = data["diagnosis"].lower()
            diagnosis = (
                diagnosis_map[diagnosis]
                if diagnosis in diagnosis_map
                else diagnosis_map.unclassified
            )
            if allow_incomplete_classification or not diagnosis.incompletely_classified:
                age = int(data["age"]) if data["age"] is not None else None
                gender = Gender.format(data["gender"].strip())
                file_paths = [
                    path
                    for path in Path(f"{data_path}/{speaker_id}").glob("*.wav")
                    if not path.name.startswith(".")
                ]
                num_files = len(file_paths)
                if min_tasks is None or num_files >= min_tasks:
                    session = ProcessedSession(
                        id=f"uaspeech_{speaker_id}",
                        speaker_id=speaker_id,
                        age=age,
                        gender=gender,
                        diagnosis=[diagnosis],
                        files=[ProcessedFile(path=path) for path in file_paths],
                        num_files=num_files,
                    )
                    sessions += [session]
        return sessions

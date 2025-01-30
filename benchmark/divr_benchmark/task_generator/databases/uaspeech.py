from pathlib import Path
from .Base import Base
from .gender import Gender
from ...prepare_dataset.processed import (
    ProcessedDataset,
    ProcessedSession,
    ProcessedFile,
)


class UASpeech(Base):

    async def prepare_dataset(
        self,
        source_path: Path,
        allow_incomplete_classification: bool,
        min_tasks: int | None,
    ) -> ProcessedDataset:
        db_name = "uaspeech"
        db_path = f"{source_path}/{db_name}"
        sessions = []
        data_path = f"{db_path}/UASpeech/audio/original/"
        df = [
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

        for data in df:
            speaker_id = data["id"]
            diagnosis = data["diagnosis"].lower()
            diagnosis = self.diagnosis_map.get(diagnosis)
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
        return self.database_generator.generate(
            db_name=db_name,
            sessions=sessions,
        )

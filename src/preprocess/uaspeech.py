from pathlib import Path
from .base import BaseProcessor
from .processed import ProcessedFile, ProcessedSession


class UASpeech(BaseProcessor):
    async def __call__(self, source_path: Path, output_path: Path) -> None:
        db_key = "uaspeech"
        sessions = []
        data_path = f"{source_path}/UASpeech/audio/original/"
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
            age = int(data["age"]) if data["age"] is not None else None
            gender = data["gender"].strip()
            session = ProcessedSession(
                id=speaker_id,
                age=age,
                gender=gender,
                diagnosis=[self.diagnosis_map.get(diagnosis)],
                files=[
                    ProcessedFile(path=path)
                    for path in Path(f"{data_path}/{speaker_id}").glob("*.wav")
                    if not path.name.startswith(".")
                ],
            )
            sessions += [session]
        await self.process(output_path=output_path, db=db_key, sessions=sessions)

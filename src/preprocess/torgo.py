from pathlib import Path
from .base import BaseProcessor
from .processed import ProcessedFile, ProcessedSession


class Torgo(BaseProcessor):
    ignore_files = [
        "FC01/Session1/wav_arrayMic/0256.wav",  # 0 length audio
    ]

    async def __call__(self, source_path: Path, output_path: Path) -> None:
        db_key = "torgo"
        sessions = []
        data_path = f"{source_path}"
        df = [
            {
                "id": "F01",
                "diagnosis": "without dysarthria",
                "gender": "F",
                "age": None,
            },
            {
                "id": "F03",
                "diagnosis": "without dysarthria",
                "gender": "F",
                "age": None,
            },
            {
                "id": "F04",
                "diagnosis": "without dysarthria",
                "gender": "F",
                "age": None,
            },
            {"id": "FC01", "diagnosis": "dysarthria", "gender": "F", "age": 28},
            {"id": "FC02", "diagnosis": "dysarthria", "gender": "F", "age": 24},
            {"id": "FC03", "diagnosis": "dysarthria", "gender": "F", "age": 21},
            {
                "id": "M01",
                "diagnosis": "without dysarthria",
                "gender": "M",
                "age": None,
            },
            {"id": "M02", "diagnosis": "without dysarthria", "gender": "M", "age": 57},
            {
                "id": "M03",
                "diagnosis": "without dysarthria",
                "gender": "M",
                "age": None,
            },
            {
                "id": "M04",
                "diagnosis": "without dysarthria",
                "gender": "M",
                "age": None,
            },
            {
                "id": "M05",
                "diagnosis": "without dysarthria",
                "gender": "M",
                "age": None,
            },
            {"id": "MC01", "diagnosis": "dysarthria", "gender": "M", "age": 20},
            {"id": "MC02", "diagnosis": "dysarthria", "gender": "M", "age": 26},
            {"id": "MC03", "diagnosis": "dysarthria", "gender": "M", "age": 29},
            {"id": "MC04", "diagnosis": "dysarthria", "gender": "M", "age": None},
        ]

        for data in df:
            speaker_id = data["id"]
            diagnosis = data["diagnosis"].lower()
            speaker_path = Path(f"{data_path}/{speaker_id}")
            age = int(data["age"]) if data["age"] is not None else None
            gender = data["gender"].strip()
            for session in speaker_path.glob("Session*"):
                sessions += [
                    ProcessedSession(
                        id=f"{speaker_id}.{session.name}",
                        age=age,
                        gender=gender,
                        diagnosis=[self.diagnosis_map.get(diagnosis)],
                        files=[
                            ProcessedFile(path=path)
                            for path in Path(f"{session}/wav_arrayMic").glob("*.wav")
                            if self.include(path)
                        ],
                    )
                ]
        await self.process(output_path=output_path, db=db_key, sessions=sessions)

    def include(self, path: Path):
        for exclusion in self.ignore_files:
            if exclusion in str(path):
                return False
        return True

import json
from pathlib import Path
from .base import BaseProcessor
from .processed import ProcessedFile, ProcessedSession


class SVD(BaseProcessor):
    async def __call__(self, source_path: Path, output_path: Path) -> None:
        db_key = "svd"
        sessions = []

        with open(f"{source_path}/data.json", "r") as inputfile:
            for speaker_id, val in json.loads(inputfile.read()).items():
                gender = val["gender"].strip()
                for session in val["sessions"]:
                    session_id = session["session_id"]
                    age = int(session["age"])
                    classification = session["classification"]
                    pathologies = session["pathologies"]
                    files = session["files"]
                    diagnosis = pathologies if pathologies != "" else classification
                    sessions += [
                        ProcessedSession(
                            id=f"{speaker_id}.{session_id}",
                            age=age,
                            gender=gender,
                            diagnosis=[
                                self.diagnosis_map.get(x.strip().lower())
                                for x in diagnosis.split(",")
                            ],
                            files=[
                                ProcessedFile(
                                    path=Path(
                                        f"{source_path}/{classification}/{gender}/{speaker_id}/{session_id}/{file.split('file=')[1]}.wav"
                                    )
                                )
                                for file in files
                            ],
                        )
                    ]
        await self.process(output_path=output_path, db=db_key, sessions=sessions)

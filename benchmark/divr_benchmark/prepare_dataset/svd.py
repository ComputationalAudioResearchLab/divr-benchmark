import json
from pathlib import Path
from .base import BaseProcessor
from .processed import ProcessedFile, ProcessedSession


class SVD(BaseProcessor):
    ignore_files = [
        "1405/713/713-iau.wav",  # invalid file
        "1405/713/713-i_n.wav",  # invalid file
    ]

    async def __call__(self, source_path: Path, output_path: Path) -> None:
        db_key = "svd"
        sessions = []

        with open(f"{source_path}/data.json", "r") as inputfile:
            for speaker_id, val in json.loads(inputfile.read()).items():
                gender = val["gender"].strip()
                for session in val["sessions"]:
                    session = self.process_session(
                        speaker_id, gender, source_path, session
                    )
                    if session is not None:
                        sessions += [session]
        await self.process(output_path=output_path, db_name=db_key, sessions=sessions)

    def process_session(self, speaker_id, gender, source_path, session):
        session_id = session["session_id"]
        age = int(session["age"])
        classification = session["classification"]
        pathologies = session["pathologies"]
        files = session["files"]
        diagnosis = pathologies if pathologies != "" else classification
        files = []
        for file in session["files"]:
            path = Path(
                f"{source_path}/{classification}/{gender}/{speaker_id}/{session_id}/{file.split('file=')[1]}.wav"
            )
            if self.include(path):
                files += [ProcessedFile(path=path)]
        if len(files) == 0:
            return None
        return ProcessedSession(
            id=f"svd.{speaker_id}.{session_id}",
            age=age,
            gender=gender,
            diagnosis=[
                self.diagnosis_map.get(x.strip().lower()) for x in diagnosis.split(",")
            ],
            files=files,
        )

    def include(self, path: Path):
        for exclusion in self.ignore_files:
            if exclusion in str(path):
                return False
        return True

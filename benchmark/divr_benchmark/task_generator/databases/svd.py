import json
from pathlib import Path
from typing import List
from .Base import Base
from .gender import Gender
from ...prepare_dataset.processed import (
    ProcessedDataset,
    ProcessedSession,
    ProcessedFile,
)


class SVD(Base):
    ignore_files = [
        "1405/713/713-iau.wav",  # invalid file
        "1405/713/713-i_n.wav",  # invalid file
    ]

    def test_set_neutral_vowels(self, level: int):
        def filter_func(files: List[ProcessedFile]) -> List[ProcessedFile]:
            return list(filter(lambda x: str(x.path).endswith("_n.wav"), files))

        return self.to_tasks(
            self.dataset.test_sessions, level=level, file_filter=filter_func
        )

    def test_set_connected_speech(self, level: int):
        def filter_func(files: List[ProcessedFile]) -> List[ProcessedFile]:
            return list(filter(lambda x: str(x.path).endswith("-phrase.wav"), files))

        return self.to_tasks(
            self.dataset.test_sessions, level=level, file_filter=filter_func
        )

    def prepare_dataset(self, source_path: Path) -> ProcessedDataset:
        db_name = "svd"
        sessions = []
        with open(f"{source_path}/{db_name}/data.json", "r") as inputfile:
            for speaker_id, val in json.loads(inputfile.read()).items():
                gender = Gender.format(val["gender"])
                for session in val["sessions"]:
                    session = self.__process_session(
                        speaker_id, gender, f"{source_path}/{db_name}", session
                    )
                    if session is not None:
                        sessions += [session]
        return self.database_generator.generate(
            db_name=db_name,
            sessions=sessions,
        )

    def __process_session(self, speaker_id, gender, source_path, session):
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
            if self.__include(path):
                files += [ProcessedFile(path=path)]
        if len(files) == 0:
            return None
        return ProcessedSession(
            id=f"svd_{speaker_id}_{session_id}",
            age=age,
            gender=gender,
            diagnosis=[
                self.diagnosis_map.get(x.strip().lower()) for x in diagnosis.split(",")
            ],
            files=files,
        )

    def __include(self, path: Path):
        for exclusion in self.ignore_files:
            if exclusion in str(path):
                return False
        return True

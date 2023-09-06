import re
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
        options = [
            # ("svd", None, None),
            ("svd_phrase", r".*-phrase", None),
            ("svd_i_n", r".*-i_n", None),
            ("svd_i_n_male", r".*-i_n", "male"),
            ("svd_i_n_female", r".*-i_n", "female"),
            ("svd_a_n", r".*-a_n", None),
            ("svd_a_n_male", r".*-a_n", "male"),
            ("svd_a_n_female", r".*-a_n", "female"),
            ("svd_u_n", r".*-u_n", None),
            ("svd_u_n_male", r".*-u_n", "male"),
            ("svd_u_n_female", r".*-u_n", "female"),
        ]
        for db_key, file_filter, gender_filter in options:
            await self.setup_db(
                db_key=db_key,
                source_path=source_path,
                output_path=output_path,
                file_filter=file_filter,
                gender_filter=gender_filter,
            )

    async def setup_db(
        self,
        db_key,
        source_path: Path,
        output_path: Path,
        file_filter: str | None,
        gender_filter: str | None,
    ):
        sessions = []

        with open(f"{source_path}/data.json", "r") as inputfile:
            for speaker_id, val in json.loads(inputfile.read()).items():
                gender = val["gender"].strip()
                if gender_filter is None or gender == gender_filter:
                    for session in val["sessions"]:
                        session = self.process_session(
                            speaker_id, gender, source_path, session, file_filter
                        )
                        if session is not None:
                            sessions += [session]
        await self.process(output_path=output_path, db=db_key, sessions=sessions)

    def process_session(self, speaker_id, gender, source_path, session, file_filter):
        session_id = session["session_id"]
        age = int(session["age"])
        classification = session["classification"]
        pathologies = session["pathologies"]
        files = session["files"]
        diagnosis = pathologies if pathologies != "" else classification
        files = []
        for file in session["files"]:
            file_name = file.split("file=")[1]
            path = Path(
                f"{source_path}/{classification}/{gender}/{speaker_id}/{session_id}/{file_name}.wav"
            )
            if self.include(path, file_filter):
                files += [ProcessedFile(path=path)]
        if len(files) == 0:
            return None
        return ProcessedSession(
            id=f"{speaker_id}.{session_id}",
            age=age,
            gender=gender,
            diagnosis=[
                self.diagnosis_map.get(x.strip().lower()) for x in diagnosis.split(",")
            ],
            files=files,
        )

    def include(self, path: Path, file_filter: str | None):
        for exclusion in self.ignore_files:
            if exclusion in str(path):
                return False
        if file_filter is not None:
            return bool(re.match(file_filter, path.name))
        return True

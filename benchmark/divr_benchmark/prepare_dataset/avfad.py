import pandas as pd
from pathlib import Path
from .base import BaseProcessor
from .processed import ProcessedFile, ProcessedSession


class AVFAD(BaseProcessor):
    ignore_files = [
        "PLS007",  # 0 length audio
    ]

    async def __call__(self, source_path: Path, output_path: Path) -> None:
        db_key = "avfad"
        sessions = []
        df = pd.read_excel(f"{source_path}/AVFAD_01_00_00_1_README/AVFAD_01_00_00.xlsx")
        df = df[["File ID", "CMVD-I Dimension 1 (word system)", "Sex", "Age"]]

        def clean_diagnosis(diagnosis):
            diagnosis = diagnosis.lower().strip()
            diagnosis = diagnosis.replace("–", "-")
            diagnosis = diagnosis.replace("’", "'")
            return diagnosis

        for _, row in df.iterrows():
            speaker_id = row["File ID"]
            diagnosis = clean_diagnosis(row["CMVD-I Dimension 1 (word system)"])
            age = int(row["Age"])
            gender = row["Sex"].strip()
            sessions += [
                ProcessedSession(
                    id=speaker_id,
                    speaker_id=speaker_id,
                    age=age,
                    gender=gender,
                    diagnosis=[self.diagnosis_map.get(diagnosis)],
                    files=[
                        ProcessedFile(path=path)
                        for path in Path(source_path).rglob(f"{speaker_id}*.wav")
                        if self.include(path)
                    ],
                )
            ]
        await self.process(output_path=output_path, db_name=db_key, sessions=sessions)

    def include(self, path: Path):
        for exclusion in self.ignore_files:
            if exclusion in str(path):
                return False
        return True

import pandas as pd
from pathlib import Path
from .base import BaseProcessor
from .processed import ProcessedFile, ProcessedSession, ProcessedDataset


class AVFAD(BaseProcessor):
    async def __call__(self, source_path: Path) -> ProcessedDataset:
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
            sessions += [
                ProcessedSession(
                    id=speaker_id,
                    age=row["Age"],
                    gender=row["Sex"],
                    diagnosis=[self.diagnosis_map.get(diagnosis)],
                    files=[
                        ProcessedFile(path=path)
                        for path in Path(source_path).rglob(f"{speaker_id}*.wav")
                    ],
                )
            ]
        return ProcessedDataset(db=db_key, sessions=sessions)

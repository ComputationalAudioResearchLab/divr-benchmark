from pathlib import Path
import pandas as pd
from .Base import Base
from .gender import Gender
from ...prepare_dataset.processed import (
    ProcessedDataset,
    ProcessedSession,
    ProcessedFile,
)


class AVFAD(Base):
    ignore_files = [
        "PLS007",  # 0 length audio
    ]

    async def prepare_dataset(
        self,
        source_path: Path,
        allow_incomplete_classification: bool,
        min_tasks: int | None,
    ) -> ProcessedDataset:
        db_name = "avfad"
        db_path = Path(f"{source_path}/{db_name}")
        sessions = []
        df = pd.read_excel(f"{db_path}/AVFAD_01_00_00_1_README/AVFAD_01_00_00.xlsx")
        df = df[["File ID", "CMVD-I Dimension 1 (word system)", "Sex", "Age"]]
        for _, row in df.iterrows():
            speaker_id = row["File ID"]
            age = int(row["Age"])
            gender = Gender.format(row["Sex"].strip())
            diagnosis = self.__clean_diagnosis(row["CMVD-I Dimension 1 (word system)"])
            diagnosis = self.diagnosis_map.get(diagnosis)
            if allow_incomplete_classification or not diagnosis.incompletely_classified:
                file_paths = [
                    path
                    for path in Path(db_path).rglob(f"{speaker_id}*.wav")
                    if self.__include(path)
                ]
                num_files = len(file_paths)
                if min_tasks is None or num_files >= min_tasks:
                    sessions += [
                        ProcessedSession(
                            id=f"avfad_{speaker_id}",
                            speaker_id=speaker_id,
                            age=age,
                            gender=gender,
                            diagnosis=[diagnosis],
                            files=[ProcessedFile(path=path) for path in file_paths],
                            num_files=num_files,
                        )
                    ]

        return self.database_generator.generate(
            db_name=db_name,
            sessions=sessions,
        )

    def __clean_diagnosis(self, diagnosis: str) -> str:
        diagnosis = diagnosis.lower().strip()
        diagnosis = diagnosis.replace("–", "-")
        diagnosis = diagnosis.replace("’", "'")
        return diagnosis

    def __include(self, path: Path) -> bool:
        for exclusion in self.ignore_files:
            if exclusion in str(path):
                return False
        return True

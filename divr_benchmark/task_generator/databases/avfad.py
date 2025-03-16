import pandas as pd
from pathlib import Path
from typing import List, Set
from divr_diagnosis import DiagnosisMap

from .Base import Base
from .gender import Gender
from ...prepare_dataset.processed import (
    ProcessedSession,
    ProcessedFile,
)


class AVFAD(Base):
    DB_NAME = "avfad"
    __ignore_files = [
        "PLS007",  # 0 length audio
    ]

    async def _collect_diagnosis_terms(self, source_path: Path) -> Set[str]:
        df = self.__read_data(source_path=source_path)
        return set(df["CMVD-I Dimension 1 (word system)"].tolist())

    async def prepare_dataset(
        self,
        source_path: Path,
        allow_incomplete_classification: bool,
        min_tasks: int | None,
        diagnosis_map: DiagnosisMap,
    ) -> List[ProcessedSession]:
        sessions = []
        df = self.__read_data(source_path)
        for _, row in df.iterrows():
            speaker_id = row["File ID"]
            age = int(row["Age"])
            gender = Gender.format(row["Sex"].strip())
            diag_term = row["CMVD-I Dimension 1 (word system)"]
            diagnosis = (
                diagnosis_map[diag_term]
                if diag_term in diagnosis_map
                else diagnosis_map.unclassified
            )
            if allow_incomplete_classification or not diagnosis.incompletely_classified:
                file_paths = [
                    path
                    for path in Path(source_path).rglob(f"{speaker_id}*.wav")
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

        return sessions

    def __read_data(self, source_path):
        df = pd.read_excel(f"{source_path}/AVFAD_01_00_00_1_README/AVFAD_01_00_00.xlsx")
        df = df[["File ID", "CMVD-I Dimension 1 (word system)", "Sex", "Age"]]
        df["CMVD-I Dimension 1 (word system)"] = df[
            "CMVD-I Dimension 1 (word system)"
        ].apply(self.__clean_diagnosis)
        return df

    def __clean_diagnosis(self, diagnosis: str) -> str:
        diagnosis = diagnosis.lower().strip()
        diagnosis = diagnosis.replace("–", "-")
        diagnosis = diagnosis.replace("’", "'")
        return diagnosis

    def __include(self, path: Path) -> bool:
        for exclusion in self.__ignore_files:
            if exclusion in str(path):
                return False
        return True

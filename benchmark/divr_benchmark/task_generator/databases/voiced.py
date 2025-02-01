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


class Voiced(Base):
    DB_NAME = "voiced"

    async def _collect_diagnosis_terms(self, source_path: Path) -> Set[str]:
        _, df = self.__read_data(source_path=source_path)
        return set(df["Diagnosis"].tolist())

    async def prepare_dataset(
        self,
        source_path: Path,
        allow_incomplete_classification: bool,
        min_tasks: int | None,
        diagnosis_map: DiagnosisMap,
    ) -> List[ProcessedSession]:
        sessions = []
        data_path, all_data = self.__read_data(source_path)

        for _, row in all_data.iterrows():
            speaker_id = row["ID"]
            diagnosis = row["Diagnosis"]
            diagnosis = diagnosis_map.get(diagnosis)
            age = int(row["Age"])
            gender = Gender.format(row["Gender"])
            if allow_incomplete_classification or not diagnosis.incompletely_classified:
                num_files = 1
                if min_tasks is None or num_files >= min_tasks:
                    sessions += [
                        ProcessedSession(
                            id=f"voiced_{speaker_id}",
                            speaker_id=speaker_id,
                            age=age,
                            gender=gender,
                            diagnosis=[diagnosis],
                            files=[
                                ProcessedFile(
                                    path=Path(f"{data_path}/{speaker_id}.wav")
                                )
                            ],
                            num_files=num_files,
                        )
                    ]
        return sessions

    def __read_data(self, source_path):
        data_path = f"{source_path}/voice-icar-federico-ii-database-1.0.0"

        info_files = list(Path(data_path).rglob("*-info.txt"))
        rows = []
        for ifile in info_files:
            df = pd.read_csv(ifile, delimiter="\t", header=None)
            df.dropna(how="all", inplace=True)
            row = pd.Series(
                list(df[1]), index=df[0].apply(lambda x: x.replace(":", ""))
            )
            row = self.__fix_errors(ifile, row)
            rows += [row]
        all_data = pd.DataFrame(rows)
        all_data = all_data[["ID", "Diagnosis", "Gender", "Age"]]
        all_data["Diagnosis"] = all_data["Diagnosis"].str.lower().str.strip()
        return data_path, all_data

    def __fix_errors(self, ifile: Path, row: pd.Series) -> pd.Series:
        """
        Used for fixing errors in the DB
        """
        filekey = ifile.stem.removesuffix("-info")
        if row["ID"] != filekey:
            print(f"Info: Fixing DB error where original ID={row['ID']}, ifile={ifile}")
            row["ID"] = filekey
        return row

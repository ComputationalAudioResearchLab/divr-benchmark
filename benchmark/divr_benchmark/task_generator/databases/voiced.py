from pathlib import Path
import pandas as pd
from .Base import Base
from .gender import Gender
from ...prepare_dataset.processed import (
    ProcessedDataset,
    ProcessedSession,
    ProcessedFile,
)


class Voiced(Base):
    def prepare_dataset(
        self,
        source_path: Path,
        allow_incomplete_classification: bool,
        min_tasks: int|None,
    ) -> ProcessedDataset:
        db_name = "voiced"
        sessions = []
        data_path = f"{source_path}/{db_name}/voice-icar-federico-ii-database-1.0.0"

        info_files = list(Path(source_path).rglob("*-info.txt"))
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

        for _, row in all_data.iterrows():
            speaker_id = row["ID"]
            diagnosis = row["Diagnosis"].lower().strip()
            diagnosis = self.diagnosis_map.get(diagnosis)
            age = int(row["Age"])
            gender = Gender.format(row["Gender"])
            if allow_incomplete_classification or not diagnosis.incompletely_classified:
                num_files = 1
                if min_tasks is None or num_files >= min_tasks:
                    sessions += [
                        ProcessedSession(
                            id=f"voiced_{speaker_id}",
                            age=age,
                            gender=gender,
                            diagnosis=[diagnosis],
                            files=[
                                ProcessedFile(path=Path(f"{data_path}/{speaker_id}.wav"))
                            ],
                            num_files=num_files,
                        )
                    ]
        return self.database_generator.generate(
            db_name=db_name,
            sessions=sessions,
        )

    def __fix_errors(self, ifile: Path, row: pd.Series) -> pd.Series:
        """
        Used for fixing errors in the DB
        """
        filekey = ifile.stem.removesuffix("-info")
        if row["ID"] != filekey:
            print(f"Info: Fixing DB error where original ID={row['ID']}, ifile={ifile}")
            row["ID"] = filekey
        return row

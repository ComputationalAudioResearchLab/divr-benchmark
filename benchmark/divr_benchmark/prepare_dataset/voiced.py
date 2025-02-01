import pandas as pd
from pathlib import Path
from .base import BaseProcessor
from .processed import ProcessedFile, ProcessedSession
from ..diagnosis import DiagnosisMap


class Voiced(BaseProcessor):
    def __init__(self, diagnosis_map: DiagnosisMap) -> None:
        super().__init__(diagnosis_map=diagnosis_map)

    async def __call__(self, source_path: Path, output_path: Path) -> None:
        db_key = "voiced"
        sessions = []
        data_path = f"{source_path}/voice-icar-federico-ii-database-1.0.0"

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
            age = int(row["Age"])
            gender = row["Gender"].strip()
            sessions += [
                ProcessedSession(
                    id=f"voiced.{speaker_id}",
                    speaker_id=speaker_id,
                    age=age,
                    gender=gender,
                    diagnosis=[self.diagnosis_map.get(diagnosis)],
                    files=[ProcessedFile(path=Path(f"{data_path}/{speaker_id}.wav"))],
                )
            ]
        await self.process(output_path=output_path, db_name=db_key, sessions=sessions)

    def __fix_errors(self, ifile: Path, row: pd.Series) -> pd.Series:
        """
        Used for fixing errors in the DB
        """
        filekey = ifile.stem.removesuffix("-info")
        if row["ID"] != filekey:
            print(f"Info: Fixing DB error where original ID={row['ID']}, ifile={ifile}")
            row["ID"] = filekey
        return row

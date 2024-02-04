import pandas as pd
from pathlib import Path
from .base import BaseProcessor
from .processed import ProcessedFile, ProcessedSession


class Voiced(BaseProcessor):
    def __init__(self, audio_extraction_path: Path) -> None:
        super().__init__()
        self.audio_extraction_path = audio_extraction_path

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
                    id=speaker_id,
                    age=age,
                    gender=gender,
                    diagnosis=[self.diagnosis_map.get(diagnosis)],
                    files=[
                        await ProcessedFile.from_wfdb(
                            dat_path=Path(f"{data_path}/{speaker_id}"),
                            extraction_path=self.audio_extraction_path,
                        )
                    ],
                )
            ]
        await self.process(output_path=output_path, db_name=db_key, sessions=sessions)

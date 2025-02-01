import pandas as pd
from pathlib import Path
from divr_diagnosis import DiagnosisMap

from .base import BaseProcessor
from .processed import ProcessedFile, ProcessedSession


class MEEI(BaseProcessor):
    def __init__(
        self, audio_extraction_path: Path, diagnosis_map: DiagnosisMap
    ) -> None:
        super().__init__(diagnosis_map=diagnosis_map)
        self.audio_extraction_path = audio_extraction_path

    async def __call__(self, source_path: Path, output_path: Path) -> None:
        db_key = "meei"
        sessions = []
        excel_path = f"{source_path}/kaylab/data/disorderedvoicedb/EXCEL50/KAYCD_DB.XLS"
        file_key = "FILE VOWEL 'AH'"

        def collate_data(df, is_pathological):
            grouped = df.groupby(file_key)
            sex = grouped["SEX"].apply(lambda x: set(x).pop() if len(x) > 0 else None)
            age = grouped["AGE"].apply(lambda x: set(x).pop() if len(x) > 0 else None)
            diagnosis = grouped["DIAGNOSIS"].apply(lambda x: ",".join(set(x)))
            df = pd.concat([diagnosis, sex, age], axis=1)
            df = df.reset_index()
            df["pathological"] = is_pathological
            return df

        def clean_white_spaces(df):
            clean_columns = map(str.strip, df.columns)
            df.columns = clean_columns
            df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            return df

        df_path = clean_white_spaces(
            pd.read_excel(
                excel_path,
                sheet_name="Pathological",
                nrows=654,
            )
        )
        df_norm = clean_white_spaces(
            pd.read_excel(excel_path, sheet_name="Normal", nrows=53)
        )
        df_full = clean_white_spaces(
            pd.read_excel(excel_path, sheet_name="Full Database")
        )
        path_files = df_path[file_key]
        norm_files = df_norm[file_key]

        path_data = collate_data(df_full[df_full[file_key].isin(path_files)], True)
        norm_data = collate_data(df_full[df_full[file_key].isin(norm_files)], False)
        full_data = pd.concat([path_data, norm_data])

        for _, row in full_data.iterrows():
            speaker_id = row[file_key][:5]
            diagnosis = []
            for x in row["DIAGNOSIS"].split(","):
                x = x.lower().strip()
                if len(x) > 0:
                    diagnosis += [self.diagnosis_map.get(x)]
            if len(diagnosis) == 0:
                diagnosis = [self.diagnosis_map.get("unknown")]
            age = int(row["AGE"]) if row["AGE"] != "" else None
            gender = row["SEX"].strip()
            files = [
                await ProcessedFile.from_nsp(
                    nsp_path=path, extraction_path=self.audio_extraction_path
                )
                for path in Path(source_path).rglob(f"{speaker_id}*.NSP")
            ]
            sessions += [
                ProcessedSession(
                    id=speaker_id,
                    speaker_id=speaker_id,
                    age=age,
                    gender=gender,
                    diagnosis=diagnosis,
                    files=files,
                    num_files=len(files),
                )
            ]
        await self.process(output_path=output_path, db_name=db_key, sessions=sessions)

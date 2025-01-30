from pathlib import Path
import pandas as pd
from .Base import Base
from .gender import Gender
from ...prepare_dataset.processed import (
    ProcessedDataset,
    ProcessedSession,
    ProcessedFile,
)


class MEEI(Base):

    async def prepare_dataset(
        self,
        source_path: Path,
        allow_incomplete_classification: bool,
        min_tasks: int | None,
    ) -> ProcessedDataset:
        db_name = "meei"
        db_path = Path(f"{source_path}/{db_name}")
        sessions = []
        excel_path = f"{db_path}/kaylab/data/disorderedvoicedb/EXCEL50/KAYCD_DB.XLS"
        audio_extraction_path = Path(f"{db_path}/.extracted")
        audio_extraction_path.mkdir(exist_ok=True)
        file_key = "FILE VOWEL 'AH'"

        df_path = self.__clean_white_spaces(
            pd.read_excel(
                excel_path,
                sheet_name="Pathological",
                nrows=654,
            )
        )
        df_norm = self.__clean_white_spaces(
            pd.read_excel(excel_path, sheet_name="Normal", nrows=53)
        )
        df_full = self.__clean_white_spaces(
            pd.read_excel(excel_path, sheet_name="Full Database")
        )
        path_files = df_path[file_key]
        norm_files = df_norm[file_key]

        path_data = self.__collate_data(
            df=df_full[df_full[file_key].isin(path_files)],
            is_pathological=True,
            file_key=file_key,
        )
        norm_data = self.__collate_data(
            df=df_full[df_full[file_key].isin(norm_files)],
            is_pathological=False,
            file_key=file_key,
        )
        full_data = pd.concat([path_data, norm_data])
        for _, row in full_data.iterrows():
            speaker_id = row[file_key][:5]
            diagnosis = []
            has_incomplete_diagnosis = False
            for x in row["DIAGNOSIS"].split(","):
                x = x.lower().strip()
                if len(x) > 0:
                    d = self.diagnosis_map.get(x)
                    diagnosis += [d]
                    if d.incompletely_classified:
                        has_incomplete_diagnosis = True
            if len(diagnosis) == 0:
                diagnosis = [self.diagnosis_map.get("unknown")]
                has_incomplete_diagnosis = True
            if allow_incomplete_classification or not has_incomplete_diagnosis:
                age = int(row["AGE"]) if row["AGE"] != "" else None
                gender = Gender.format(row["SEX"].strip())
                file_paths = list(Path(db_path).rglob(f"{speaker_id}*.NSP"))
                num_files = len(file_paths)
                if min_tasks is None or num_files >= min_tasks:
                    sessions += [
                        ProcessedSession(
                            id=f"meei_{speaker_id}",
                            speaker_id=speaker_id,
                            age=age,
                            gender=gender,
                            diagnosis=diagnosis,
                            files=[
                                await ProcessedFile.from_nsp(
                                    nsp_path=path,
                                    extraction_path=audio_extraction_path,
                                )
                                for path in file_paths
                            ],
                            num_files=num_files,
                        )
                    ]
        return self.database_generator.generate(
            db_name=db_name,
            sessions=sessions,
        )

    def __collate_data(self, df: pd.DataFrame, is_pathological: bool, file_key: str):
        grouped = df.groupby(file_key)
        sex = grouped["SEX"].apply(lambda x: set(x).pop() if len(x) > 0 else None)
        age = grouped["AGE"].apply(lambda x: set(x).pop() if len(x) > 0 else None)
        diagnosis = grouped["DIAGNOSIS"].apply(lambda x: ",".join(set(x)))
        df = pd.concat([diagnosis, sex, age], axis=1)
        df = df.reset_index()
        df["pathological"] = is_pathological
        return df

    def __clean_white_spaces(self, df: pd.DataFrame):
        clean_columns = list(map(str.strip, df.columns))
        df.columns = clean_columns
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        return df

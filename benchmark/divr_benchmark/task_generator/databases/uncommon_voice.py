import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Set

from .Base import Base
from .gender import Gender
from ...prepare_dataset.processed import (
    ProcessedSession,
    ProcessedFile,
)
from ...diagnosis import DiagnosisMap


class UncommonVoice(Base):
    DB_NAME = "uncommon_voice"

    async def _collect_diagnosis_terms(self, source_path: Path) -> Set[str]:
        _, df = self.__read_data(source_path=source_path)
        return set(df["Voice Disorder"].tolist())

    async def prepare_dataset(
        self,
        source_path: Path,
        allow_incomplete_classification: bool,
        min_tasks: int | None,
        diagnosis_map: DiagnosisMap,
    ) -> List[ProcessedSession]:
        sessions = []
        data_path, df = self.__read_data(source_path)
        for row_idx, data in df.iterrows():
            speaker_id = data["new_ID"]
            if speaker_id is None:
                speaker_id = str(row_idx)
            diagnosis = data["Voice Disorder"]
            diagnosis = diagnosis_map.get(diagnosis)
            if allow_incomplete_classification or not diagnosis.incompletely_classified:
                age = None
                gender = Gender.format(data["Gender"].strip())
                file_paths = list(Path(data_path).glob(f"{speaker_id}_*.wav"))
                num_files = len(file_paths)
                if min_tasks is None or num_files >= min_tasks:
                    session = ProcessedSession(
                        id=f"uncommon_voice_{speaker_id}",
                        speaker_id=speaker_id,
                        age=age,
                        gender=gender,
                        diagnosis=[diagnosis],
                        files=[ProcessedFile(path=path) for path in file_paths],
                        num_files=num_files,
                    )
                    sessions += [session]
        return sessions

    def __read_data(self, source_path):
        data_path = f"{source_path}/UncommonVoice/UncommonVoice_final"
        df = pd.read_csv(f"{source_path}/uncommonvoice_user_data.csv").replace(
            {np.nan: None}
        )
        df = df[["new_ID", "Voice Disorder", "Gender"]]
        df["Voice Disorder"] = df["Voice Disorder"].apply(
            lambda x: "normal" if x == 0 else "pathological"
        )
        return data_path, df

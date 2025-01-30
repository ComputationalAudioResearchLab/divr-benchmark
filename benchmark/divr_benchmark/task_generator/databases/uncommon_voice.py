import numpy as np
import pandas as pd
from pathlib import Path

from .Base import Base
from .gender import Gender
from ...prepare_dataset.processed import (
    ProcessedDataset,
    ProcessedSession,
    ProcessedFile,
)


class UncommonVoice(Base):

    async def prepare_dataset(
        self,
        source_path: Path,
        allow_incomplete_classification: bool,
        min_tasks: int | None,
    ) -> ProcessedDataset:
        db_name = "uncommon_voice"
        db_path = f"{source_path}/{db_name}"
        sessions = []
        data_path = f"{db_path}/UncommonVoice/UncommonVoice_final"
        df = pd.read_csv(f"{db_path}/uncommonvoice_user_data.csv").replace(
            {np.nan: None}
        )
        df = df[["new_ID", "Voice Disorder", "Gender"]]
        for row_idx, data in df.iterrows():
            speaker_id = data["new_ID"]
            if speaker_id is None:
                speaker_id = str(row_idx)
            diagnosis = "normal" if data["Voice Disorder"] == 0 else "pathological"
            diagnosis = self.diagnosis_map.get(diagnosis)
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
        return self.database_generator.generate(
            db_name=db_name,
            sessions=sessions,
        )

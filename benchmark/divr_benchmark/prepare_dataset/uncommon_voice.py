import numpy as np
import pandas as pd
from pathlib import Path
from .base import BaseProcessor
from .processed import ProcessedFile, ProcessedSession


class UncommonVoice(BaseProcessor):
    async def __call__(self, source_path: Path, output_path: Path) -> None:
        db_key = "uncommon_voice"
        sessions = []
        data_path = f"{source_path}/UncommonVoice_final"
        df = pd.read_csv(f"{source_path}/data.csv").replace({np.nan: None})
        df = df[["new_ID", "Voice Disorder", "Gender"]]
        for _, data in df.iterrows():
            speaker_id = data["new_ID"]
            diagnosis = "normal" if data["Voice Disorder"] == 0 else "pathological"
            age = None
            gender = data["Gender"].strip()
            files = [
                ProcessedFile(path=path)
                for path in Path(data_path).glob(f"{speaker_id}_*.wav")
            ]
            session = ProcessedSession(
                id=speaker_id,
                speaker_id=speaker_id,
                age=age,
                gender=gender,
                diagnosis=[self.diagnosis_map.get(diagnosis)],
                files=files,
                num_files=len(files),
            )
            sessions += [session]
        await self.process(output_path=output_path, db_name=db_key, sessions=sessions)

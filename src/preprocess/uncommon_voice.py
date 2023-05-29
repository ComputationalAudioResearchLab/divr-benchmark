import pandas as pd
from pathlib import Path
from .base import BaseProcessor
from .processed import ProcessedFile, ProcessedSession, ProcessedDataset


class UncommonVoice(BaseProcessor):
    async def __call__(self, source_path: Path) -> ProcessedDataset:
        db_key = "uncommon_voice"
        sessions = []
        data_path = f"{source_path}/UncomonVoice_final"
        df = pd.read_csv(f"{source_path}/data.csv")
        df = df[["new_ID", "Voice Disorder", "Gender"]]
        for _, data in df.iterrows():
            speaker_id = data["new_ID"]
            diagnosis = "normal" if data["Voice Disorder"] == 0 else "pathological"
            session = ProcessedSession(
                id=speaker_id,
                age=None,
                gender=data["Gender"],
                diagnosis=[self.diagnosis_map.get(diagnosis)],
                files=[
                    ProcessedFile(path=path)
                    for path in Path(data_path).glob(f"{speaker_id}_*.wav")
                ],
            )
            sessions += [session]
        return ProcessedDataset(db=db_key, sessions=sessions)

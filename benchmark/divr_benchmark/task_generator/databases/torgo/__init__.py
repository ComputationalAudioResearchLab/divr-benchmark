import pandas as pd
from typing import List
from pathlib import Path
from ..Base import Base
from ..gender import Gender
from ....prepare_dataset.processed import (
    ProcessedDataset,
    ProcessedSession,
    ProcessedFile,
)
from ....diagnosis import DiagnosisMap


class Torgo(Base):
    ignore_files = [
        "FC01/Session1/wav_arrayMic/0256.wav",  # 0 length audio
    ]
    max_tasks = 142
    avg_tasks = 60
    min_tasks = 0

    def __init__(
        self,
        source_path: Path,
        allow_incomplete_classification: bool,
        min_tasks: int | None,
        diagnosis_map: DiagnosisMap,
    ) -> None:
        curdir = Path(__file__).parent.resolve()
        # only valid prompts that all speakers have spoken are selected
        self.__selected_prompts: List[str] = pd.read_csv(
            f"{curdir}/selected_prompts.csv"
        )["prompt"].to_list()

        super().__init__(
            source_path=source_path,
            allow_incomplete_classification=allow_incomplete_classification,
            min_tasks=min_tasks,
            diagnosis_map=diagnosis_map,
        )

    async def prepare_dataset(
        self,
        source_path: Path,
        allow_incomplete_classification: bool,
        min_tasks: int | None,
    ) -> ProcessedDataset:
        db_name = "torgo"
        sessions = []
        data_path = f"{source_path}/{db_name}"
        df = [
            {
                "id": "F01",
                "diagnosis": "without dysarthria",
                "gender": "F",
                "age": None,
            },
            {
                "id": "F03",
                "diagnosis": "without dysarthria",
                "gender": "F",
                "age": None,
            },
            {
                "id": "F04",
                "diagnosis": "without dysarthria",
                "gender": "F",
                "age": None,
            },
            {"id": "FC01", "diagnosis": "dysarthria", "gender": "F", "age": 28},
            {"id": "FC02", "diagnosis": "dysarthria", "gender": "F", "age": 24},
            {"id": "FC03", "diagnosis": "dysarthria", "gender": "F", "age": 21},
            {
                "id": "M01",
                "diagnosis": "without dysarthria",
                "gender": "M",
                "age": None,
            },
            {"id": "M02", "diagnosis": "without dysarthria", "gender": "M", "age": 57},
            {
                "id": "M03",
                "diagnosis": "without dysarthria",
                "gender": "M",
                "age": None,
            },
            {
                "id": "M04",
                "diagnosis": "without dysarthria",
                "gender": "M",
                "age": None,
            },
            {
                "id": "M05",
                "diagnosis": "without dysarthria",
                "gender": "M",
                "age": None,
            },
            {"id": "MC01", "diagnosis": "dysarthria", "gender": "M", "age": 20},
            {"id": "MC02", "diagnosis": "dysarthria", "gender": "M", "age": 26},
            {"id": "MC03", "diagnosis": "dysarthria", "gender": "M", "age": 29},
            {"id": "MC04", "diagnosis": "dysarthria", "gender": "M", "age": None},
        ]

        for data in df:
            speaker_id = data["id"]
            diagnosis = data["diagnosis"].lower()
            diagnosis = self.diagnosis_map.get(diagnosis)
            speaker_path = Path(f"{data_path}/{speaker_id}")
            age = int(data["age"]) if data["age"] is not None else None
            gender = Gender.format(data["gender"])
            if not diagnosis.incompletely_classified or allow_incomplete_classification:
                for session in speaker_path.glob("Session*"):
                    files = self.__select_files(session=session)
                    num_files = len(files)
                    if min_tasks is None or num_files >= min_tasks:
                        sessions += [
                            ProcessedSession(
                                id=f"torgo_{speaker_id}_{session.name}",
                                speaker_id=speaker_id,
                                age=age,
                                gender=gender,
                                diagnosis=[diagnosis],
                                files=files,
                                num_files=num_files,
                            )
                        ]
        return self.database_generator.generate(
            db_name=db_name,
            sessions=sessions,
        )

    def __select_files(self, session):
        files = []
        for path in Path(f"{session}/wav_arrayMic").glob("*.wav"):
            if self.__include(path) and self.__is_allowed_prompt(path):
                files.append(ProcessedFile(path=path))
        return files

    def __include(self, path: Path):
        for exclusion in self.ignore_files:
            if exclusion in str(path):
                return False
        return True

    def __is_allowed_prompt(self, wav_path: Path) -> bool:
        prompt_path = Path(
            str(wav_path).replace("wav_arrayMic", "prompts").replace(".wav", ".txt")
        )
        if not prompt_path.is_file():
            return False
        with open(prompt_path, "r") as prompt_file:
            line = prompt_file.readline()
            prompt = line.lower().replace('"', "").replace("'", "")
            return prompt in self.__selected_prompts

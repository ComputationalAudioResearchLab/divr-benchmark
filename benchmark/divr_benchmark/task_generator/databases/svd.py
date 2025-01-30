import json
from pathlib import Path
from typing import List, Literal
from .Base import Base
from .gender import Gender
from ...prepare_dataset.processed import (
    ProcessedDataset,
    ProcessedSession,
    ProcessedFile,
)


VOWELS = Literal["a", "i", "u", ""]


class SVD(Base):
    ignore_files = [
        "1405/713/713-iau.wav",  # invalid file
        "1405/713/713-i_n.wav",  # invalid file
    ]
    max_tasks = 14

    def train_set_multi_neutral_vowels(self, level: int, vowels: List[VOWELS]):
        return self.filtered_multi_file_tasks(
            self.dataset.train_sessions,
            level=level,
            suffixes=[f"{v}_n.wav" for v in vowels],
        )

    def val_set_multi_neutral_vowels(self, level: int, vowels: List[VOWELS]):
        return self.filtered_multi_file_tasks(
            self.dataset.val_sessions,
            level=level,
            suffixes=[f"{v}_n.wav" for v in vowels],
        )

    def test_set_multi_neutral_vowels(self, level: int, vowels: List[VOWELS]):
        return self.filtered_multi_file_tasks(
            self.dataset.test_sessions,
            level=level,
            suffixes=[f"{v}_n.wav" for v in vowels],
        )

    def train_set_combined_vowel_vocalisation(self, level: int):
        return self.filtered_single_file_tasks(
            self.dataset.train_sessions, level=level, suffix="iau.wav"
        )

    def val_set_combined_vowel_vocalisation(self, level: int):
        return self.filtered_single_file_tasks(
            self.dataset.val_sessions, level=level, suffix="iau.wav"
        )

    def test_set_combined_vowel_vocalisation(self, level: int):
        return self.filtered_single_file_tasks(
            self.dataset.test_sessions, level=level, suffix="iau.wav"
        )

    def train_set_lhl_vowels(self, level: int, vowel: VOWELS = ""):
        return self.filtered_single_file_tasks(
            self.dataset.train_sessions, level=level, suffix=f"{vowel}_lhl.wav"
        )

    def val_set_lhl_vowels(self, level: int, vowel: VOWELS = ""):
        return self.filtered_single_file_tasks(
            self.dataset.val_sessions, level=level, suffix=f"{vowel}_lhl.wav"
        )

    def test_set_lhl_vowels(self, level: int, vowel: VOWELS = ""):
        return self.filtered_single_file_tasks(
            self.dataset.test_sessions, level=level, suffix=f"{vowel}_lhl.wav"
        )

    def train_set_neutral_vowels(self, level: int, vowel: VOWELS = ""):
        return self.filtered_single_file_tasks(
            self.dataset.train_sessions, level=level, suffix=f"{vowel}_n.wav"
        )

    def val_set_neutral_vowels(self, level: int, vowel: VOWELS = ""):
        return self.filtered_single_file_tasks(
            self.dataset.val_sessions, level=level, suffix=f"{vowel}_n.wav"
        )

    def test_set_neutral_vowels(self, level: int, vowel: VOWELS = ""):
        return self.filtered_single_file_tasks(
            self.dataset.test_sessions, level=level, suffix=f"{vowel}_n.wav"
        )

    def train_set_connected_speech(self, level: int):
        return self.filtered_single_file_tasks(
            self.dataset.train_sessions, level=level, suffix="-phrase.wav"
        )

    def val_set_connected_speech(self, level: int):
        return self.filtered_single_file_tasks(
            self.dataset.val_sessions, level=level, suffix="-phrase.wav"
        )

    def test_set_connected_speech(self, level: int):
        return self.filtered_single_file_tasks(
            self.dataset.test_sessions, level=level, suffix="-phrase.wav"
        )

    def filtered_single_file_tasks(
        self, sessions: List[ProcessedSession], level: int, suffix: str
    ):
        def filter_func(files: List[ProcessedFile]) -> List[ProcessedFile]:
            return list(filter(lambda x: str(x.path).endswith(suffix), files))

        return self.to_individual_file_tasks(
            sessions, level=level, file_filter=filter_func
        )

    def filtered_multi_file_tasks(
        self, sessions: List[ProcessedSession], level: int, suffixes: List[str]
    ):
        def file_name_filter_func(processed_file: ProcessedFile):
            file_path_str = str(processed_file.path)
            for suffix in suffixes:
                if file_path_str.endswith(suffix):
                    return True
            return False

        def filter_func(files: List[ProcessedFile]) -> List[ProcessedFile]:
            return list(filter(file_name_filter_func, files))

        return self.to_multi_file_tasks(sessions, level=level, file_filter=filter_func)

    async def prepare_dataset(
        self,
        source_path: Path,
        allow_incomplete_classification: bool,
        min_tasks: int | None,
    ) -> ProcessedDataset:
        db_name = "svd"
        db_path = f"{source_path}/{db_name}"
        sessions = []
        with open(f"{db_path}/data.json", "r") as inputfile:
            for speaker_id, val in json.loads(inputfile.read()).items():
                gender = Gender.format(val["gender"])
                for session in val["sessions"]:
                    session = self.__process_session(
                        speaker_id=speaker_id,
                        gender=gender,
                        source_path=db_path,
                        session=session,
                        allow_incomplete_classification=allow_incomplete_classification,
                    )
                    if (session is not None) and (
                        min_tasks is None or session.num_files >= min_tasks
                    ):
                        sessions += [session]
        return self.database_generator.generate(
            db_name=db_name,
            sessions=sessions,
        )

    def __process_session(
        self,
        speaker_id,
        gender,
        source_path,
        session,
        allow_incomplete_classification: bool,
    ):
        session_id = session["session_id"]
        age = int(session["age"])
        classification = session["classification"]
        pathologies = session["pathologies"]
        files = session["files"]
        diagnosis = pathologies if pathologies != "" else classification
        files = []
        for file in session["files"]:
            path = Path(
                f"{source_path}/{classification}/{gender}/{speaker_id}/{session_id}/{file.split('file=')[1]}.wav"
            )
            if self.__include(path):
                files += [ProcessedFile(path=path)]
        if len(files) == 0:
            return None
        diagnosis = [
            self.diagnosis_map.get(x.strip().lower()) for x in diagnosis.split(",")
        ]
        session = ProcessedSession(
            id=f"svd_{speaker_id}_{session_id}",
            speaker_id=speaker_id,
            age=age,
            gender=gender,
            diagnosis=diagnosis,
            files=files,
            num_files=len(files),
        )
        if (
            session.best_diagnosis.incompletely_classified
            and not allow_incomplete_classification
        ):
            if session.id == "svd_2537_2403":
                print(session.best_diagnosis.name)
                # print([d.name for d in sorted_diagnosis])
                # print([d.name for d in complete_diagnosis])
                # print(complete_diagnosis[0].incompletely_classified)
                exit()
            return None
        return session

    def __include(self, path: Path):
        for exclusion in self.ignore_files:
            if exclusion in str(path):
                return False
        return True

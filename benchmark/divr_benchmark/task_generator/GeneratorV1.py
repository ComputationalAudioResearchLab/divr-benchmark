import asyncio
from pathlib import Path
from typing import Awaitable, Callable, Dict, List

from .generator import Generator, DatabaseFunc, Dataset
from .databases import (
    SVD,
    Torgo,
    Voiced,
    AVFAD,
    MEEI,
    UASpeech,
    UncommonVoice,
    Base as Database,
)
from .databases.svd import VOWELS
from ..diagnosis import DiagnosisMap


class GeneratorV1(Generator):

    __db_map = {
        AVFAD.DB_NAME: AVFAD,
        MEEI.DB_NAME: MEEI,
        SVD.DB_NAME: SVD,
        Torgo.DB_NAME: Torgo,
        UASpeech.DB_NAME: UASpeech,
        UncommonVoice.DB_NAME: UncommonVoice,
        Voiced.DB_NAME: Voiced,
    }

    async def collect_diagnosis_terms(self, source_path: Path) -> Dict[str, List[str]]:
        dbs = [AVFAD, MEEI, SVD, Torgo, UASpeech, UncommonVoice, Voiced]
        terms = {}
        for db in dbs:
            for term in await db(source_path=source_path).collect_diagnosis_terms():
                if term not in terms:
                    terms[term] = set()
                terms[term].add(db.DB_NAME)
        del terms[""]
        return terms

    async def generate_task(
        self,
        source_path: Path,
        filter_func: Callable[[DatabaseFunc], Awaitable[Dataset]],
        task_path: Path,
        diagnosis_map: DiagnosisMap,
        allow_incomplete_classification: bool = False,
    ) -> None:
        async def __database(name: str, min_tasks: int | None = None) -> Database:
            db = self.__db_map[name.lower()](source_path=source_path)
            await db.init(
                diagnosis_map=diagnosis_map,
                allow_incomplete_classification=allow_incomplete_classification,
                min_tasks=min_tasks,
            )
            return db

        tasks = await filter_func(__database)
        self.to_task_file(tasks=tasks.train, output_path=Path(f"{task_path}/train"))
        self.to_task_file(tasks=tasks.val, output_path=Path(f"{task_path}/val"))
        self.to_task_file(tasks=tasks.test, output_path=Path(f"{task_path}/test"))

    async def __call__(
        self,
        source_path: Path,
        tasks_path: Path,
        diagnosis_map: DiagnosisMap,
    ) -> None:
        print("Generating benchmark v1 tasks")
        Path(f"{tasks_path}/streams").mkdir(exist_ok=True)
        svd = SVD(source_path=source_path)
        torgo = Torgo(source_path=source_path)
        voiced = Voiced(source_path=source_path)
        await asyncio.gather(
            svd.init(
                diagnosis_map=diagnosis_map,
                allow_incomplete_classification=False,
                min_tasks=SVD.max_tasks,
            ),
            torgo.init(
                diagnosis_map=diagnosis_map,
                allow_incomplete_classification=False,
                min_tasks=None,
            ),
            voiced.init(
                diagnosis_map=diagnosis_map,
                allow_incomplete_classification=False,
                min_tasks=None,
            ),
        )
        self.__stream0(
            stream_path=Path(f"{tasks_path}/streams/0"),
            svd=svd,
            torgo=torgo,
            voiced=voiced,
        )
        self.__stream1(
            stream_path=Path(f"{tasks_path}/streams/1"),
            svd=svd,
        )
        self.__stream2(
            stream_path=Path(f"{tasks_path}/streams/2"),
            svd=svd,
            torgo=torgo,
            voiced=voiced,
        )
        self.__stream3(
            stream_path=Path(f"{tasks_path}/streams/3"),
            svd=svd,
            torgo=torgo,
            voiced=voiced,
        )

    def __stream0(self, stream_path: Path, svd: SVD, voiced: Voiced, torgo: Torgo):
        print(f"Generating stream 0 from at {stream_path}")
        stream_path.mkdir(exist_ok=True)
        self.to_task_file(
            (
                svd.all_train(level=0)
                + torgo.all_train(level=0)
                + voiced.all_train(level=0)
            ),
            Path(f"{stream_path}/train"),
        )
        self.to_task_file(
            (svd.all_val(level=0) + torgo.all_val(level=0) + voiced.all_val(level=0)),
            Path(f"{stream_path}/val"),
        )

        # task 1
        Path(f"{stream_path}/{1}").mkdir(exist_ok=True)
        self.to_task_file(
            svd.test_set_neutral_vowels(level=0),
            Path(f"{stream_path}/{1}/test"),
        )

        # task 2
        Path(f"{stream_path}/{2}").mkdir(exist_ok=True)
        self.to_task_file(
            svd.test_set_connected_speech(level=0),
            Path(f"{stream_path}/{2}/test"),
        )

        # task 3
        Path(f"{stream_path}/{3}").mkdir(exist_ok=True)
        self.to_task_file(
            voiced.all_test(level=0),
            Path(f"{stream_path}/{3}/test"),
        )

        # task 4
        Path(f"{stream_path}/{4}").mkdir(exist_ok=True)
        self.to_task_file(
            torgo.all_test(level=0),
            Path(f"{stream_path}/{4}/test"),
        )

    def __stream1(self, stream_path: Path, svd: SVD):
        print(f"Generating stream 1 from at {stream_path}")
        stream_path.mkdir(exist_ok=True)

        def single_vowel_task(task_idx: int, level: int, vowel: VOWELS):
            Path(f"{stream_path}/{task_idx}").mkdir(exist_ok=True)
            self.to_task_file(
                svd.train_set_neutral_vowels(level=level, vowel=vowel),
                Path(f"{stream_path}/{task_idx}/train"),
            )
            self.to_task_file(
                svd.test_set_neutral_vowels(level=level, vowel=vowel),
                Path(f"{stream_path}/{task_idx}/test"),
            )
            self.to_task_file(
                svd.val_set_neutral_vowels(level=level, vowel=vowel),
                Path(f"{stream_path}/{task_idx}/val"),
            )

        def connected_speech_task(task_idx: int, level: int):
            Path(f"{stream_path}/{task_idx}").mkdir(exist_ok=True)
            self.to_task_file(
                svd.train_set_connected_speech(level=level),
                Path(f"{stream_path}/{task_idx}/train"),
            )
            self.to_task_file(
                svd.test_set_connected_speech(level=level),
                Path(f"{stream_path}/{task_idx}/test"),
            )
            self.to_task_file(
                svd.val_set_connected_speech(level=level),
                Path(f"{stream_path}/{task_idx}/val"),
            )

        def combined_vocalisation_task(task_idx: int, level: int):
            Path(f"{stream_path}/{task_idx}").mkdir(exist_ok=True)
            self.to_task_file(
                svd.train_set_combined_vowel_vocalisation(level=level),
                Path(f"{stream_path}/{task_idx}/train"),
            )
            self.to_task_file(
                svd.test_set_combined_vowel_vocalisation(level=level),
                Path(f"{stream_path}/{task_idx}/test"),
            )
            self.to_task_file(
                svd.val_set_combined_vowel_vocalisation(level=level),
                Path(f"{stream_path}/{task_idx}/val"),
            )

        def multi_vowel_task(task_idx: int, level: int):
            Path(f"{stream_path}/{task_idx}").mkdir(exist_ok=True)
            vowels: List[VOWELS] = ["a", "i", "u"]
            self.to_task_file(
                svd.train_set_multi_neutral_vowels(level=level, vowels=vowels),
                Path(f"{stream_path}/{task_idx}/train"),
            )
            self.to_task_file(
                svd.test_set_multi_neutral_vowels(level=level, vowels=vowels),
                Path(f"{stream_path}/{task_idx}/test"),
            )
            self.to_task_file(
                svd.val_set_multi_neutral_vowels(level=level, vowels=vowels),
                Path(f"{stream_path}/{task_idx}/val"),
            )

        def lhl_vowel_task(task_idx: int, level: int):
            Path(f"{stream_path}/{task_idx}").mkdir(exist_ok=True)
            self.to_task_file(
                svd.train_set_lhl_vowels(level=level, vowel="a"),
                Path(f"{stream_path}/{task_idx}/train"),
            )
            self.to_task_file(
                svd.test_set_lhl_vowels(level=level, vowel="a"),
                Path(f"{stream_path}/{task_idx}/test"),
            )
            self.to_task_file(
                svd.val_set_lhl_vowels(level=level, vowel="a"),
                Path(f"{stream_path}/{task_idx}/val"),
            )

        def all_combined_task(task_idx: int, level: int):
            Path(f"{stream_path}/{task_idx}").mkdir(exist_ok=True)
            suffixes = [
                "a_n.wav",
                "i_n.wav",
                "u_n.wav",
                "a_lhl.wav",
                "i_lhl.wav",
                "u_lhl.wav",
                "iau.wav",
                "-phrase.wav",
            ]
            self.to_task_file(
                svd.filtered_multi_file_tasks(
                    sessions=svd.dataset.train_sessions, level=level, suffixes=suffixes
                ),
                Path(f"{stream_path}/{task_idx}/train"),
            )
            self.to_task_file(
                svd.filtered_multi_file_tasks(
                    sessions=svd.dataset.test_sessions, level=level, suffixes=suffixes
                ),
                Path(f"{stream_path}/{task_idx}/test"),
            )
            self.to_task_file(
                svd.filtered_multi_file_tasks(
                    sessions=svd.dataset.val_sessions, level=level, suffixes=suffixes
                ),
                Path(f"{stream_path}/{task_idx}/val"),
            )

        # task 1
        single_vowel_task(task_idx=1, level=1, vowel="a")
        # task 2
        single_vowel_task(task_idx=2, level=1, vowel="i")
        # task 3
        single_vowel_task(task_idx=3, level=1, vowel="u")
        # task 4
        lhl_vowel_task(task_idx=4, level=1)
        # task 5
        multi_vowel_task(task_idx=5, level=1)
        # task 6
        combined_vocalisation_task(task_idx=6, level=1)
        # task 7
        connected_speech_task(task_idx=7, level=1)
        # task 8
        all_combined_task(task_idx=8, level=1)
        # task 9
        single_vowel_task(task_idx=9, level=2, vowel="a")
        # task 10
        single_vowel_task(task_idx=10, level=2, vowel="i")
        # task 11
        single_vowel_task(task_idx=11, level=2, vowel="u")
        # task 12
        lhl_vowel_task(task_idx=12, level=2)
        # task 13
        multi_vowel_task(task_idx=13, level=2)
        # task 14
        combined_vocalisation_task(task_idx=14, level=2)
        # task 15
        connected_speech_task(task_idx=15, level=2)
        # task 16
        all_combined_task(task_idx=16, level=2)

    def __stream2(self, stream_path: Path, svd: SVD, torgo: Torgo, voiced: Voiced):
        print(f"Generating stream 2 from at {stream_path}")
        stream_path.mkdir(exist_ok=True)

        def train_val_set(task_idx: int, level: int):
            Path(f"{stream_path}/{task_idx}").mkdir(exist_ok=True)
            self.to_task_file(
                (
                    svd.train_set_neutral_vowels(level=level, vowel="a")
                    + svd.test_set_neutral_vowels(level=level, vowel="a")
                ),
                Path(f"{stream_path}/{task_idx}/train"),
            )
            self.to_task_file(
                svd.val_set_neutral_vowels(level=level, vowel="a"),
                Path(f"{stream_path}/{task_idx}/val"),
            )

        def voiced_test_set(task_idx: int, level: int):
            Path(f"{stream_path}/{task_idx}").mkdir(exist_ok=True)
            self.to_task_file(
                (
                    voiced.all_test(level=level)
                    + voiced.all_train(level=level)
                    + voiced.all_val(level=level)
                ),
                Path(f"{stream_path}/{task_idx}/test"),
            )

        def torgo_test_set(task_idx: int, level: int):
            Path(f"{stream_path}/{task_idx}").mkdir(exist_ok=True)
            self.to_task_file(
                (
                    torgo.all_test(level=level)
                    + torgo.all_train(level=level)
                    + torgo.all_val(level=level)
                ),
                Path(f"{stream_path}/{task_idx}/test"),
            )

        # task 1
        train_val_set(task_idx=1, level=0)
        voiced_test_set(task_idx=1, level=0)
        # task 2
        train_val_set(task_idx=2, level=1)
        voiced_test_set(task_idx=2, level=1)
        # task 3
        train_val_set(task_idx=3, level=2)
        voiced_test_set(task_idx=3, level=2)
        # task 4
        train_val_set(task_idx=4, level=0)
        torgo_test_set(task_idx=4, level=0)

    def __stream3(self, stream_path: Path, svd: SVD, torgo: Torgo, voiced: Voiced):
        print(f"Generating stream 3 from at {stream_path}")
        stream_path.mkdir(exist_ok=True)

        def train_val_set(task_idx: int, level: int):
            Path(f"{stream_path}/{task_idx}").mkdir(exist_ok=True)
            self.to_task_file(
                (voiced.all_train(level=level) + torgo.all_train(level=level)),
                Path(f"{stream_path}/{task_idx}/train"),
            )
            self.to_task_file(
                (voiced.all_val(level=level) + torgo.all_val(level=level)),
                Path(f"{stream_path}/{task_idx}/val"),
            )

        # task 1
        train_val_set(task_idx=1, level=1)
        self.to_task_file(
            voiced.all_test(level=1),
            Path(f"{stream_path}/{1}/test"),
        )
        # task 2
        train_val_set(task_idx=2, level=0)
        self.to_task_file(
            torgo.all_test(level=0),
            Path(f"{stream_path}/{2}/test"),
        )
        # task 3
        train_val_set(task_idx=3, level=1)
        self.to_task_file(
            (
                svd.test_set_neutral_vowels(level=1, vowel="a")
                + svd.val_set_neutral_vowels(level=1, vowel="a")
                + svd.train_set_neutral_vowels(level=1, vowel="a")
            ),
            Path(f"{stream_path}/{3}/test"),
        )
        # task 4
        train_val_set(task_idx=4, level=2)
        self.to_task_file(
            voiced.all_test(level=2),
            Path(f"{stream_path}/{4}/test"),
        )
        # task 5
        train_val_set(task_idx=5, level=2)
        self.to_task_file(
            (
                svd.test_set_neutral_vowels(level=2, vowel="a")
                + svd.val_set_neutral_vowels(level=2, vowel="a")
                + svd.train_set_neutral_vowels(level=2, vowel="a")
            ),
            Path(f"{stream_path}/{5}/test"),
        )

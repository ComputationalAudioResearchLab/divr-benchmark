from pathlib import Path
from typing import List
from .generator import Generator
from .databases import SVD, Torgo, Voiced
from .databases.svd import VOWELS


class GeneratorV1(Generator):
    def __call__(self, source_path: Path, tasks_path: Path) -> None:
        print("Generating benchmark v1 tasks")
        Path(f"{tasks_path}/streams").mkdir(exist_ok=True)
        svd = SVD(source_path=source_path, allow_incomplete_classification=False)
        torgo = Torgo(source_path=source_path, allow_incomplete_classification=True)
        voiced = Voiced(source_path=source_path, allow_incomplete_classification=False)
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

    def __stream0(self, stream_path: Path, svd: SVD, voiced: Voiced, torgo: Torgo):
        print(f"Generating stream 0 from at {stream_path}")
        stream_path.mkdir(exist_ok=True)
        train = (
            svd.all_train(level=0)
            + torgo.all_train(level=0)
            + voiced.all_train(level=0)
        )
        self.to_task_file(train, Path(f"{stream_path}/train"))
        val = svd.all_val(level=0) + torgo.all_val(level=0) + voiced.all_val(level=0)
        self.to_task_file(val, Path(f"{stream_path}/val"))

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
        # Path(f"{stream_path}/{4}").mkdir(exist_ok=True)
        # self.to_task_file(
        #     torgo.test_set_connected_speech(level=0),
        #     Path(f"{stream_path}/{4}/test"),
        # )

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

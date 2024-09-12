from __future__ import annotations
import yaml
from pathlib import Path
from typing import Literal
from divr_benchmark.benchmark import Task
from divr_benchmark.diagnosis import DiagnosisMap
from divr_benchmark.benchmark.audio_loader import AudioLoader
from divr_benchmark.task_generator.generator import Generator
from divr_benchmark.task_generator.databases import SVD, Voiced

from .dtypes import Database


class Tasks:

    keys = Literal["svd_speech", "svd_a", "svd_i", "svd_u", "voiced"]

    def __init__(self, data_path: Path, tasks_path: Path) -> None:
        self.__data_path = data_path
        self.__tasks_path = tasks_path

    def prepare_task(self, diagnosis_level: int, task_key: Tasks.keys):
        task_path = Path(f"{self.__tasks_path}/{diagnosis_level}/{task_key}")
        task_path.mkdir(parents=True, exist_ok=True)
        if task_key == "svd_speech":
            [train_set, val_set, test_set] = self.__svd_speech(diagnosis_level=diagnosis_level)
        elif task_key == "svd_a":
            [train_set, val_set, test_set] = self.__svd_vowel(diagnosis_level=diagnosis_level, vowel="a")
        elif task_key == "svd_i":
            [train_set, val_set, test_set] = self.__svd_vowel(diagnosis_level=diagnosis_level, vowel="i")
        elif task_key == "svd_u":
            [train_set, val_set, test_set] = self.__svd_vowel(diagnosis_level=diagnosis_level, vowel="u")
        elif task_key == "voiced":
            test_set = self.__voiced(diagnosis_level=diagnosis_level)

        gen = Generator()
        if task_key != "voiced":
            train_set, val_set, test_set = gen.truncate_low_resource_classes(
                task_list=[train_set, val_set, test_set],
                min_examples=5,
            )
            gen.to_task_file(train_set, output_path=f"{task_path}/train")
            gen.to_task_file(val_set, output_path=f"{task_path}/val")

        gen.to_task_file(test_set, output_path=f"{task_path}/test")

    def __voiced(self, diagnosis_level: int):
        voiced = Voiced(source_path=self.__data_path, allow_incomplete_classification=False)
        test_set = voiced.all_train(level=diagnosis_level)
        test_set += voiced.all_val(level=diagnosis_level)
        test_set += voiced.all_test(level=diagnosis_level)
        return test_set
    
    def __svd_speech(self, diagnosis_level: int):
        svd = SVD(source_path=self.__data_path, allow_incomplete_classification=False)
        train_set = svd.train_set_connected_speech(level=diagnosis_level)
        val_set = svd.val_set_connected_speech(level=diagnosis_level)
        test_set = svd.test_set_connected_speech(level=diagnosis_level)
        return train_set, val_set, test_set
    
    def __svd_vowel(self, diagnosis_level: int, vowel: Literal["a","i","u"]):
        svd = SVD(source_path=self.__data_path, allow_incomplete_classification=False)
        train_set = svd.train_set_neutral_vowels(level=diagnosis_level, vowel=vowel)
        val_set = svd.val_set_neutral_vowels(level=diagnosis_level, vowel=vowel)
        test_set = svd.test_set_neutral_vowels(level=diagnosis_level, vowel=vowel)
        return train_set, val_set, test_set

    def load_task(self, task_key: Tasks.keys, diagnosis_level: int) -> Database:
        task_path = Path(f"{self.__tasks_path}/{diagnosis_level}/{task_key}")
        train_set = self.load_task_file(Path(f"{task_path}/train.yml"))
        val_set = self.load_task_file(Path(f"{task_path}/val.yml"))
        test_set = self.load_task_file(Path(f"{task_path}/test.yml"))
        return (train_set, val_set, test_set)
    
    def load_test_task(self, task_key: Tasks.keys, diagnosis_level: int) -> Database:
        task_path = Path(f"{self.__tasks_path}/{diagnosis_level}/{task_key}")
        test_set = self.load_task_file(Path(f"{task_path}/test.yml"))
        return ([], [], test_set)

    def load_task_file(self, task_file: Path):
        with open(task_file, "r") as data_file:
            data = yaml.load(data_file, yaml.FullLoader)
            dataset = []
            for row in data.values():
                age = row["age"]
                gender = row["gender"]
                label = row["label"]
                audio_keys = row["audio_keys"]
                audio_path = audio_keys[0]
                dataset += [(age, gender, label, audio_path)]
            return dataset
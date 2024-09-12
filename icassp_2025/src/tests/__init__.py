import torch
from pathlib import Path
from typing import Literal

from .tester import Tester, CrossTester
from ..data_loader import Tasks
from ..model import UnispeechSAT, MFCC

class Runner:
    __exp = {
        "svd_a_0_mfcc": ["svd_a", 0, MFCC, 2700, None],
        "svd_a_0_unispeechSAT": ["svd_a", 0, UnispeechSAT, 168, None],
        "svd_a_1_mfcc": ["svd_a", 1, MFCC, 2700, None],
        "svd_a_1_unispeechSAT": ["svd_a", 1, UnispeechSAT, 55, None],
        "svd_a_2_mfcc": ["svd_a", 2, MFCC, 1340, None],
        "svd_a_2_unispeechSAT": ["svd_a", 2, UnispeechSAT, 171, None],
        "svd_a_3_mfcc": ["svd_a", 3, MFCC, 855, None],
        "svd_a_3_unispeechSAT": ["svd_a", 3, UnispeechSAT, 153, None],
        "svd_i_0_mfcc": ["svd_i", 0, MFCC, 505, None],
        "svd_i_0_unispeechSAT": ["svd_i", 0, UnispeechSAT, 168, None],
        "svd_i_1_mfcc": ["svd_i", 1, MFCC, 2700, None],
        "svd_i_1_unispeechSAT": ["svd_i", 1, UnispeechSAT, 68, None],
        "svd_i_2_mfcc": ["svd_i", 2, MFCC, 2700, None],
        "svd_i_2_unispeechSAT": ["svd_i", 2, UnispeechSAT, 161, None],
        "svd_i_3_mfcc": ["svd_i", 3, MFCC, 2498, None],
        "svd_i_3_unispeechSAT": ["svd_i", 3, UnispeechSAT, 192, None],
        "svd_u_0_mfcc": ["svd_u", 0, MFCC, 2700, None],
        "svd_u_0_unispeechSAT": ["svd_u", 0, UnispeechSAT, 181, None],
        "svd_u_1_mfcc": ["svd_u", 1, MFCC, 2700, None],
        "svd_u_1_unispeechSAT": ["svd_u", 1, UnispeechSAT, 184, None],
        "svd_u_2_mfcc": ["svd_u", 2, MFCC, 2700, None],
        "svd_u_2_unispeechSAT": ["svd_u", 2, UnispeechSAT, 186, None],
        "svd_u_3_mfcc": ["svd_u", 3, MFCC, 2700, None],
        "svd_u_3_unispeechSAT": ["svd_u", 3, UnispeechSAT, 186, None],
        "svd_speech_0_0.125_mfcc": ["svd_speech", 0, MFCC, 2529, 0.125],
        "svd_speech_0_0.125_unispeechSAT": ["svd_speech", 0, UnispeechSAT, 187, 0.125],
        "svd_speech_0_0.25_mfcc": ["svd_speech", 0, MFCC, 742, 0.25],
        "svd_speech_0_0.25_unispeechSAT": ["svd_speech", 0, UnispeechSAT, 101, 0.25],
        "svd_speech_0_0.5_mfcc": ["svd_speech", 0, MFCC, 1479, 0.5],
        "svd_speech_0_0.5_unispeechSAT": ["svd_speech", 0, UnispeechSAT, 128, 0.5],
        "svd_speech_0_1_mfcc": ["svd_speech", 0, MFCC, 662, 1],
        "svd_speech_0_1_unispeechSAT": ["svd_speech", 0, UnispeechSAT, 169, 1],
        "svd_speech_0_mfcc": ["svd_speech", 0, MFCC, 1521, None],
        "svd_speech_0_unispeechSAT": ["svd_speech", 0, UnispeechSAT, 97, None],
        "svd_speech_1_0.125_mfcc": ["svd_speech", 1, MFCC, 883, 0.125],
        "svd_speech_1_0.125_unispeechSAT": ["svd_speech", 1, UnispeechSAT, 175, 0.125],
        "svd_speech_1_0.25_mfcc": ["svd_speech", 1, MFCC, 758, 0.25],
        "svd_speech_1_0.25_unispeechSAT": ["svd_speech", 1, UnispeechSAT, 132, 0.25],
        "svd_speech_1_0.5_mfcc": ["svd_speech", 1, MFCC, 935, 0.5],
        "svd_speech_1_0.5_unispeechSAT": ["svd_speech", 1, UnispeechSAT, 193, 0.5],
        "svd_speech_1_1_mfcc": ["svd_speech", 1, MFCC, 2700, 1],
        "svd_speech_1_1_unispeechSAT": ["svd_speech", 1, UnispeechSAT, 164, 1],
        "svd_speech_1_mfcc": ["svd_speech", 1, MFCC, 2700, None],
        "svd_speech_1_unispeechSAT": ["svd_speech", 1, UnispeechSAT, 200, None],
        "svd_speech_2_0.125_unispeechSAT": ["svd_speech", 2, UnispeechSAT, 127, 0.125],
        "svd_speech_2_0.25_unispeechSAT": ["svd_speech", 2, UnispeechSAT, 187, 0.25],
        "svd_speech_2_0.5_unispeechSAT": ["svd_speech", 2, UnispeechSAT, 149, 0.5],
        "svd_speech_2_1_mfcc": ["svd_speech", 2, MFCC, 2687, 1],
        "svd_speech_2_1_unispeechSAT": ["svd_speech", 2, UnispeechSAT, 138, 1],
        "svd_speech_2_mfcc": ["svd_speech", 2, MFCC, 2616, None],
        "svd_speech_2_unispeechSAT": ["svd_speech", 2, UnispeechSAT, 200, None],
        "svd_speech_3_0.125_unispeechSAT": ["svd_speech", 3, UnispeechSAT, 200, 0.125],
        "svd_speech_3_0.25_unispeechSAT": ["svd_speech", 3, UnispeechSAT, 196, 0.25],
        "svd_speech_3_0.5_unispeechSAT": ["svd_speech", 3, UnispeechSAT, 198, 0.5],
        "svd_speech_3_1_mfcc": ["svd_speech", 3, MFCC, 2154, 1],
        "svd_speech_3_1_unispeechSAT": ["svd_speech", 3, UnispeechSAT, 188, 1],
        "svd_speech_3_mfcc": ["svd_speech", 3, MFCC, 2110, None],
        "svd_speech_3_unispeechSAT": ["svd_speech", 3, UnispeechSAT, 200, None],
    }
    EXP_KEYS = Literal[
        "svd_a_0_mfcc",
        "svd_a_0_unispeechSAT",
        "svd_a_1_mfcc",
        "svd_a_1_unispeechSAT",
        "svd_a_2_mfcc",
        "svd_a_2_unispeechSAT",
        "svd_a_3_mfcc",
        "svd_a_3_unispeechSAT",
        "svd_i_0_mfcc",
        "svd_i_0_unispeechSAT",
        "svd_i_1_mfcc",
        "svd_i_1_unispeechSAT",
        "svd_i_2_mfcc",
        "svd_i_2_unispeechSAT",
        "svd_i_3_mfcc",
        "svd_i_3_unispeechSAT",
        "svd_u_0_mfcc",
        "svd_u_0_unispeechSAT",
        "svd_u_1_mfcc",
        "svd_u_1_unispeechSAT",
        "svd_u_2_mfcc",
        "svd_u_2_unispeechSAT",
        "svd_u_3_mfcc",
        "svd_u_3_unispeechSAT",
        "svd_speech_0_0.125_mfcc",
        "svd_speech_0_0.125_unispeechSAT",
        "svd_speech_0_0.25_mfcc",
        "svd_speech_0_0.25_unispeechSAT",
        "svd_speech_0_0.5_mfcc",
        "svd_speech_0_0.5_unispeechSAT",
        "svd_speech_0_1_mfcc",
        "svd_speech_0_1_unispeechSAT",
        "svd_speech_0_mfcc",
        "svd_speech_0_unispeechSAT",
        "svd_speech_1_0.125_mfcc",
        "svd_speech_1_0.125_unispeechSAT",
        "svd_speech_1_0.25_mfcc",
        "svd_speech_1_0.25_unispeechSAT",
        "svd_speech_1_0.5_mfcc",
        "svd_speech_1_0.5_unispeechSAT",
        "svd_speech_1_1_mfcc",
        "svd_speech_1_1_unispeechSAT",
        "svd_speech_1_mfcc",
        "svd_speech_1_unispeechSAT",
        "svd_speech_2_0.125_unispeechSAT",
        "svd_speech_2_0.25_unispeechSAT",
        "svd_speech_2_0.5_unispeechSAT",
        "svd_speech_2_1_mfcc",
        "svd_speech_2_1_unispeechSAT",
        "svd_speech_2_mfcc",
        "svd_speech_2_unispeechSAT",
        "svd_speech_3_0.125_unispeechSAT",
        "svd_speech_3_0.25_unispeechSAT",
        "svd_speech_3_0.5_unispeechSAT",
        "svd_speech_3_1_mfcc",
        "svd_speech_3_1_unispeechSAT",
        "svd_speech_3_mfcc",
        "svd_speech_3_unispeechSAT",
    ]

    def __init__(self, tasks: Tasks, data_path: Path, cache_path: Path) -> None:
        self.__tasks = tasks
        self.__data_path = data_path
        self.__cache_path = cache_path

    def run(self, exp_key: EXP_KEYS):
        task_key, diagnosis_level, feature_cls, best_checkpoint_epoch, max_audio_seconds = self.__exp[exp_key]
        device = torch.device('cuda')
        feature = feature_cls(device=device)
        database = self.__tasks.load_task(task_key=task_key, diagnosis_level=diagnosis_level)
        tester = Tester(
            data_path=self.__data_path,
            cache_path=self.__cache_path,
            device=device,
            feature=feature,
            database=database,
            exp_key=exp_key,
            best_checkpoint_epoch=best_checkpoint_epoch,
            max_audio_seconds=max_audio_seconds
        )
        tester.run()
    
    def run_cross(self):
        device = torch.device('cuda')
        test_exp_key = "voiced"
        for exp_key, exp_val in self.__exp.items():
            task_key, diagnosis_level, feature_cls, best_checkpoint_epoch, max_audio_seconds = exp_val
            feature = feature_cls(device=device)
            train_database = self.__tasks.load_task(task_key=task_key, diagnosis_level=diagnosis_level)
            cross_database = self.__tasks.load_test_task(task_key="voiced", diagnosis_level=diagnosis_level)
            tester = CrossTester(
                data_path=self.__data_path,
                cache_path=self.__cache_path,
                device=device,
                feature=feature,
                train_database=train_database,
                cross_database=cross_database,
                train_exp_key=exp_key,
                test_exp_key=test_exp_key,
                best_checkpoint_epoch=best_checkpoint_epoch,
                max_audio_seconds=max_audio_seconds
            )
            if diagnosis_level == 3:
                tester.run_without_confusion()
            else:
                tester.run()

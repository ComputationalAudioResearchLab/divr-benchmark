import torch
from pathlib import Path
from typing import Literal

from .trainer import Trainer, TrainerShort
from ..data_loader import Tasks
from ..model import UnispeechSAT, MFCC

class Runner:
    __exp = {
        "svd_speech_0_unispeechSAT": ["svd_speech", 0, UnispeechSAT, 201],
        "svd_a_0_unispeechSAT": ["svd_a", 0, UnispeechSAT, 201],
        "svd_i_0_unispeechSAT": ["svd_i", 0, UnispeechSAT, 201],
        "svd_u_0_unispeechSAT": ["svd_u", 0, UnispeechSAT, 201],

        "svd_speech_1_unispeechSAT": ["svd_speech", 1, UnispeechSAT, 201],
        "svd_a_1_unispeechSAT": ["svd_a", 1, UnispeechSAT, 201],
        "svd_i_1_unispeechSAT": ["svd_i", 1, UnispeechSAT, 201],
        "svd_u_1_unispeechSAT": ["svd_u", 1, UnispeechSAT, 201],

        "svd_speech_2_unispeechSAT": ["svd_speech", 2, UnispeechSAT, 201],
        "svd_a_2_unispeechSAT": ["svd_a", 2, UnispeechSAT, 201],
        "svd_i_2_unispeechSAT": ["svd_i", 2, UnispeechSAT, 201],
        "svd_u_2_unispeechSAT": ["svd_u", 2, UnispeechSAT, 201],
        
        "svd_speech_3_unispeechSAT": ["svd_speech", 3, UnispeechSAT, 201],
        "svd_a_3_unispeechSAT": ["svd_a", 3, UnispeechSAT, 201],
        "svd_i_3_unispeechSAT": ["svd_i", 3, UnispeechSAT, 201],
        "svd_u_3_unispeechSAT": ["svd_u", 3, UnispeechSAT, 201],

        
        "svd_speech_0_mfcc": ["svd_speech", 0, MFCC, 3000],
        "svd_a_0_mfcc": ["svd_a", 0, MFCC, 3000],
        "svd_i_0_mfcc": ["svd_i", 0, MFCC, 3000],
        "svd_u_0_mfcc": ["svd_u", 0, MFCC, 3000],

        "svd_speech_1_mfcc": ["svd_speech", 1, MFCC, 3000],
        "svd_a_1_mfcc": ["svd_a", 1, MFCC, 3000],
        "svd_i_1_mfcc": ["svd_i", 1, MFCC, 3000],
        "svd_u_1_mfcc": ["svd_u", 1, MFCC, 3000],

        "svd_speech_2_mfcc": ["svd_speech", 2, MFCC, 3000],
        "svd_a_2_mfcc": ["svd_a", 2, MFCC, 3000],
        "svd_i_2_mfcc": ["svd_i", 2, MFCC, 3000],
        "svd_u_2_mfcc": ["svd_u", 2, MFCC, 3000],
        
        "svd_speech_3_mfcc": ["svd_speech", 3, MFCC, 3000],
        "svd_a_3_mfcc": ["svd_a", 3, MFCC, 3000],
        "svd_i_3_mfcc": ["svd_i", 3, MFCC, 3000],
        "svd_u_3_mfcc": ["svd_u", 3, MFCC, 3000],
    }
    EXP_KEYS = Literal[
        "svd_speech_0_unispeechSAT",
        "svd_a_0_unispeechSAT",
        "svd_i_0_unispeechSAT",
        "svd_u_0_unispeechSAT",

        "svd_speech_1_unispeechSAT",
        "svd_a_1_unispeechSAT",
        "svd_i_1_unispeechSAT",
        "svd_u_1_unispeechSAT",
        
        "svd_speech_2_unispeechSAT",
        "svd_a_2_unispeechSAT",
        "svd_i_2_unispeechSAT",
        "svd_u_2_unispeechSAT",
        
        "svd_speech_3_unispeechSAT",
        "svd_a_3_unispeechSAT",
        "svd_i_3_unispeechSAT",
        "svd_u_3_unispeechSAT",

        "svd_speech_0_mfcc",
        "svd_a_0_mfcc",
        "svd_i_0_mfcc",
        "svd_u_0_mfcc",
        "svd_speech_1_mfcc",
        "svd_a_1_mfcc",
        "svd_i_1_mfcc",
        "svd_u_1_mfcc",
        "svd_speech_2_mfcc",
        "svd_a_2_mfcc",
        "svd_i_2_mfcc",
        "svd_u_2_mfcc",
        "svd_speech_3_mfcc",
        "svd_a_3_mfcc",
        "svd_i_3_mfcc",
        "svd_u_3_mfcc",
    ]

    
    __exp_short = {
        "svd_speech_0_1_unispeechSAT": ["svd_speech", 0, 1, UnispeechSAT, 201],
        "svd_speech_1_1_unispeechSAT": ["svd_speech", 1, 1, UnispeechSAT, 201],
        "svd_speech_2_1_unispeechSAT": ["svd_speech", 2, 1, UnispeechSAT, 201],
        "svd_speech_3_1_unispeechSAT": ["svd_speech", 3, 1, UnispeechSAT, 201],
        "svd_speech_0_0.5_unispeechSAT": ["svd_speech", 0, 0.5, UnispeechSAT, 201],
        "svd_speech_1_0.5_unispeechSAT": ["svd_speech", 1, 0.5, UnispeechSAT, 201],
        "svd_speech_2_0.5_unispeechSAT": ["svd_speech", 2, 0.5, UnispeechSAT, 201],
        "svd_speech_3_0.5_unispeechSAT": ["svd_speech", 3, 0.5, UnispeechSAT, 201],
        "svd_speech_0_0.25_unispeechSAT": ["svd_speech", 0, 0.25, UnispeechSAT, 201],
        "svd_speech_1_0.25_unispeechSAT": ["svd_speech", 1, 0.25, UnispeechSAT, 201],
        "svd_speech_2_0.25_unispeechSAT": ["svd_speech", 2, 0.25, UnispeechSAT, 201],
        "svd_speech_3_0.25_unispeechSAT": ["svd_speech", 3, 0.25, UnispeechSAT, 201],
        "svd_speech_0_0.125_unispeechSAT": ["svd_speech", 0, 0.125, UnispeechSAT, 201],
        "svd_speech_1_0.125_unispeechSAT": ["svd_speech", 1, 0.125, UnispeechSAT, 201],
        "svd_speech_2_0.125_unispeechSAT": ["svd_speech", 2, 0.125, UnispeechSAT, 201],
        "svd_speech_3_0.125_unispeechSAT": ["svd_speech", 3, 0.125, UnispeechSAT, 201],


        
        "svd_speech_0_1_mfcc": ["svd_speech", 0, 1, MFCC, 3000],
        "svd_speech_1_1_mfcc": ["svd_speech", 1, 1, MFCC, 3000],
        "svd_speech_2_1_mfcc": ["svd_speech", 2, 1, MFCC, 3000],
        "svd_speech_3_1_mfcc": ["svd_speech", 3, 1, MFCC, 3000],

        "svd_speech_0_0.5_mfcc": ["svd_speech", 0, 0.5, MFCC, 3000],
        "svd_speech_1_0.5_mfcc": ["svd_speech", 1, 0.5, MFCC, 3000],
        
        "svd_speech_0_0.25_mfcc": ["svd_speech", 0, 0.25, MFCC, 3000],
        "svd_speech_1_0.25_mfcc": ["svd_speech", 1, 0.25, MFCC, 3000],
        
        "svd_speech_0_0.125_mfcc": ["svd_speech", 0, 0.125, MFCC, 3000],
        "svd_speech_1_0.125_mfcc": ["svd_speech", 1, 0.125, MFCC, 3000],
    }
    EXP_SHORT_KEYS = Literal[
        "svd_speech_0_1_unispeechSAT",
        "svd_speech_1_1_unispeechSAT",
        "svd_speech_2_1_unispeechSAT",
        "svd_speech_3_1_unispeechSAT",
        "svd_speech_0_0.5_unispeechSAT",
        "svd_speech_1_0.5_unispeechSAT",
        "svd_speech_2_0.5_unispeechSAT",
        "svd_speech_3_0.5_unispeechSAT",
        "svd_speech_0_0.25_unispeechSAT",
        "svd_speech_1_0.25_unispeechSAT",
        "svd_speech_2_0.25_unispeechSAT",
        "svd_speech_3_0.25_unispeechSAT",
        "svd_speech_0_0.125_unispeechSAT",
        "svd_speech_1_0.125_unispeechSAT",
        "svd_speech_2_0.125_unispeechSAT",
        "svd_speech_3_0.125_unispeechSAT",

        "svd_speech_0_1_mfcc",
        "svd_speech_1_1_mfcc",
        "svd_speech_2_1_mfcc",
        "svd_speech_3_1_mfcc",
        "svd_speech_0_0.5_mfcc",
        "svd_speech_1_0.5_mfcc",
        "svd_speech_0_0.25_mfcc",
        "svd_speech_1_0.25_mfcc",
        "svd_speech_0_0.125_mfcc",
        "svd_speech_1_0.125_mfcc",
    ]

    def __init__(self, tasks: Tasks, data_path: Path, cache_path: Path) -> None:
        self.__tasks = tasks
        self.__data_path = data_path
        self.__cache_path = cache_path

    def run(self, exp_key: EXP_KEYS):
        task_key, diagnosis_level, feature_cls, num_epochs = self.__exp[exp_key]
        device = torch.device('cuda')
        feature = feature_cls(device=device)
        database = self.__tasks.load_task(task_key=task_key, diagnosis_level=diagnosis_level)
        trainer = Trainer(
            num_epochs=num_epochs,
            data_path=self.__data_path,
            cache_path=self.__cache_path,
            device=device,
            feature=feature,
            database=database,
            exp_key=exp_key,
        )
        trainer.run()
        

    def run_short(self, exp_key: EXP_SHORT_KEYS):
        task_key, diagnosis_level, max_audio_seconds, feature_cls, num_epochs = self.__exp_short[exp_key]
        device = torch.device('cuda')
        feature = feature_cls(device=device)
        database = self.__tasks.load_task(task_key=task_key, diagnosis_level=diagnosis_level)
        trainer = TrainerShort(
            num_epochs=num_epochs,
            data_path=self.__data_path,
            cache_path=self.__cache_path,
            device=device,
            feature=feature,
            database=database,
            exp_key=exp_key,
            max_audio_seconds=max_audio_seconds,
        )
        trainer.run()
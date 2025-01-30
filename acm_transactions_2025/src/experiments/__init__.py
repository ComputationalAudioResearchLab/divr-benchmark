from __future__ import annotations
import torch
from pathlib import Path
from typing import Literal

from ..model import UnispeechSAT
from ..tasks_generator import TaskGenerator
from ..data_loader import DataLoader
from .trainer import Trainer
from .trainer_multitask import TrainerMultiTask
from .trainer_multicrit import TrainerMultiCrit
from .tester import Tester
from .tester_multitask import TesterMultiTask
from .tester_multicrit import TesterMultiCrit


class Runner:

    # fmt: off
    __exp = {
        "unispeechSAT_phrase_0": ["phrase", [0], UnispeechSAT, 200, 16, Trainer],
        "unispeechSAT_a_0": ["a_n", [0], UnispeechSAT, 200, 16, Trainer],
        "unispeechSAT_i_0": ["i_n", [0], UnispeechSAT, 200, 16, Trainer],
        "unispeechSAT_u_0": ["u_n", [0], UnispeechSAT, 200, 16, Trainer],
        "unispeechSAT_phrase_1": ["phrase", [1], UnispeechSAT, 200, 16, Trainer],
        "unispeechSAT_a_1": ["a_n", [1], UnispeechSAT, 200, 16, Trainer],
        "unispeechSAT_i_1": ["i_n", [1], UnispeechSAT, 200, 16, Trainer],
        "unispeechSAT_u_1": ["u_n", [1], UnispeechSAT, 200, 16, Trainer],
        "unispeechSAT_phrase_2": ["phrase", [2], UnispeechSAT, 200, 16, Trainer],
        "unispeechSAT_phrase_3": ["phrase", [3], UnispeechSAT, 200, 16, Trainer],
        "unispeechSAT_a_3": ["a_n", [3], UnispeechSAT, 200, 16, Trainer],
        "unispeechSAT_i_3": ["i_n", [3], UnispeechSAT, 200, 16, Trainer],
        "unispeechSAT_u_3": ["u_n", [3], UnispeechSAT, 200, 16, Trainer],
        # Multitask experiments
        "unispeechSAT_phrase_0+1": ["phrase", [0, 1], UnispeechSAT, 200, 16, TrainerMultiTask],
        "unispeechSAT_phrase_0+2": ["phrase", [0, 2], UnispeechSAT, 200, 16, TrainerMultiTask],
        "unispeechSAT_phrase_0+3": ["phrase", [0, 3], UnispeechSAT, 200, 16, TrainerMultiTask],
        "unispeechSAT_phrase_1+2": ["phrase", [1, 2], UnispeechSAT, 200, 16, TrainerMultiTask],
        "unispeechSAT_phrase_1+3": ["phrase", [1, 3], UnispeechSAT, 200, 16, TrainerMultiTask],
        "unispeechSAT_phrase_2+3": ["phrase", [2, 3], UnispeechSAT, 200, 16, TrainerMultiTask],
        "unispeechSAT_phrase_0+1+2": ["phrase", [0, 1, 2], UnispeechSAT, 200, 16, TrainerMultiTask],
        "unispeechSAT_phrase_0+1+3": ["phrase", [0, 1, 3], UnispeechSAT, 200, 16, TrainerMultiTask],
        "unispeechSAT_phrase_0+2+3": ["phrase", [0, 2, 3], UnispeechSAT, 200, 16, TrainerMultiTask],
        "unispeechSAT_phrase_1+2+3": ["phrase", [1, 2, 3], UnispeechSAT, 200, 16, TrainerMultiTask],
        "unispeechSAT_phrase_0+1+2+3": ["phrase", [0, 1, 2, 3], UnispeechSAT, 200, 16, TrainerMultiTask],
        # Single output multitask experiments
        "mc_unispeechSAT_phrase_0+1": ["phrase", [0, 1], UnispeechSAT, 200, 16, TrainerMultiCrit],
        "mc_unispeechSAT_phrase_0+2": ["phrase", [0, 2], UnispeechSAT, 200, 16, TrainerMultiCrit],
        "mc_unispeechSAT_phrase_0+3": ["phrase", [0, 3], UnispeechSAT, 200, 16, TrainerMultiCrit],
        "mc_unispeechSAT_phrase_1+2": ["phrase", [1, 2], UnispeechSAT, 200, 16, TrainerMultiCrit],
        "mc_unispeechSAT_phrase_1+3": ["phrase", [1, 3], UnispeechSAT, 1000, 16, TrainerMultiCrit],
        "mc_unispeechSAT_phrase_2+3": ["phrase", [2, 3], UnispeechSAT, 200, 16, TrainerMultiCrit],
        "mc_unispeechSAT_phrase_0+1+2": ["phrase", [0, 1, 2], UnispeechSAT, 200, 16, TrainerMultiCrit],
        "mc_unispeechSAT_phrase_0+1+3": ["phrase", [0, 1, 3], UnispeechSAT, 200, 16, TrainerMultiCrit],
        "mc_unispeechSAT_phrase_0+2+3": ["phrase", [0, 2, 3], UnispeechSAT, 200, 16, TrainerMultiCrit],
        "mc_unispeechSAT_phrase_1+2+3": ["phrase", [1, 2, 3], UnispeechSAT, 200, 16, TrainerMultiCrit],
        "mc_unispeechSAT_phrase_0+1+2+3": ["phrase", [0, 1, 2, 3], UnispeechSAT, 200, 16, TrainerMultiCrit],
    }
    # fmt: on

    EXP_KEYS = Literal[
        "unispeechSAT_phrase_0",
        "unispeechSAT_a_0",
        "unispeechSAT_i_0",
        "unispeechSAT_u_0",
        "unispeechSAT_phrase_1",
        "unispeechSAT_a_1",
        "unispeechSAT_i_1",
        "unispeechSAT_u_1",
        "unispeechSAT_phrase_2",
        "unispeechSAT_phrase_3",
        "unispeechSAT_a_3",
        "unispeechSAT_i_3",
        "unispeechSAT_u_3",
        "unispeechSAT_phrase_0+1",
        "unispeechSAT_phrase_0+2",
        "unispeechSAT_phrase_0+3",
        "unispeechSAT_phrase_1+2",
        "unispeechSAT_phrase_1+3",
        "unispeechSAT_phrase_2+3",
        "unispeechSAT_phrase_0+1+2",
        "unispeechSAT_phrase_0+1+3",
        "unispeechSAT_phrase_0+2+3",
        "unispeechSAT_phrase_1+2+3",
        "unispeechSAT_phrase_0+1+2+3",
        "mc_unispeechSAT_phrase_0+1",
        "mc_unispeechSAT_phrase_0+2",
        "mc_unispeechSAT_phrase_0+3",
        "mc_unispeechSAT_phrase_1+2",
        "mc_unispeechSAT_phrase_1+3",
        "mc_unispeechSAT_phrase_2+3",
        "mc_unispeechSAT_phrase_0+1+2",
        "mc_unispeechSAT_phrase_0+1+3",
        "mc_unispeechSAT_phrase_0+2+3",
        "mc_unispeechSAT_phrase_1+2+3",
        "mc_unispeechSAT_phrase_0+1+2+3",
    ]

    def __init__(
        self,
        tasks_generator: TaskGenerator,
        cache_path: Path,
        results_path: Path,
    ):
        self.__tasks_generator = tasks_generator
        self.__cache_path = cache_path
        self.__results_path = results_path

    def train(self, exp_key: Runner.EXP_KEYS, tboard_enabled: bool):
        device = torch.device("cuda")
        (
            task_key,
            diag_levels,
            feature_cls,
            num_epochs,
            batch_size,
            trainer_cls,
        ) = self.__exp[exp_key]
        task = self.__tasks_generator.load_task(
            task=task_key,
            diag_level=max(diag_levels),
        )
        feature_function = (
            feature_cls(device=device) if feature_cls is not None else None
        )
        data_loader = DataLoader(
            random_seed=42,
            shuffle_train=True,
            batch_size=batch_size,
            device=device,
            task=task,
            feature_function=feature_function,
            diag_levels=diag_levels,
            return_ids=False,
        )
        if trainer_cls == Trainer:
            trainer = Trainer(
                cache_path=self.__cache_path,
                device=device,
                data_loader=data_loader,
                exp_key=exp_key,
                num_epochs=num_epochs,
                tboard_enabled=tboard_enabled,
            )
        elif trainer_cls == TrainerMultiTask:
            trainer = TrainerMultiTask(
                cache_path=self.__cache_path,
                device=device,
                data_loader=data_loader,
                exp_key=exp_key,
                num_epochs=num_epochs,
                tboard_enabled=tboard_enabled,
            )
        elif trainer_cls == TrainerMultiCrit:
            trainer = TrainerMultiCrit(
                cache_path=self.__cache_path,
                device=device,
                data_loader=data_loader,
                exp_key=exp_key,
                num_epochs=num_epochs,
                tboard_enabled=tboard_enabled,
            )
        else:
            raise ValueError(f"Unsupported trainer type: {trainer_cls}")
        trainer.run()

    def eval(self, exp_key: Runner.EXP_KEYS, load_epoch: int):
        device = torch.device("cuda")
        (
            task_key,
            diag_levels,
            feature_cls,
            num_epochs,
            batch_size,
            trainer_cls,
        ) = self.__exp[exp_key]
        task = self.__tasks_generator.load_task(
            task=task_key,
            diag_level=max(diag_levels),
        )
        feature_function = (
            feature_cls(device=device) if feature_cls is not None else None
        )
        data_loader = DataLoader(
            random_seed=42,
            shuffle_train=True,
            batch_size=batch_size,
            device=device,
            task=task,
            feature_function=feature_function,
            diag_levels=diag_levels,
            return_ids=True,
        )
        if trainer_cls == Trainer:
            tester = Tester(
                cache_path=self.__cache_path,
                results_path=self.__results_path,
                device=device,
                data_loader=data_loader,
                exp_key=exp_key,
                load_epoch=load_epoch,
            )
        elif trainer_cls == TrainerMultiTask:
            tester = TesterMultiTask(
                cache_path=self.__cache_path,
                results_path=self.__results_path,
                device=device,
                data_loader=data_loader,
                exp_key=exp_key,
                load_epoch=load_epoch,
            )
        elif trainer_cls == TrainerMultiCrit:
            tester = TesterMultiCrit(
                cache_path=self.__cache_path,
                results_path=self.__results_path,
                device=device,
                data_loader=data_loader,
                exp_key=exp_key,
                load_epoch=load_epoch,
            )
        else:
            raise ValueError(f"Unsupported trainer type: {trainer_cls}")
        tester.eval()

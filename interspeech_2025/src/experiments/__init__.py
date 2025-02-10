from __future__ import annotations
import torch
from pathlib import Path
from typing import Literal

from ..model import MFCCDD, Wav2Vec, UnispeechSAT
from ..tasks_generator import TaskGenerator
from ..data_loader import (
    CachedDataLoader,
    DataLoader,
    EmoDB,
    CommonVoiceDeltaSegment20,
    LibrispeechDevClean,
)
from .trainer import Trainer
from .trainer_multicrit import TrainerMultiCrit
from .testers.tester import Tester
from .testers.tester_multicrit import TesterMultiCrit


class Runner:

    extra_audios_bucket_size = 2000
    max_extra_db_audio_length = 2  # seconds
    random_seed = 42

    # fmt: off
    _exp = {
        # MFCC + Deltas
        "mfccdd_phrase_0_25_emodb": ["phrase", [0], MFCCDD, 2000, Trainer, 25, EmoDB],
        "mfccdd_phrase_0_50_emodb": ["phrase", [0], MFCCDD, 2000, Trainer, 50, EmoDB],
        "mfccdd_phrase_1_25_emodb": ["phrase", [1], MFCCDD, 2000, Trainer, 25, EmoDB],
        "mfccdd_phrase_1_50_emodb": ["phrase", [1], MFCCDD, 2000, Trainer, 50, EmoDB],
        "mfccdd_phrase_4_25_emodb": ["phrase", [4], MFCCDD, 2000, Trainer, 25, EmoDB],
        "mfccdd_phrase_4_50_emodb": ["phrase", [4], MFCCDD, 2000, Trainer, 50, EmoDB],
        "mfccdd_phrase_0_25_commonvoice": ["phrase", [0], MFCCDD, 2000, Trainer, 25, CommonVoiceDeltaSegment20],
        "mfccdd_phrase_0_50_commonvoice": ["phrase", [0], MFCCDD, 2000, Trainer, 50, CommonVoiceDeltaSegment20],
        "mfccdd_phrase_1_25_commonvoice": ["phrase", [1], MFCCDD, 2000, Trainer, 25, CommonVoiceDeltaSegment20],
        "mfccdd_phrase_1_50_commonvoice": ["phrase", [1], MFCCDD, 2000, Trainer, 50, CommonVoiceDeltaSegment20],
        "mfccdd_phrase_4_25_commonvoice": ["phrase", [4], MFCCDD, 2000, Trainer, 25, CommonVoiceDeltaSegment20],
        "mfccdd_phrase_4_50_commonvoice": ["phrase", [4], MFCCDD, 2000, Trainer, 50, CommonVoiceDeltaSegment20],
        "mfccdd_phrase_0_25_librispeech": ["phrase", [0], MFCCDD, 2000, Trainer, 25, LibrispeechDevClean],
        "mfccdd_phrase_0_50_librispeech": ["phrase", [0], MFCCDD, 2000, Trainer, 50, LibrispeechDevClean],
        "mfccdd_phrase_1_25_librispeech": ["phrase", [1], MFCCDD, 2000, Trainer, 25, LibrispeechDevClean],
        "mfccdd_phrase_1_50_librispeech": ["phrase", [1], MFCCDD, 2000, Trainer, 50, LibrispeechDevClean],
        "mfccdd_phrase_4_25_librispeech": ["phrase", [4], MFCCDD, 2000, Trainer, 25, LibrispeechDevClean],
        "mfccdd_phrase_4_50_librispeech": ["phrase", [4], MFCCDD, 2000, Trainer, 50, LibrispeechDevClean],

        "mfccdd_a_0_25_emodb": ["a_n", [0], MFCCDD, 2000, Trainer, 25, EmoDB],
        "mfccdd_a_0_50_emodb": ["a_n", [0], MFCCDD, 2000, Trainer, 50, EmoDB],
        "mfccdd_a_1_25_emodb": ["a_n", [1], MFCCDD, 2000, Trainer, 25, EmoDB],
        "mfccdd_a_1_50_emodb": ["a_n", [1], MFCCDD, 2000, Trainer, 50, EmoDB],
        "mfccdd_a_4_25_emodb": ["a_n", [4], MFCCDD, 2000, Trainer, 25, EmoDB],
        "mfccdd_a_4_50_emodb": ["a_n", [4], MFCCDD, 2000, Trainer, 50, EmoDB],
        "mfccdd_a_0_25_commonvoice": ["a_n", [0], MFCCDD, 2000, Trainer, 25, CommonVoiceDeltaSegment20],
        "mfccdd_a_0_50_commonvoice": ["a_n", [0], MFCCDD, 2000, Trainer, 50, CommonVoiceDeltaSegment20],
        "mfccdd_a_1_25_commonvoice": ["a_n", [1], MFCCDD, 2000, Trainer, 25, CommonVoiceDeltaSegment20],
        "mfccdd_a_1_50_commonvoice": ["a_n", [1], MFCCDD, 2000, Trainer, 50, CommonVoiceDeltaSegment20],
        "mfccdd_a_4_25_commonvoice": ["a_n", [4], MFCCDD, 2000, Trainer, 25, CommonVoiceDeltaSegment20],
        "mfccdd_a_4_50_commonvoice": ["a_n", [4], MFCCDD, 2000, Trainer, 50, CommonVoiceDeltaSegment20],
        "mfccdd_a_0_25_librispeech": ["a_n", [0], MFCCDD, 2000, Trainer, 25, LibrispeechDevClean],
        "mfccdd_a_0_50_librispeech": ["a_n", [0], MFCCDD, 2000, Trainer, 50, LibrispeechDevClean],
        "mfccdd_a_1_25_librispeech": ["a_n", [1], MFCCDD, 2000, Trainer, 25, LibrispeechDevClean],
        "mfccdd_a_1_50_librispeech": ["a_n", [1], MFCCDD, 2000, Trainer, 50, LibrispeechDevClean],
        "mfccdd_a_4_25_librispeech": ["a_n", [4], MFCCDD, 2000, Trainer, 25, LibrispeechDevClean],
        "mfccdd_a_4_50_librispeech": ["a_n", [4], MFCCDD, 2000, Trainer, 50, LibrispeechDevClean],

        # Wav2Vec
        "wav2vec_phrase_0_25_emodb": ["phrase", [0], Wav2Vec, 200, Trainer, 25, EmoDB],
        "wav2vec_phrase_0_50_emodb": ["phrase", [0], Wav2Vec, 200, Trainer, 50, EmoDB],
        "wav2vec_phrase_1_25_emodb": ["phrase", [1], Wav2Vec, 200, Trainer, 25, EmoDB],
        "wav2vec_phrase_1_50_emodb": ["phrase", [1], Wav2Vec, 200, Trainer, 50, EmoDB],
        "wav2vec_phrase_4_25_emodb": ["phrase", [4], Wav2Vec, 200, Trainer, 25, EmoDB],
        "wav2vec_phrase_4_50_emodb": ["phrase", [4], Wav2Vec, 200, Trainer, 50, EmoDB],
        "wav2vec_phrase_0_25_commonvoice": ["phrase", [0], Wav2Vec, 200, Trainer, 25, CommonVoiceDeltaSegment20],
        "wav2vec_phrase_0_50_commonvoice": ["phrase", [0], Wav2Vec, 200, Trainer, 50, CommonVoiceDeltaSegment20],
        "wav2vec_phrase_1_25_commonvoice": ["phrase", [1], Wav2Vec, 200, Trainer, 25, CommonVoiceDeltaSegment20],
        "wav2vec_phrase_1_50_commonvoice": ["phrase", [1], Wav2Vec, 200, Trainer, 50, CommonVoiceDeltaSegment20],
        "wav2vec_phrase_4_25_commonvoice": ["phrase", [4], Wav2Vec, 200, Trainer, 25, CommonVoiceDeltaSegment20],
        "wav2vec_phrase_4_50_commonvoice": ["phrase", [4], Wav2Vec, 200, Trainer, 50, CommonVoiceDeltaSegment20],
        "wav2vec_phrase_0_25_librispeech": ["phrase", [0], Wav2Vec, 200, Trainer, 25, LibrispeechDevClean],
        "wav2vec_phrase_0_50_librispeech": ["phrase", [0], Wav2Vec, 200, Trainer, 50, LibrispeechDevClean],
        "wav2vec_phrase_1_25_librispeech": ["phrase", [1], Wav2Vec, 200, Trainer, 25, LibrispeechDevClean],
        "wav2vec_phrase_1_50_librispeech": ["phrase", [1], Wav2Vec, 200, Trainer, 50, LibrispeechDevClean],
        "wav2vec_phrase_4_25_librispeech": ["phrase", [4], Wav2Vec, 200, Trainer, 25, LibrispeechDevClean],
        "wav2vec_phrase_4_50_librispeech": ["phrase", [4], Wav2Vec, 200, Trainer, 50, LibrispeechDevClean],

        "wav2vec_a_0_25_emodb": ["a_n", [0], Wav2Vec, 200, Trainer, 25, EmoDB],
        "wav2vec_a_0_50_emodb": ["a_n", [0], Wav2Vec, 200, Trainer, 50, EmoDB],
        "wav2vec_a_1_25_emodb": ["a_n", [1], Wav2Vec, 200, Trainer, 25, EmoDB],
        "wav2vec_a_1_50_emodb": ["a_n", [1], Wav2Vec, 200, Trainer, 50, EmoDB],
        "wav2vec_a_4_25_emodb": ["a_n", [4], Wav2Vec, 200, Trainer, 25, EmoDB],
        "wav2vec_a_4_50_emodb": ["a_n", [4], Wav2Vec, 200, Trainer, 50, EmoDB],
        "wav2vec_a_0_25_commonvoice": ["a_n", [0], Wav2Vec, 200, Trainer, 25, CommonVoiceDeltaSegment20],
        "wav2vec_a_0_50_commonvoice": ["a_n", [0], Wav2Vec, 200, Trainer, 50, CommonVoiceDeltaSegment20],
        "wav2vec_a_1_25_commonvoice": ["a_n", [1], Wav2Vec, 200, Trainer, 25, CommonVoiceDeltaSegment20],
        "wav2vec_a_1_50_commonvoice": ["a_n", [1], Wav2Vec, 200, Trainer, 50, CommonVoiceDeltaSegment20],
        "wav2vec_a_4_25_commonvoice": ["a_n", [4], Wav2Vec, 200, Trainer, 25, CommonVoiceDeltaSegment20],
        "wav2vec_a_4_50_commonvoice": ["a_n", [4], Wav2Vec, 200, Trainer, 50, CommonVoiceDeltaSegment20],
        "wav2vec_a_0_25_librispeech": ["a_n", [0], Wav2Vec, 200, Trainer, 25, LibrispeechDevClean],
        "wav2vec_a_0_50_librispeech": ["a_n", [0], Wav2Vec, 200, Trainer, 50, LibrispeechDevClean],
        "wav2vec_a_1_25_librispeech": ["a_n", [1], Wav2Vec, 200, Trainer, 25, LibrispeechDevClean],
        "wav2vec_a_1_50_librispeech": ["a_n", [1], Wav2Vec, 200, Trainer, 50, LibrispeechDevClean],
        "wav2vec_a_4_25_librispeech": ["a_n", [4], Wav2Vec, 200, Trainer, 25, LibrispeechDevClean],
        "wav2vec_a_4_50_librispeech": ["a_n", [4], Wav2Vec, 200, Trainer, 50, LibrispeechDevClean],

        # UnispeechSAT
        "unispeechSAT_phrase_0_25_emodb": ["phrase", [0], UnispeechSAT, 200, Trainer, 25, EmoDB],
        "unispeechSAT_phrase_0_50_emodb": ["phrase", [0], UnispeechSAT, 200, Trainer, 50, EmoDB],
        "unispeechSAT_phrase_1_25_emodb": ["phrase", [1], UnispeechSAT, 200, Trainer, 25, EmoDB],
        "unispeechSAT_phrase_1_50_emodb": ["phrase", [1], UnispeechSAT, 200, Trainer, 50, EmoDB],
        "unispeechSAT_phrase_4_25_emodb": ["phrase", [4], UnispeechSAT, 200, Trainer, 25, EmoDB],
        "unispeechSAT_phrase_4_50_emodb": ["phrase", [4], UnispeechSAT, 200, Trainer, 50, EmoDB],
        "unispeechSAT_phrase_0_25_commonvoice": ["phrase", [0], UnispeechSAT, 200, Trainer, 25, CommonVoiceDeltaSegment20],
        "unispeechSAT_phrase_0_50_commonvoice": ["phrase", [0], UnispeechSAT, 200, Trainer, 50, CommonVoiceDeltaSegment20],
        "unispeechSAT_phrase_1_25_commonvoice": ["phrase", [1], UnispeechSAT, 200, Trainer, 25, CommonVoiceDeltaSegment20],
        "unispeechSAT_phrase_1_50_commonvoice": ["phrase", [1], UnispeechSAT, 200, Trainer, 50, CommonVoiceDeltaSegment20],
        "unispeechSAT_phrase_4_25_commonvoice": ["phrase", [4], UnispeechSAT, 200, Trainer, 25, CommonVoiceDeltaSegment20],
        "unispeechSAT_phrase_4_50_commonvoice": ["phrase", [4], UnispeechSAT, 200, Trainer, 50, CommonVoiceDeltaSegment20],
        "unispeechSAT_phrase_0_25_librispeech": ["phrase", [0], UnispeechSAT, 200, Trainer, 25, LibrispeechDevClean],
        "unispeechSAT_phrase_0_50_librispeech": ["phrase", [0], UnispeechSAT, 200, Trainer, 50, LibrispeechDevClean],
        "unispeechSAT_phrase_1_25_librispeech": ["phrase", [1], UnispeechSAT, 200, Trainer, 25, LibrispeechDevClean],
        "unispeechSAT_phrase_1_50_librispeech": ["phrase", [1], UnispeechSAT, 200, Trainer, 50, LibrispeechDevClean],
        "unispeechSAT_phrase_4_25_librispeech": ["phrase", [4], UnispeechSAT, 200, Trainer, 25, LibrispeechDevClean],
        "unispeechSAT_phrase_4_50_librispeech": ["phrase", [4], UnispeechSAT, 200, Trainer, 50, LibrispeechDevClean],

        "unispeechSAT_a_0_25_emodb": ["a_n", [0], UnispeechSAT, 200, Trainer, 25, EmoDB],
        "unispeechSAT_a_0_50_emodb": ["a_n", [0], UnispeechSAT, 200, Trainer, 50, EmoDB],
        "unispeechSAT_a_1_25_emodb": ["a_n", [1], UnispeechSAT, 200, Trainer, 25, EmoDB],
        "unispeechSAT_a_1_50_emodb": ["a_n", [1], UnispeechSAT, 200, Trainer, 50, EmoDB],
        "unispeechSAT_a_4_25_emodb": ["a_n", [4], UnispeechSAT, 200, Trainer, 25, EmoDB],
        "unispeechSAT_a_4_50_emodb": ["a_n", [4], UnispeechSAT, 200, Trainer, 50, EmoDB],
        "unispeechSAT_a_0_25_commonvoice": ["a_n", [0], UnispeechSAT, 200, Trainer, 25, CommonVoiceDeltaSegment20],
        "unispeechSAT_a_0_50_commonvoice": ["a_n", [0], UnispeechSAT, 200, Trainer, 50, CommonVoiceDeltaSegment20],
        "unispeechSAT_a_1_25_commonvoice": ["a_n", [1], UnispeechSAT, 200, Trainer, 25, CommonVoiceDeltaSegment20],
        "unispeechSAT_a_1_50_commonvoice": ["a_n", [1], UnispeechSAT, 200, Trainer, 50, CommonVoiceDeltaSegment20],
        "unispeechSAT_a_4_25_commonvoice": ["a_n", [4], UnispeechSAT, 200, Trainer, 25, CommonVoiceDeltaSegment20],
        "unispeechSAT_a_4_50_commonvoice": ["a_n", [4], UnispeechSAT, 200, Trainer, 50, CommonVoiceDeltaSegment20],
        "unispeechSAT_a_0_25_librispeech": ["a_n", [0], UnispeechSAT, 200, Trainer, 25, LibrispeechDevClean],
        "unispeechSAT_a_0_50_librispeech": ["a_n", [0], UnispeechSAT, 200, Trainer, 50, LibrispeechDevClean],
        "unispeechSAT_a_1_25_librispeech": ["a_n", [1], UnispeechSAT, 200, Trainer, 25, LibrispeechDevClean],
        "unispeechSAT_a_1_50_librispeech": ["a_n", [1], UnispeechSAT, 200, Trainer, 50, LibrispeechDevClean],
        "unispeechSAT_a_4_25_librispeech": ["a_n", [4], UnispeechSAT, 200, Trainer, 25, LibrispeechDevClean],
        "unispeechSAT_a_4_50_librispeech": ["a_n", [4], UnispeechSAT, 200, Trainer, 50, LibrispeechDevClean],
    }
    # fmt: on

    EXP_KEYS = Literal[tuple(_exp.keys())]

    def __init__(
        self,
        tasks_generator: TaskGenerator,
        cache_path: Path,
        results_path: Path,
        research_data_path: Path,
    ):
        self.__tasks_generator = tasks_generator
        self.__cache_path = cache_path
        self.__results_path = results_path
        self.__research_data_path = Path(f"{research_data_path}/data")

    def train(
        self,
        exp_key: Runner.EXP_KEYS,  # type: ignore
        tboard_enabled: bool,
        use_cache_loader: bool,
        limit_vram: float | None,
    ):
        device = torch.device("cuda")
        (
            task_key,
            diag_levels,
            feature_cls,
            num_epochs,
            trainer_cls,
            percent_injection,
            extra_db_cls,
        ) = self._exp[exp_key]
        batch_size = 16
        lr = 1e-5
        if limit_vram is not None:
            torch.cuda.set_per_process_memory_fraction(limit_vram, device=None)
            torch.cuda.empty_cache()

        task = self.__tasks_generator.load_task(
            task=task_key,
            diag_level=max(diag_levels),
        )
        extra_db = extra_db_cls(
            research_data_path=self.__research_data_path,
            max_audio_length=self.max_extra_db_audio_length,
            sample_rate=task.sample_rate,
            random_seed=self.random_seed,
            extra_audios_bucket_size=self.extra_audios_bucket_size,
        )
        feature_function = (
            feature_cls(device=device, sampling_rate=task.sample_rate)
            if feature_cls is not None
            else None
        )
        if use_cache_loader:
            cache_path = Path(
                f"{self.__cache_path}/.data_loader/{task_key}/{feature_cls.__name__}"
            )
            data_loader = CachedDataLoader(
                extra_db=extra_db,
                percent_injection=percent_injection,
                random_seed=self.random_seed,
                shuffle_train=True,
                batch_size=batch_size,
                device=device,
                task=task,
                feature_function=feature_function,
                diag_levels=diag_levels,
                return_ids=False,
                cache_path=cache_path,
                test_only=False,
                allow_inter_level_comparison=False,
            )
        else:
            data_loader = DataLoader(
                extra_db=extra_db,
                percent_injection=percent_injection,
                random_seed=self.random_seed,
                shuffle_train=True,
                batch_size=batch_size,
                device=device,
                task=task,
                feature_function=feature_function,
                diag_levels=diag_levels,
                return_ids=False,
                test_only=False,
                allow_inter_level_comparison=False,
            )
        if trainer_cls == Trainer:
            trainer = Trainer(
                cache_path=self.__cache_path,
                device=device,
                data_loader=data_loader,
                exp_key=exp_key,
                num_epochs=num_epochs,
                tboard_enabled=tboard_enabled,
                lr=lr,
            )
        elif trainer_cls == TrainerMultiCrit:
            trainer = TrainerMultiCrit(
                cache_path=self.__cache_path,
                device=device,
                data_loader=data_loader,
                exp_key=exp_key,
                num_epochs=num_epochs,
                tboard_enabled=tboard_enabled,
                lr=lr,
            )
        else:
            raise ValueError(f"Unsupported trainer type: {trainer_cls}")
        trainer.run()

    def test(self, exp_key: Runner.EXP_KEYS, load_epoch: int):  # type: ignore
        device = torch.device("cuda")
        (
            task_key,
            diag_levels,
            feature_cls,
            trainer_cls,
            percent_injection,
            extra_db_cls,
        ) = self._exp[exp_key]
        batch_size = 16
        task = self.__tasks_generator.load_task(
            task=task_key,
            diag_level=max(diag_levels),
        )
        extra_db = extra_db_cls(
            research_data_path=self.__research_data_path,
            max_audio_length=self.max_extra_db_audio_length,
            sample_rate=task.sample_rate,
            random_seed=self.random_seed,
            extra_audios_bucket_size=self.extra_audios_bucket_size,
        )
        feature_function = (
            feature_cls(device=device, sampling_rate=task.sample_rate)
            if feature_cls is not None
            else None
        )
        data_loader = DataLoader(
            extra_db=extra_db,
            percent_injection=percent_injection,
            random_seed=self.random_seed,
            shuffle_train=True,
            batch_size=batch_size,
            device=device,
            task=task,
            feature_function=feature_function,
            diag_levels=diag_levels,
            return_ids=True,
            test_only=False,
            allow_inter_level_comparison=False,
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
        tester.test()

from __future__ import annotations
import torch
from pathlib import Path
from typing import Literal

from ..model import (
    UnispeechSAT,
    MFCCDD,
    Wav2Vec,
    Compare2016Functional,
    Compare2016LLD,
    Compare2016LLDDE,
    EGEMapsv2Functional,
    EGEMapsv2LLD,
)
from ..tasks_generator import TaskGenerator
from ..data_loader import CachedDataLoader, DataLoader
from .trainer import Trainer
from .trainer_multitask import TrainerMultiTask
from .trainer_multicrit import TrainerMultiCrit
from .tester import Tester
from .tester_multitask import TesterMultiTask
from .tester_multicrit import TesterMultiCrit


class Runner:

    # fmt: off
    __exp = {
        # Compare2016 Functional
        "compare2016_func_phrase_0": ["phrase", [0], Compare2016Functional, 2000, 16, Trainer, 1e-5],
        "compare2016_func_a_0": ["a_n", [0], Compare2016Functional, 2000, 16, Trainer, 1e-5],
        "compare2016_func_i_0": ["i_n", [0], Compare2016Functional, 2000, 16, Trainer, 1e-5],
        "compare2016_func_u_0": ["u_n", [0], Compare2016Functional, 2000, 16, Trainer, 1e-5],
        "compare2016_func_phrase_1": ["phrase", [1], Compare2016Functional, 2000, 16, Trainer, 1e-5],
        "compare2016_func_a_1": ["a_n", [1], Compare2016Functional, 2000, 16, Trainer, 1e-5],
        "compare2016_func_i_1": ["i_n", [1], Compare2016Functional, 2000, 16, Trainer, 1e-5],
        "compare2016_func_u_1": ["u_n", [1], Compare2016Functional, 2000, 16, Trainer, 1e-5],
        "compare2016_func_phrase_2": ["phrase", [2], Compare2016Functional, 2000, 16, Trainer, 1e-5],
        "compare2016_func_a_2": ["a_n", [2], Compare2016Functional, 2000, 16, Trainer, 1e-5],
        "compare2016_func_i_2": ["i_n", [2], Compare2016Functional, 2000, 16, Trainer, 1e-5],
        "compare2016_func_u_2": ["u_n", [2], Compare2016Functional, 2000, 16, Trainer, 1e-5],
        "compare2016_func_phrase_3": ["phrase", [3], Compare2016Functional, 2000, 16, Trainer, 1e-5],
        "compare2016_func_a_3": ["a_n", [3], Compare2016Functional, 2000, 16, Trainer, 1e-5],
        "compare2016_func_i_3": ["i_n", [3], Compare2016Functional, 2000, 16, Trainer, 1e-5],
        "compare2016_func_u_3": ["u_n", [3], Compare2016Functional, 2000, 16, Trainer, 1e-5],
        "compare2016_func_phrase_4": ["phrase", [4], Compare2016Functional, 2000, 16, Trainer, 1e-5],
        "compare2016_func_a_4": ["a_n", [4], Compare2016Functional, 2000, 16, Trainer, 1e-5],
        "compare2016_func_i_4": ["i_n", [4], Compare2016Functional, 2000, 16, Trainer, 1e-5],
        "compare2016_func_u_4": ["u_n", [4], Compare2016Functional, 2000, 16, Trainer, 1e-5],

        # Compare2016 LLD
        "compare2016_lld_phrase_0": ["phrase", [0], Compare2016LLD, 2000, 16, Trainer, 1e-5],
        "compare2016_lld_a_0": ["a_n", [0], Compare2016LLD, 2000, 16, Trainer, 1e-5],
        "compare2016_lld_i_0": ["i_n", [0], Compare2016LLD, 2000, 16, Trainer, 1e-5],
        "compare2016_lld_u_0": ["u_n", [0], Compare2016LLD, 2000, 16, Trainer, 1e-5],
        "compare2016_lld_phrase_1": ["phrase", [1], Compare2016LLD, 2000, 16, Trainer, 1e-5],
        "compare2016_lld_a_1": ["a_n", [1], Compare2016LLD, 2000, 16, Trainer, 1e-5],
        "compare2016_lld_i_1": ["i_n", [1], Compare2016LLD, 2000, 16, Trainer, 1e-5],
        "compare2016_lld_u_1": ["u_n", [1], Compare2016LLD, 2000, 16, Trainer, 1e-5],
        "compare2016_lld_phrase_2": ["phrase", [2], Compare2016LLD, 2000, 16, Trainer, 1e-5],
        "compare2016_lld_a_2": ["a_n", [2], Compare2016LLD, 2000, 16, Trainer, 1e-5],
        "compare2016_lld_i_2": ["i_n", [2], Compare2016LLD, 2000, 16, Trainer, 1e-5],
        "compare2016_lld_u_2": ["u_n", [2], Compare2016LLD, 2000, 16, Trainer, 1e-5],
        "compare2016_lld_phrase_3": ["phrase", [3], Compare2016LLD, 2000, 16, Trainer, 1e-5],
        "compare2016_lld_a_3": ["a_n", [3], Compare2016LLD, 2000, 16, Trainer, 1e-5],
        "compare2016_lld_i_3": ["i_n", [3], Compare2016LLD, 2000, 16, Trainer, 1e-5],
        "compare2016_lld_u_3": ["u_n", [3], Compare2016LLD, 2000, 16, Trainer, 1e-5],
        "compare2016_lld_phrase_4": ["phrase", [4], Compare2016LLD, 2000, 16, Trainer, 1e-5],
        "compare2016_lld_a_4": ["a_n", [4], Compare2016LLD, 2000, 16, Trainer, 1e-5],
        "compare2016_lld_i_4": ["i_n", [4], Compare2016LLD, 2000, 16, Trainer, 1e-5],
        "compare2016_lld_u_4": ["u_n", [4], Compare2016LLD, 2000, 16, Trainer, 1e-5],

        # Compare2016 LLD_DE
        "compare2016_lldde_phrase_0": ["phrase", [0], Compare2016LLDDE, 2000, 16, Trainer, 1e-5],
        "compare2016_lldde_a_0": ["a_n", [0], Compare2016LLDDE, 2000, 16, Trainer, 1e-5],
        "compare2016_lldde_i_0": ["i_n", [0], Compare2016LLDDE, 2000, 16, Trainer, 1e-5],
        "compare2016_lldde_u_0": ["u_n", [0], Compare2016LLDDE, 2000, 16, Trainer, 1e-5],
        "compare2016_lldde_phrase_1": ["phrase", [1], Compare2016LLDDE, 2000, 16, Trainer, 1e-5],
        "compare2016_lldde_a_1": ["a_n", [1], Compare2016LLDDE, 2000, 16, Trainer, 1e-5],
        "compare2016_lldde_i_1": ["i_n", [1], Compare2016LLDDE, 2000, 16, Trainer, 1e-5],
        "compare2016_lldde_u_1": ["u_n", [1], Compare2016LLDDE, 2000, 16, Trainer, 1e-5],
        "compare2016_lldde_phrase_2": ["phrase", [2], Compare2016LLDDE, 2000, 16, Trainer, 1e-5],
        "compare2016_lldde_a_2": ["a_n", [2], Compare2016LLDDE, 2000, 16, Trainer, 1e-5],
        "compare2016_lldde_i_2": ["i_n", [2], Compare2016LLDDE, 2000, 16, Trainer, 1e-5],
        "compare2016_lldde_u_2": ["u_n", [2], Compare2016LLDDE, 2000, 16, Trainer, 1e-5],
        "compare2016_lldde_phrase_3": ["phrase", [3], Compare2016LLDDE, 2000, 16, Trainer, 1e-5],
        "compare2016_lldde_a_3": ["a_n", [3], Compare2016LLDDE, 2000, 16, Trainer, 1e-5],
        "compare2016_lldde_i_3": ["i_n", [3], Compare2016LLDDE, 2000, 16, Trainer, 1e-5],
        "compare2016_lldde_u_3": ["u_n", [3], Compare2016LLDDE, 2000, 16, Trainer, 1e-5],
        "compare2016_lldde_phrase_4": ["phrase", [4], Compare2016LLDDE, 2000, 16, Trainer, 1e-5],
        "compare2016_lldde_a_4": ["a_n", [4], Compare2016LLDDE, 2000, 16, Trainer, 1e-5],
        "compare2016_lldde_i_4": ["i_n", [4], Compare2016LLDDE, 2000, 16, Trainer, 1e-5],
        "compare2016_lldde_u_4": ["u_n", [4], Compare2016LLDDE, 2000, 16, Trainer, 1e-5],

        # EGEMapsv2 Functional
        "egemapsv2_func_phrase_0": ["phrase", [0], EGEMapsv2Functional, 2000, 16, Trainer, 1e-5],
        "egemapsv2_func_a_0": ["a_n", [0], EGEMapsv2Functional, 2000, 16, Trainer, 1e-5],
        "egemapsv2_func_i_0": ["i_n", [0], EGEMapsv2Functional, 2000, 16, Trainer, 1e-5],
        "egemapsv2_func_u_0": ["u_n", [0], EGEMapsv2Functional, 2000, 16, Trainer, 1e-5],
        "egemapsv2_func_phrase_1": ["phrase", [1], EGEMapsv2Functional, 2000, 16, Trainer, 1e-5],
        "egemapsv2_func_a_1": ["a_n", [1], EGEMapsv2Functional, 2000, 16, Trainer, 1e-5],
        "egemapsv2_func_i_1": ["i_n", [1], EGEMapsv2Functional, 2000, 16, Trainer, 1e-5],
        "egemapsv2_func_u_1": ["u_n", [1], EGEMapsv2Functional, 2000, 16, Trainer, 1e-5],
        "egemapsv2_func_phrase_2": ["phrase", [2], EGEMapsv2Functional, 2000, 16, Trainer, 1e-5],
        "egemapsv2_func_a_2": ["a_n", [2], EGEMapsv2Functional, 2000, 16, Trainer, 1e-5],
        "egemapsv2_func_i_2": ["i_n", [2], EGEMapsv2Functional, 2000, 16, Trainer, 1e-5],
        "egemapsv2_func_u_2": ["u_n", [2], EGEMapsv2Functional, 2000, 16, Trainer, 1e-5],
        "egemapsv2_func_phrase_3": ["phrase", [3], EGEMapsv2Functional, 2000, 16, Trainer, 1e-5],
        "egemapsv2_func_a_3": ["a_n", [3], EGEMapsv2Functional, 2000, 16, Trainer, 1e-5],
        "egemapsv2_func_i_3": ["i_n", [3], EGEMapsv2Functional, 2000, 16, Trainer, 1e-5],
        "egemapsv2_func_u_3": ["u_n", [3], EGEMapsv2Functional, 2000, 16, Trainer, 1e-5],
        "egemapsv2_func_phrase_4": ["phrase", [4], EGEMapsv2Functional, 2000, 16, Trainer, 1e-5],
        "egemapsv2_func_a_4": ["a_n", [4], EGEMapsv2Functional, 2000, 16, Trainer, 1e-5],
        "egemapsv2_func_i_4": ["i_n", [4], EGEMapsv2Functional, 2000, 16, Trainer, 1e-5],
        "egemapsv2_func_u_4": ["u_n", [4], EGEMapsv2Functional, 2000, 16, Trainer, 1e-5],

        # EGEMapsv2 LLD
        "egemapsv2_lld_phrase_0": ["phrase", [0], EGEMapsv2LLD, 2000, 16, Trainer, 1e-5],
        "egemapsv2_lld_a_0": ["a_n", [0], EGEMapsv2LLD, 2000, 16, Trainer, 1e-5],
        "egemapsv2_lld_i_0": ["i_n", [0], EGEMapsv2LLD, 2000, 16, Trainer, 1e-5],
        "egemapsv2_lld_u_0": ["u_n", [0], EGEMapsv2LLD, 2000, 16, Trainer, 1e-5],
        "egemapsv2_lld_phrase_1": ["phrase", [1], EGEMapsv2LLD, 2000, 16, Trainer, 1e-5],
        "egemapsv2_lld_a_1": ["a_n", [1], EGEMapsv2LLD, 2000, 16, Trainer, 1e-5],
        "egemapsv2_lld_i_1": ["i_n", [1], EGEMapsv2LLD, 2000, 16, Trainer, 1e-5],
        "egemapsv2_lld_u_1": ["u_n", [1], EGEMapsv2LLD, 2000, 16, Trainer, 1e-5],
        "egemapsv2_lld_phrase_2": ["phrase", [2], EGEMapsv2LLD, 2000, 16, Trainer, 1e-5],
        "egemapsv2_lld_a_2": ["a_n", [2], EGEMapsv2LLD, 2000, 16, Trainer, 1e-5],
        "egemapsv2_lld_i_2": ["i_n", [2], EGEMapsv2LLD, 2000, 16, Trainer, 1e-5],
        "egemapsv2_lld_u_2": ["u_n", [2], EGEMapsv2LLD, 2000, 16, Trainer, 1e-5],
        "egemapsv2_lld_phrase_3": ["phrase", [3], EGEMapsv2LLD, 2000, 16, Trainer, 1e-5],
        "egemapsv2_lld_a_3": ["a_n", [3], EGEMapsv2LLD, 2000, 16, Trainer, 1e-5],
        "egemapsv2_lld_i_3": ["i_n", [3], EGEMapsv2LLD, 2000, 16, Trainer, 1e-5],
        "egemapsv2_lld_u_3": ["u_n", [3], EGEMapsv2LLD, 2000, 16, Trainer, 1e-5],
        "egemapsv2_lld_phrase_4": ["phrase", [4], EGEMapsv2LLD, 2000, 16, Trainer, 1e-5],
        "egemapsv2_lld_a_4": ["a_n", [4], EGEMapsv2LLD, 2000, 16, Trainer, 1e-5],
        "egemapsv2_lld_i_4": ["i_n", [4], EGEMapsv2LLD, 2000, 16, Trainer, 1e-5],
        "egemapsv2_lld_u_4": ["u_n", [4], EGEMapsv2LLD, 2000, 16, Trainer, 1e-5],

        # MFCC + Deltas
        "mfccdd_phrase_0": ["phrase", [0], MFCCDD, 2000, 16, Trainer, 1e-5],
        "mfccdd_a_0": ["a_n", [0], MFCCDD, 2000, 16, Trainer, 1e-5],
        "mfccdd_i_0": ["i_n", [0], MFCCDD, 2000, 16, Trainer, 1e-5],
        "mfccdd_u_0": ["u_n", [0], MFCCDD, 2000, 16, Trainer, 1e-5],
        "mfccdd_phrase_1": ["phrase", [1], MFCCDD, 2000, 16, Trainer, 1e-5],
        "mfccdd_a_1": ["a_n", [1], MFCCDD, 2000, 16, Trainer, 1e-5],
        "mfccdd_i_1": ["i_n", [1], MFCCDD, 2000, 16, Trainer, 1e-5],
        "mfccdd_u_1": ["u_n", [1], MFCCDD, 2000, 16, Trainer, 1e-5],
        "mfccdd_phrase_2": ["phrase", [2], MFCCDD, 2000, 16, Trainer, 1e-5],
        "mfccdd_a_2": ["a_n", [2], MFCCDD, 2000, 16, Trainer, 1e-5],
        "mfccdd_i_2": ["i_n", [2], MFCCDD, 2000, 16, Trainer, 1e-5],
        "mfccdd_u_2": ["u_n", [2], MFCCDD, 2000, 16, Trainer, 1e-5],
        "mfccdd_phrase_3": ["phrase", [3], MFCCDD, 2000, 16, Trainer, 1e-5],
        "mfccdd_a_3": ["a_n", [3], MFCCDD, 2000, 16, Trainer, 1e-5],
        "mfccdd_i_3": ["i_n", [3], MFCCDD, 2000, 16, Trainer, 1e-5],
        "mfccdd_u_3": ["u_n", [3], MFCCDD, 2000, 16, Trainer, 1e-5],
        "mfccdd_phrase_4": ["phrase", [4], MFCCDD, 2000, 16, Trainer, 1e-5],
        "mfccdd_a_4": ["a_n", [4], MFCCDD, 2000, 16, Trainer, 1e-5],
        "mfccdd_i_4": ["i_n", [4], MFCCDD, 2000, 16, Trainer, 1e-5],
        "mfccdd_u_4": ["u_n", [4], MFCCDD, 2000, 16, Trainer, 1e-5],
        "mfccdd_all_0": ["all", [0], MFCCDD, 2000, 16, Trainer, 1e-5],
        "mfccdd_all_1": ["all", [1], MFCCDD, 2000, 16, Trainer, 1e-5],
        "mfccdd_all_2": ["all", [2], MFCCDD, 2000, 16, Trainer, 1e-5],
        "mfccdd_all_3": ["all", [3], MFCCDD, 2000, 16, Trainer, 1e-5],
        "mfccdd_all_4": ["all", [4], MFCCDD, 2000, 16, Trainer, 1e-5],
        "mc_mfccdd_phrase_0+2": ["phrase", [0, 2], MFCCDD, 2000, 16, TrainerMultiCrit, 1e-5],
        "mc_mfccdd_a_0+2": ["a_n", [0, 2], MFCCDD, 2000, 16, TrainerMultiCrit, 1e-5],
        "mc_mfccdd_i_0+2": ["i_n", [0, 2], MFCCDD, 2000, 16, TrainerMultiCrit, 1e-5],
        "mc_mfccdd_u_0+2": ["u_n", [0, 2], MFCCDD, 2000, 16, TrainerMultiCrit, 1e-5],
        "mc_mfccdd_phrase_0+3": ["phrase", [0, 3], MFCCDD, 2000, 16, TrainerMultiCrit, 1e-5],
        "mc_mfccdd_a_0+3": ["a_n", [0, 3], MFCCDD, 2000, 16, TrainerMultiCrit, 1e-5],
        "mc_mfccdd_i_0+3": ["i_n", [0, 3], MFCCDD, 2000, 16, TrainerMultiCrit, 1e-5],
        "mc_mfccdd_u_0+3": ["u_n", [0, 3], MFCCDD, 2000, 16, TrainerMultiCrit, 1e-5],
        "mc_mfccdd_phrase_0+4": ["phrase", [0, 4], MFCCDD, 2000, 16, TrainerMultiCrit, 1e-5],
        "mc_mfccdd_a_0+4": ["a_n", [0, 4], MFCCDD, 2000, 16, TrainerMultiCrit, 1e-5],
        "mc_mfccdd_i_0+4": ["i_n", [0, 4], MFCCDD, 2000, 16, TrainerMultiCrit, 1e-5],
        "mc_mfccdd_u_0+4": ["u_n", [0, 4], MFCCDD, 2000, 16, TrainerMultiCrit, 1e-5],

        # Wav2Vec
        "wav2vec_phrase_0": ["phrase", [0], Wav2Vec, 200, 16, Trainer, 1e-5],
        "wav2vec_a_0": ["a_n", [0], Wav2Vec, 200, 16, Trainer, 1e-5],
        "wav2vec_i_0": ["i_n", [0], Wav2Vec, 200, 16, Trainer, 1e-5],
        "wav2vec_u_0": ["u_n", [0], Wav2Vec, 200, 16, Trainer, 1e-5],
        "wav2vec_phrase_1": ["phrase", [1], Wav2Vec, 200, 16, Trainer, 1e-5],
        "wav2vec_a_1": ["a_n", [1], Wav2Vec, 200, 16, Trainer, 1e-5],
        "wav2vec_i_1": ["i_n", [1], Wav2Vec, 200, 16, Trainer, 1e-5],
        "wav2vec_u_1": ["u_n", [1], Wav2Vec, 200, 16, Trainer, 1e-5],
        "wav2vec_phrase_2": ["phrase", [2], Wav2Vec, 200, 16, Trainer, 1e-5],
        "wav2vec_a_2": ["a_n", [2], Wav2Vec, 200, 16, Trainer, 1e-5],
        "wav2vec_i_2": ["i_n", [2], Wav2Vec, 200, 16, Trainer, 1e-5],
        "wav2vec_u_2": ["u_n", [2], Wav2Vec, 200, 16, Trainer, 1e-5],
        "wav2vec_phrase_3": ["phrase", [3], Wav2Vec, 200, 16, Trainer, 1e-5],
        "wav2vec_a_3": ["a_n", [3], Wav2Vec, 200, 16, Trainer, 1e-5],
        "wav2vec_i_3": ["i_n", [3], Wav2Vec, 200, 16, Trainer, 1e-5],
        "wav2vec_u_3": ["u_n", [3], Wav2Vec, 200, 16, Trainer, 1e-5],
        "wav2vec_phrase_4": ["phrase", [4], Wav2Vec, 200, 16, Trainer, 1e-5],
        "wav2vec_a_4": ["a_n", [4], Wav2Vec, 200, 16, Trainer, 1e-5],
        "wav2vec_i_4": ["i_n", [4], Wav2Vec, 200, 16, Trainer, 1e-5],
        "wav2vec_u_4": ["u_n", [4], Wav2Vec, 200, 16, Trainer, 1e-5],
        "wav2vec_all_0": ["all", [0], Wav2Vec, 200, 16, Trainer, 1e-5],
        "wav2vec_all_1": ["all", [1], Wav2Vec, 200, 16, Trainer, 1e-5],
        "wav2vec_all_2": ["all", [2], Wav2Vec, 200, 16, Trainer, 1e-5],
        "wav2vec_all_3": ["all", [3], Wav2Vec, 200, 16, Trainer, 1e-5],
        "wav2vec_all_4": ["all", [4], Wav2Vec, 200, 16, Trainer, 1e-5],

        "mc_wav2vec_phrase_0+2": ["phrase", [0, 2], Wav2Vec, 200, 16, TrainerMultiCrit, 1e-5],
        "mc_wav2vec_a_0+2": ["a_n", [0, 2], Wav2Vec, 200, 16, TrainerMultiCrit, 1e-5],
        "mc_wav2vec_i_0+2": ["i_n", [0, 2], Wav2Vec, 200, 16, TrainerMultiCrit, 1e-5],
        "mc_wav2vec_u_0+2": ["u_n", [0, 2], Wav2Vec, 200, 16, TrainerMultiCrit, 1e-5],
        "mc_wav2vec_phrase_0+3": ["phrase", [0, 3], Wav2Vec, 200, 16, TrainerMultiCrit, 1e-5],
        "mc_wav2vec_a_0+3": ["a_n", [0, 3], Wav2Vec, 200, 16, TrainerMultiCrit, 1e-5],
        "mc_wav2vec_i_0+3": ["i_n", [0, 3], Wav2Vec, 200, 16, TrainerMultiCrit, 1e-5],
        "mc_wav2vec_u_0+3": ["u_n", [0, 3], Wav2Vec, 200, 16, TrainerMultiCrit, 1e-5],
        "mc_wav2vec_phrase_0+4": ["phrase", [0, 4], Wav2Vec, 200, 16, TrainerMultiCrit, 1e-5],
        "mc_wav2vec_a_0+4": ["a_n", [0, 4], Wav2Vec, 200, 16, TrainerMultiCrit, 1e-5],
        "mc_wav2vec_i_0+4": ["i_n", [0, 4], Wav2Vec, 200, 16, TrainerMultiCrit, 1e-5],
        "mc_wav2vec_u_0+4": ["u_n", [0, 4], Wav2Vec, 200, 16, TrainerMultiCrit, 1e-5],


        # UnispeechSAT
        "unispeechSAT_phrase_0": ["phrase", [0], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "unispeechSAT_a_0": ["a_n", [0], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "unispeechSAT_i_0": ["i_n", [0], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "unispeechSAT_u_0": ["u_n", [0], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "unispeechSAT_phrase_1": ["phrase", [1], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "unispeechSAT_a_1": ["a_n", [1], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "unispeechSAT_i_1": ["i_n", [1], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "unispeechSAT_u_1": ["u_n", [1], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "unispeechSAT_phrase_2": ["phrase", [2], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "unispeechSAT_a_2": ["a_n", [2], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "unispeechSAT_i_2": ["i_n", [2], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "unispeechSAT_u_2": ["u_n", [2], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "unispeechSAT_phrase_3": ["phrase", [3], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "unispeechSAT_a_3": ["a_n", [3], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "unispeechSAT_i_3": ["i_n", [3], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "unispeechSAT_u_3": ["u_n", [3], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "unispeechSAT_phrase_4": ["phrase", [4], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "unispeechSAT_a_4": ["a_n", [4], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "unispeechSAT_i_4": ["i_n", [4], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "unispeechSAT_u_4": ["u_n", [4], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "unispeechSAT_all_0": ["all", [0], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "unispeechSAT_all_1": ["all", [1], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "unispeechSAT_all_2": ["all", [2], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "unispeechSAT_all_3": ["all", [3], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "unispeechSAT_all_4": ["all", [4], UnispeechSAT, 200, 16, Trainer, 1e-5],
        # Multitask experiments
        "unispeechSAT_phrase_0+1": ["phrase", [0, 1], UnispeechSAT, 200, 16, TrainerMultiTask, 1e-5],
        "unispeechSAT_phrase_0+2": ["phrase", [0, 2], UnispeechSAT, 200, 16, TrainerMultiTask, 1e-5],
        "unispeechSAT_phrase_0+3": ["phrase", [0, 3], UnispeechSAT, 200, 16, TrainerMultiTask, 1e-5],
        "unispeechSAT_phrase_0+4": ["phrase", [0, 4], UnispeechSAT, 200, 16, TrainerMultiTask, 1e-5],
        "unispeechSAT_phrase_1+2": ["phrase", [1, 2], UnispeechSAT, 200, 16, TrainerMultiTask, 1e-5],
        "unispeechSAT_phrase_1+3": ["phrase", [1, 3], UnispeechSAT, 200, 16, TrainerMultiTask, 1e-5],
        "unispeechSAT_phrase_1+4": ["phrase", [1, 4], UnispeechSAT, 200, 16, TrainerMultiTask, 1e-5],
        "unispeechSAT_phrase_2+3": ["phrase", [2, 3], UnispeechSAT, 200, 16, TrainerMultiTask, 1e-5],
        "unispeechSAT_phrase_2+4": ["phrase", [2, 4], UnispeechSAT, 200, 16, TrainerMultiTask, 1e-5],
        "unispeechSAT_phrase_0+1+2": ["phrase", [0, 1, 2], UnispeechSAT, 200, 16, TrainerMultiTask, 1e-5],
        "unispeechSAT_phrase_0+1+3": ["phrase", [0, 1, 3], UnispeechSAT, 200, 16, TrainerMultiTask, 1e-5],
        "unispeechSAT_phrase_0+1+4": ["phrase", [0, 1, 4], UnispeechSAT, 200, 16, TrainerMultiTask, 1e-5],
        "unispeechSAT_phrase_0+2+3": ["phrase", [0, 2, 3], UnispeechSAT, 200, 16, TrainerMultiTask, 1e-5],
        "unispeechSAT_phrase_0+2+4": ["phrase", [0, 2, 4], UnispeechSAT, 200, 16, TrainerMultiTask, 1e-5],
        "unispeechSAT_phrase_1+2+3": ["phrase", [1, 2, 3], UnispeechSAT, 200, 16, TrainerMultiTask, 1e-5],
        "unispeechSAT_phrase_1+2+4": ["phrase", [1, 2, 4], UnispeechSAT, 200, 16, TrainerMultiTask, 1e-5],
        "unispeechSAT_phrase_0+1+2+3": ["phrase", [0, 1, 2, 3], UnispeechSAT, 200, 16, TrainerMultiTask, 1e-5],
        "unispeechSAT_phrase_0+1+2+4": ["phrase", [0, 1, 2, 4], UnispeechSAT, 200, 16, TrainerMultiTask, 1e-5],
        "unispeechSAT_phrase_0+1+2+3+4": ["phrase", [0, 1, 2, 3, 4], UnispeechSAT, 200, 16, TrainerMultiTask, 1e-5],
        # Single output multitask experiments
        "mc_unispeechSAT_phrase_0+1": ["phrase", [0, 1], UnispeechSAT, 200, 16, TrainerMultiCrit, 1e-5],
        "mc_unispeechSAT_phrase_0+2": ["phrase", [0, 2], UnispeechSAT, 200, 16, TrainerMultiCrit, 1e-5],
        "mc_unispeechSAT_phrase_0+3": ["phrase", [0, 3], UnispeechSAT, 200, 16, TrainerMultiCrit, 1e-5],
        "mc_unispeechSAT_phrase_0+4": ["phrase", [0, 4], UnispeechSAT, 200, 16, TrainerMultiCrit, 1e-5],
        "mc_unispeechSAT_phrase_1+2": ["phrase", [1, 2], UnispeechSAT, 200, 16, TrainerMultiCrit, 1e-5],
        "mc_unispeechSAT_phrase_1+3": ["phrase", [1, 3], UnispeechSAT, 200, 16, TrainerMultiCrit, 1e-5],
        "mc_unispeechSAT_phrase_1+4": ["phrase", [1, 4], UnispeechSAT, 200, 16, TrainerMultiCrit, 1e-5],
        "mc_unispeechSAT_phrase_2+3": ["phrase", [2, 3], UnispeechSAT, 200, 16, TrainerMultiCrit, 1e-5],
        "mc_unispeechSAT_phrase_2+4": ["phrase", [2, 4], UnispeechSAT, 200, 16, TrainerMultiCrit, 1e-5],
        "mc_unispeechSAT_phrase_0+1+2": ["phrase", [0, 1, 2], UnispeechSAT, 200, 16, TrainerMultiCrit, 1e-5],
        "mc_unispeechSAT_phrase_0+1+3": ["phrase", [0, 1, 3], UnispeechSAT, 200, 16, TrainerMultiCrit, 1e-5],
        "mc_unispeechSAT_phrase_0+1+4": ["phrase", [0, 1, 4], UnispeechSAT, 200, 16, TrainerMultiCrit, 1e-5],
        "mc_unispeechSAT_phrase_0+2+3": ["phrase", [0, 2, 3], UnispeechSAT, 200, 16, TrainerMultiCrit, 1e-5],
        "mc_unispeechSAT_phrase_0+2+4": ["phrase", [0, 2, 4], UnispeechSAT, 200, 16, TrainerMultiCrit, 1e-5],
        "mc_unispeechSAT_phrase_1+2+3": ["phrase", [1, 2, 3], UnispeechSAT, 200, 16, TrainerMultiCrit, 1e-5],
        "mc_unispeechSAT_phrase_1+2+4": ["phrase", [1, 2, 4], UnispeechSAT, 200, 16, TrainerMultiCrit, 1e-5],
        "mc_unispeechSAT_phrase_0+1+2+3": ["phrase", [0, 1, 2, 3], UnispeechSAT, 200, 16, TrainerMultiCrit, 1e-5],
        "mc_unispeechSAT_phrase_0+1+2+4": ["phrase", [0, 1, 2, 4], UnispeechSAT, 200, 16, TrainerMultiCrit, 1e-5],
        "mc_unispeechSAT_phrase_0+1+2+3+4": ["phrase", [0, 1, 2, 3, 4], UnispeechSAT, 200, 16, TrainerMultiCrit, 1e-5],


        # Other diagnostic maps experiments
        ## Compton_2022
        "Compton_2022-mfccdd_a_n_1": ["Compton_2022-a_n", [1], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Compton_2022-mfccdd_a_n_2": ["Compton_2022-a_n", [2], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Compton_2022-mfccdd_a_n_3": ["Compton_2022-a_n", [3], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Compton_2022-mc_mfccdd_a_n_0+1+2+3": ["Compton_2022-a_n", [0, 1, 2, 3], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Compton_2022-mfccdd_i_n_1": ["Compton_2022-i_n", [1], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Compton_2022-mfccdd_i_n_2": ["Compton_2022-i_n", [2], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Compton_2022-mfccdd_i_n_3": ["Compton_2022-i_n", [3], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Compton_2022-mc_mfccdd_i_n_0+1+2+3": ["Compton_2022-i_n", [0, 1, 2, 3], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Compton_2022-mfccdd_phrase_1": ["Compton_2022-phrase", [1], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Compton_2022-mfccdd_phrase_2": ["Compton_2022-phrase", [2], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Compton_2022-mfccdd_phrase_3": ["Compton_2022-phrase", [3], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Compton_2022-mc_mfccdd_phrase_0+1+2+3": ["Compton_2022-phrase", [0, 1, 2, 3], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Compton_2022-mfccdd_u_n_1": ["Compton_2022-u_n", [1], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Compton_2022-mfccdd_u_n_2": ["Compton_2022-u_n", [2], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Compton_2022-mfccdd_u_n_3": ["Compton_2022-u_n", [3], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Compton_2022-mc_mfccdd_u_n_0+1+2+3": ["Compton_2022-u_n", [0, 1, 2, 3], MFCCDD, 2000, 16, Trainer, 1e-5],

        "Compton_2022-wav2vec_a_n_1": ["Compton_2022-a_n", [1], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Compton_2022-wav2vec_a_n_2": ["Compton_2022-a_n", [2], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Compton_2022-wav2vec_a_n_3": ["Compton_2022-a_n", [3], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Compton_2022-mc_wav2vec_a_n_0+1+2+3": ["Compton_2022-a_n", [0, 1, 2, 3], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Compton_2022-wav2vec_i_n_1": ["Compton_2022-i_n", [1], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Compton_2022-wav2vec_i_n_2": ["Compton_2022-i_n", [2], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Compton_2022-wav2vec_i_n_3": ["Compton_2022-i_n", [3], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Compton_2022-mc_wav2vec_i_n_0+1+2+3": ["Compton_2022-i_n", [0, 1, 2, 3], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Compton_2022-wav2vec_phrase_1": ["Compton_2022-phrase", [1], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Compton_2022-wav2vec_phrase_2": ["Compton_2022-phrase", [2], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Compton_2022-wav2vec_phrase_3": ["Compton_2022-phrase", [3], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Compton_2022-mc_wav2vec_phrase_0+1+2+3": ["Compton_2022-phrase", [0, 1, 2, 3], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Compton_2022-wav2vec_u_n_1": ["Compton_2022-u_n", [1], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Compton_2022-wav2vec_u_n_2": ["Compton_2022-u_n", [2], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Compton_2022-wav2vec_u_n_3": ["Compton_2022-u_n", [3], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Compton_2022-mc_wav2vec_u_n_0+1+2+3": ["Compton_2022-u_n", [0, 1, 2, 3], Wav2Vec, 200, 16, Trainer, 1e-5],

        "Compton_2022-unispeechSAT_a_n_1": ["Compton_2022-a_n", [1], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Compton_2022-unispeechSAT_a_n_2": ["Compton_2022-a_n", [2], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Compton_2022-unispeechSAT_a_n_3": ["Compton_2022-a_n", [3], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Compton_2022-mc_unispeechSAT_a_n_0+1+2+3": ["Compton_2022-a_n", [0, 1, 2, 3], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Compton_2022-unispeechSAT_i_n_1": ["Compton_2022-i_n", [1], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Compton_2022-unispeechSAT_i_n_2": ["Compton_2022-i_n", [2], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Compton_2022-unispeechSAT_i_n_3": ["Compton_2022-i_n", [3], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Compton_2022-mc_unispeechSAT_i_n_0+1+2+3": ["Compton_2022-i_n", [0, 1, 2, 3], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Compton_2022-unispeechSAT_phrase_1": ["Compton_2022-phrase", [1], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Compton_2022-unispeechSAT_phrase_2": ["Compton_2022-phrase", [2], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Compton_2022-unispeechSAT_phrase_3": ["Compton_2022-phrase", [3], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Compton_2022-mc_unispeechSAT_phrase_0+1+2+3": ["Compton_2022-phrase", [0, 1, 2, 3], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Compton_2022-unispeechSAT_u_n_1": ["Compton_2022-u_n", [1], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Compton_2022-unispeechSAT_u_n_2": ["Compton_2022-u_n", [2], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Compton_2022-unispeechSAT_u_n_3": ["Compton_2022-u_n", [3], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Compton_2022-mc_unispeechSAT_u_n_0+1+2+3": ["Compton_2022-u_n", [0, 1, 2, 3], UnispeechSAT, 200, 16, Trainer, 1e-5],

        ## daSilvaMoura_2024
        "daSilvaMoura_2024-mfccdd_a_n_1": ["daSilvaMoura_2024-a_n", [1], MFCCDD, 2000, 16, Trainer, 1e-5],
        "daSilvaMoura_2024-mfccdd_a_n_2": ["daSilvaMoura_2024-a_n", [2], MFCCDD, 2000, 16, Trainer, 1e-5],
        "daSilvaMoura_2024-mc_mfccdd_a_n_0+1+2": ["daSilvaMoura_2024-a_n", [0, 1, 2], MFCCDD, 2000, 16, Trainer, 1e-5],
        "daSilvaMoura_2024-mfccdd_i_n_1": ["daSilvaMoura_2024-i_n", [1], MFCCDD, 2000, 16, Trainer, 1e-5],
        "daSilvaMoura_2024-mfccdd_i_n_2": ["daSilvaMoura_2024-i_n", [2], MFCCDD, 2000, 16, Trainer, 1e-5],
        "daSilvaMoura_2024-mc_mfccdd_i_n_0+1+2": ["daSilvaMoura_2024-i_n", [0, 1, 2], MFCCDD, 2000, 16, Trainer, 1e-5],
        "daSilvaMoura_2024-mfccdd_phrase_1": ["daSilvaMoura_2024-phrase", [1], MFCCDD, 2000, 16, Trainer, 1e-5],
        "daSilvaMoura_2024-mfccdd_phrase_2": ["daSilvaMoura_2024-phrase", [2], MFCCDD, 2000, 16, Trainer, 1e-5],
        "daSilvaMoura_2024-mc_mfccdd_phrase_0+1+2": ["daSilvaMoura_2024-phrase", [0, 1, 3], MFCCDD, 2000, 16, Trainer, 1e-5],
        "daSilvaMoura_2024-mfccdd_u_n_1": ["daSilvaMoura_2024-u_n", [1], MFCCDD, 2000, 16, Trainer, 1e-5],
        "daSilvaMoura_2024-mfccdd_u_n_2": ["daSilvaMoura_2024-u_n", [2], MFCCDD, 2000, 16, Trainer, 1e-5],
        "daSilvaMoura_2024-mc_mfccdd_u_n_0+1+2": ["daSilvaMoura_2024-u_n", [0, 1, 2], MFCCDD, 2000, 16, Trainer, 1e-5],

        "daSilvaMoura_2024-wav2vec_a_n_1": ["daSilvaMoura_2024-a_n", [1], Wav2Vec, 200, 16, Trainer, 1e-5],
        "daSilvaMoura_2024-wav2vec_a_n_2": ["daSilvaMoura_2024-a_n", [2], Wav2Vec, 200, 16, Trainer, 1e-5],
        "daSilvaMoura_2024-mc_wav2vec_a_n_0+1+2": ["daSilvaMoura_2024-a_n", [0, 1, 2], Wav2Vec, 200, 16, Trainer, 1e-5],
        "daSilvaMoura_2024-wav2vec_i_n_1": ["daSilvaMoura_2024-i_n", [1], Wav2Vec, 200, 16, Trainer, 1e-5],
        "daSilvaMoura_2024-wav2vec_i_n_2": ["daSilvaMoura_2024-i_n", [2], Wav2Vec, 200, 16, Trainer, 1e-5],
        "daSilvaMoura_2024-mc_wav2vec_i_n_0+1+2": ["daSilvaMoura_2024-i_n", [0, 1, 2], Wav2Vec, 200, 16, Trainer, 1e-5],
        "daSilvaMoura_2024-wav2vec_phrase_1": ["daSilvaMoura_2024-phrase", [1], Wav2Vec, 200, 16, Trainer, 1e-5],
        "daSilvaMoura_2024-wav2vec_phrase_2": ["daSilvaMoura_2024-phrase", [2], Wav2Vec, 200, 16, Trainer, 1e-5],
        "daSilvaMoura_2024-mc_wav2vec_phrase_0+1+2": ["daSilvaMoura_2024-phrase", [0, 1, 3], Wav2Vec, 200, 16, Trainer, 1e-5],
        "daSilvaMoura_2024-wav2vec_u_n_1": ["daSilvaMoura_2024-u_n", [1], Wav2Vec, 200, 16, Trainer, 1e-5],
        "daSilvaMoura_2024-wav2vec_u_n_2": ["daSilvaMoura_2024-u_n", [2], Wav2Vec, 200, 16, Trainer, 1e-5],
        "daSilvaMoura_2024-mc_wav2vec_u_n_0+1+2": ["daSilvaMoura_2024-u_n", [0, 1, 2], Wav2Vec, 200, 16, Trainer, 1e-5],

        "daSilvaMoura_2024-unispeechSAT_a_n_1": ["daSilvaMoura_2024-a_n", [1], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "daSilvaMoura_2024-unispeechSAT_a_n_2": ["daSilvaMoura_2024-a_n", [2], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "daSilvaMoura_2024-mc_unispeechSAT_a_n_0+1+2": ["daSilvaMoura_2024-a_n", [0, 1, 2], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "daSilvaMoura_2024-unispeechSAT_i_n_1": ["daSilvaMoura_2024-i_n", [1], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "daSilvaMoura_2024-unispeechSAT_i_n_2": ["daSilvaMoura_2024-i_n", [2], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "daSilvaMoura_2024-mc_unispeechSAT_i_n_0+1+2": ["daSilvaMoura_2024-i_n", [0, 1, 2], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "daSilvaMoura_2024-unispeechSAT_phrase_1": ["daSilvaMoura_2024-phrase", [1], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "daSilvaMoura_2024-unispeechSAT_phrase_2": ["daSilvaMoura_2024-phrase", [2], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "daSilvaMoura_2024-mc_unispeechSAT_phrase_0+1+2": ["daSilvaMoura_2024-phrase", [0, 1, 3], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "daSilvaMoura_2024-unispeechSAT_u_n_1": ["daSilvaMoura_2024-u_n", [1], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "daSilvaMoura_2024-unispeechSAT_u_n_2": ["daSilvaMoura_2024-u_n", [2], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "daSilvaMoura_2024-mc_unispeechSAT_u_n_0+1+2": ["daSilvaMoura_2024-u_n", [0, 1, 2], UnispeechSAT, 200, 16, Trainer, 1e-5],

        ## Sztaho_2018
        "Sztaho_2018-mfccdd_a_n_1": ["Sztaho_2018-a_n", [1], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Sztaho_2018-mfccdd_a_n_2": ["Sztaho_2018-a_n", [2], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Sztaho_2018-mfccdd_a_n_3": ["Sztaho_2018-a_n", [3], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Sztaho_2018-mc_mfccdd_a_n_0+1+2+3": ["Sztaho_2018-a_n", [0, 1, 2, 3], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Sztaho_2018-mfccdd_i_n_1": ["Sztaho_2018-i_n", [1], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Sztaho_2018-mfccdd_i_n_2": ["Sztaho_2018-i_n", [2], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Sztaho_2018-mfccdd_i_n_3": ["Sztaho_2018-i_n", [3], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Sztaho_2018-mc_mfccdd_i_n_0+1+2+3": ["Sztaho_2018-i_n", [0, 1, 2, 3], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Sztaho_2018-mfccdd_phrase_1": ["Sztaho_2018-phrase", [1], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Sztaho_2018-mfccdd_phrase_2": ["Sztaho_2018-phrase", [2], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Sztaho_2018-mfccdd_phrase_3": ["Sztaho_2018-phrase", [3], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Sztaho_2018-mc_mfccdd_phrase_0+1+2+3": ["Sztaho_2018-phrase", [0, 1, 2, 3], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Sztaho_2018-mfccdd_u_n_1": ["Sztaho_2018-u_n", [1], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Sztaho_2018-mfccdd_u_n_2": ["Sztaho_2018-u_n", [2], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Sztaho_2018-mfccdd_u_n_3": ["Sztaho_2018-u_n", [3], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Sztaho_2018-mc_mfccdd_u_n_0+1+2+3": ["Sztaho_2018-u_n", [0, 1, 2, 3], MFCCDD, 2000, 16, Trainer, 1e-5],

        "Sztaho_2018-wav2vec_a_n_1": ["Sztaho_2018-a_n", [1], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Sztaho_2018-wav2vec_a_n_2": ["Sztaho_2018-a_n", [2], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Sztaho_2018-wav2vec_a_n_3": ["Sztaho_2018-a_n", [3], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Sztaho_2018-mc_wav2vec_a_n_0+1+2+3": ["Sztaho_2018-a_n", [0, 1, 2, 3], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Sztaho_2018-wav2vec_i_n_1": ["Sztaho_2018-i_n", [1], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Sztaho_2018-wav2vec_i_n_2": ["Sztaho_2018-i_n", [2], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Sztaho_2018-wav2vec_i_n_3": ["Sztaho_2018-i_n", [3], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Sztaho_2018-mc_wav2vec_i_n_0+1+2+3": ["Sztaho_2018-i_n", [0, 1, 2, 3], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Sztaho_2018-wav2vec_phrase_1": ["Sztaho_2018-phrase", [1], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Sztaho_2018-wav2vec_phrase_2": ["Sztaho_2018-phrase", [2], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Sztaho_2018-wav2vec_phrase_3": ["Sztaho_2018-phrase", [3], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Sztaho_2018-mc_wav2vec_phrase_0+1+2+3": ["Sztaho_2018-phrase", [0, 1, 2, 3], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Sztaho_2018-wav2vec_u_n_1": ["Sztaho_2018-u_n", [1], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Sztaho_2018-wav2vec_u_n_2": ["Sztaho_2018-u_n", [2], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Sztaho_2018-wav2vec_u_n_3": ["Sztaho_2018-u_n", [3], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Sztaho_2018-mc_wav2vec_u_n_0+1+2+3": ["Sztaho_2018-u_n", [0, 1, 2, 3], Wav2Vec, 200, 16, Trainer, 1e-5],

        "Sztaho_2018-unispeechSAT_a_n_1": ["Sztaho_2018-a_n", [1], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Sztaho_2018-unispeechSAT_a_n_2": ["Sztaho_2018-a_n", [2], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Sztaho_2018-unispeechSAT_a_n_3": ["Sztaho_2018-a_n", [3], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Sztaho_2018-mc_unispeechSAT_a_n_0+1+2+3": ["Sztaho_2018-a_n", [0, 1, 2, 3], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Sztaho_2018-unispeechSAT_i_n_1": ["Sztaho_2018-i_n", [1], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Sztaho_2018-unispeechSAT_i_n_2": ["Sztaho_2018-i_n", [2], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Sztaho_2018-unispeechSAT_i_n_3": ["Sztaho_2018-i_n", [3], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Sztaho_2018-mc_unispeechSAT_i_n_0+1+2+3": ["Sztaho_2018-i_n", [0, 1, 2, 3], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Sztaho_2018-unispeechSAT_phrase_1": ["Sztaho_2018-phrase", [1], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Sztaho_2018-unispeechSAT_phrase_2": ["Sztaho_2018-phrase", [2], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Sztaho_2018-unispeechSAT_phrase_3": ["Sztaho_2018-phrase", [3], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Sztaho_2018-mc_unispeechSAT_phrase_0+1+2+3": ["Sztaho_2018-phrase", [0, 1, 2, 3], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Sztaho_2018-unispeechSAT_u_n_1": ["Sztaho_2018-u_n", [1], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Sztaho_2018-unispeechSAT_u_n_2": ["Sztaho_2018-u_n", [2], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Sztaho_2018-unispeechSAT_u_n_3": ["Sztaho_2018-u_n", [3], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Sztaho_2018-mc_unispeechSAT_u_n_0+1+2+3": ["Sztaho_2018-u_n", [0, 1, 2, 3], UnispeechSAT, 200, 16, Trainer, 1e-5],

        ## Zaim_2023
        "Zaim_2023-mfccdd_a_n_1": ["Zaim_2023-a_n", [1], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Zaim_2023-mc_mfccdd_a_n_0+1+2": ["Zaim_2023-a_n", [0, 1, 2], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Zaim_2023-mfccdd_i_n_1": ["Zaim_2023-i_n", [1], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Zaim_2023-mfccdd_i_n_2": ["Zaim_2023-i_n", [2], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Zaim_2023-mc_mfccdd_i_n_0+1+2": ["Zaim_2023-i_n", [0, 1, 2], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Zaim_2023-mfccdd_phrase_1": ["Zaim_2023-phrase", [1], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Zaim_2023-mfccdd_phrase_2": ["Zaim_2023-phrase", [2], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Zaim_2023-mc_mfccdd_phrase_0+1+2": ["Zaim_2023-phrase", [0, 1, 3], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Zaim_2023-mfccdd_u_n_1": ["Zaim_2023-u_n", [1], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Zaim_2023-mfccdd_u_n_2": ["Zaim_2023-u_n", [2], MFCCDD, 2000, 16, Trainer, 1e-5],
        "Zaim_2023-mc_mfccdd_u_n_0+1+2": ["Zaim_2023-u_n", [0, 1, 2], MFCCDD, 2000, 16, Trainer, 1e-5],

        "Zaim_2023-wav2vec_a_n_1": ["Zaim_2023-a_n", [1], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Zaim_2023-wav2vec_a_n_2": ["Zaim_2023-a_n", [2], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Zaim_2023-mc_wav2vec_a_n_0+1+2": ["Zaim_2023-a_n", [0, 1, 2], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Zaim_2023-wav2vec_i_n_1": ["Zaim_2023-i_n", [1], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Zaim_2023-wav2vec_i_n_2": ["Zaim_2023-i_n", [2], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Zaim_2023-mc_wav2vec_i_n_0+1+2": ["Zaim_2023-i_n", [0, 1, 2], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Zaim_2023-wav2vec_phrase_1": ["Zaim_2023-phrase", [1], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Zaim_2023-wav2vec_phrase_2": ["Zaim_2023-phrase", [2], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Zaim_2023-mc_wav2vec_phrase_0+1+2": ["Zaim_2023-phrase", [0, 1, 3], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Zaim_2023-wav2vec_u_n_1": ["Zaim_2023-u_n", [1], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Zaim_2023-wav2vec_u_n_2": ["Zaim_2023-u_n", [2], Wav2Vec, 200, 16, Trainer, 1e-5],
        "Zaim_2023-mc_wav2vec_u_n_0+1+2": ["Zaim_2023-u_n", [0, 1, 2], Wav2Vec, 200, 16, Trainer, 1e-5],

        "Zaim_2023-unispeechSAT_a_n_1": ["Zaim_2023-a_n", [1], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Zaim_2023-unispeechSAT_a_n_2": ["Zaim_2023-a_n", [2], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Zaim_2023-mc_unispeechSAT_a_n_0+1+2": ["Zaim_2023-a_n", [0, 1, 2], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Zaim_2023-unispeechSAT_i_n_1": ["Zaim_2023-i_n", [1], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Zaim_2023-unispeechSAT_i_n_2": ["Zaim_2023-i_n", [2], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Zaim_2023-mc_unispeechSAT_i_n_0+1+2": ["Zaim_2023-i_n", [0, 1, 2], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Zaim_2023-unispeechSAT_phrase_1": ["Zaim_2023-phrase", [1], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Zaim_2023-unispeechSAT_phrase_2": ["Zaim_2023-phrase", [2], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Zaim_2023-mc_unispeechSAT_phrase_0+1+2": ["Zaim_2023-phrase", [0, 1, 3], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Zaim_2023-unispeechSAT_u_n_1": ["Zaim_2023-u_n", [1], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Zaim_2023-unispeechSAT_u_n_2": ["Zaim_2023-u_n", [2], UnispeechSAT, 200, 16, Trainer, 1e-5],
        "Zaim_2023-mc_unispeechSAT_u_n_0+1+2": ["Zaim_2023-u_n", [0, 1, 2], UnispeechSAT, 200, 16, Trainer, 1e-5],
        
    }
    # fmt: on

    EXP_KEYS = Literal[tuple(__exp.keys())]

    def __init__(
        self,
        tasks_generator: TaskGenerator,
        cache_path: Path,
        results_path: Path,
    ):
        self.__tasks_generator = tasks_generator
        self.__cache_path = cache_path
        self.__results_path = results_path

    def train(
        self, exp_key: Runner.EXP_KEYS, tboard_enabled: bool, use_cache_loader: bool  # type: ignore
    ):
        device = torch.device("cuda")
        (
            task_key,
            diag_levels,
            feature_cls,
            num_epochs,
            batch_size,
            trainer_cls,
            lr,
        ) = self.__exp[exp_key]
        task = self.__tasks_generator.load_task(
            task=task_key,
            diag_level=max(diag_levels),
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
                random_seed=42,
                shuffle_train=True,
                batch_size=batch_size,
                device=device,
                task=task,
                feature_function=feature_function,
                diag_levels=diag_levels,
                return_ids=False,
                cache_path=cache_path,
            )
        else:
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
                lr=lr,
            )
        elif trainer_cls == TrainerMultiTask:
            trainer = TrainerMultiTask(
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
            num_epochs,
            batch_size,
            trainer_cls,
            lr,
        ) = self.__exp[exp_key]
        task = self.__tasks_generator.load_task(
            task=task_key,
            diag_level=max(diag_levels),
        )
        feature_function = (
            feature_cls(device=device, sampling_rate=task.sample_rate)
            if feature_cls is not None
            else None
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
        tester.test()

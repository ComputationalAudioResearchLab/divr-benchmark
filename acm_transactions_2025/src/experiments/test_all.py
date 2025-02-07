import torch
from tqdm import tqdm
from pathlib import Path

from . import Runner
from .trainer import Trainer
from ..data_loader import DataLoader
from ..tasks_generator import TaskGenerator
from .trainer_multicrit import TrainerMultiCrit
from .trainer_multitask import TrainerMultiTask
from ..model import Normalized, NormalizedMultiCrit, NormalizedMultitask, Feature


class TestAll:

    __model_map = {
        Trainer: Normalized,
        TrainerMultiCrit: NormalizedMultiCrit,
        TrainerMultiTask: NormalizedMultitask,
    }
    __ignored_exp = [
        "unispeechSAT_all_0",
        "unispeechSAT_all_1",
        "unispeechSAT_all_2",
        "unispeechSAT_all_3",
        "unispeechSAT_all_4",
        "daSilvaMoura_2024-mc_unispeechSAT_phrase_0+1+2",  # Need to do this one
        "Zaim_2023-mc_mfccdd_phrase_0+1+2",  # Need to do this one
        "Zaim_2023-mc_wav2vec_phrase_0+1+2",  # Need to do this one
        "Zaim_2023-mc_unispeechSAT_phrase_0+1+2",  # Need to do this one
    ]
    __device = torch.device("cpu")
    __sampling_rate = 16000
    __random_seed = 42
    __batch_size = 16

    def __init__(
        self,
        research_data_path: Path,
        cache_path: Path,
        results_path: Path,
    ):
        self.__research_data_path = research_data_path
        self.__cache_path = cache_path
        self.__ckpt_path = Path(f"{cache_path}/checkpoints")
        self.__results_path = results_path
        self.__task_generator = TaskGenerator(
            research_data_path=self.__research_data_path
        )

    def self_test(self):
        tests = {}
        for key, items in Runner._exp.items():
            if key in self.__ignored_exp:
                continue
            (
                task_key,
                diag_levels,
                feature_cls,
                num_epochs,
                batch_size,
                trainer_cls,
                lr,
            ) = items
            model_cls = self.__model_map[trainer_cls]
            diag_levels = ",".join(map(str, diag_levels))
            if feature_cls not in tests:
                tests[feature_cls] = {}
            feature_tests = tests[feature_cls]
            if task_key not in feature_tests:
                feature_tests[task_key] = {}
            task_tests = feature_tests[task_key]
            if diag_levels not in task_tests:
                task_tests[diag_levels] = {}
            task_diag_tests = task_tests[diag_levels]
            if model_cls not in task_diag_tests:
                task_diag_tests[model_cls] = {}
            model_tests = task_diag_tests[model_cls]
            model_tests[key] = self.__find_checkpoints(key)

        total_exps = len(Runner._exp) - len(self.__ignored_exp)
        pbar_top = tqdm(desc="Testing models", total=total_exps)

        for feature_cls, feature_tests in tests.items():
            feature_function: Feature = feature_cls(
                device=self.__device, sampling_rate=self.__sampling_rate
            )
            for task_key, task_tests in feature_tests.items():
                for diag_levels, task_diag_tests in task_tests.items():
                    data_loader = self.__get_data_loader(
                        task_key=task_key,
                        diag_levels_str=diag_levels,
                        feature_function=feature_function,
                    )
                    for model_cls, model_tests in task_diag_tests.items():
                        model = model_cls(
                            input_size=feature_function.feature_size,
                            num_classes=data_loader.num_classes,
                            checkpoint_path=Path("/tmp"),
                        )
                        for model_key, model_key_tests in model_tests.items():
                            pbar_top.update(1)
                            pbar_top.write(f"{model_key}: {len(model_key_tests)}")

    def __get_data_loader(
        self, task_key: str, diag_levels_str: str, feature_function: Feature
    ) -> DataLoader:
        diag_levels = [int(x) for x in diag_levels_str.split(",")]
        task = self.__task_generator.load_task(
            task=task_key,
            diag_level=max(diag_levels),
        )
        return DataLoader(
            random_seed=self.__random_seed,
            shuffle_train=False,
            batch_size=self.__batch_size,
            device=self.__device,
            diag_levels=diag_levels,
            task=task,
            feature_function=feature_function,
            return_ids=False,
        )

    def __find_checkpoints(self, exp_key: str) -> list[Path]:
        checkpoints = sorted(list(Path(f"{self.__ckpt_path}/{exp_key}").glob("*.h5")))
        if len(checkpoints) < 1:
            raise ValueError(f"No checkpoints found for {exp_key}")
        return checkpoints

    def old(self):
        tasks_generator = TaskGenerator(
            research_data_path=self.__research_data_path,
        )
        runner = Runner(
            tasks_generator=tasks_generator,
            cache_path=self.__cache_path,
            results_path=Path(f"{self.__results_path}/test"),
        )

        checkpoints_path = Path(f"{self.__cache_path}/checkpoints")
        checkpoints = sorted(list(checkpoints_path.rglob("*.h5")))
        pbar = tqdm(checkpoints, desc="testing")
        ignored_keys = [
            "Compton_2022-mfccdd_phrase_1",
            "Compton_2022-mfccdd_a_n_1",
            "Compton_2022-mfccdd_i_n_1",
            "Compton_2022-mfccdd_u_n_1",
            "Sztaho_2018-mfccdd_phrase_1",
            "Sztaho_2018-mfccdd_a_n_1",
            "Sztaho_2018-mfccdd_i_n_1",
            "Sztaho_2018-mfccdd_u_n_1",
        ]
        for checkpoint in pbar:
            exp_key = checkpoint.parent.stem
            epoch = int(checkpoint.stem)
            pbar.set_postfix({"exp_key": exp_key, "epoch": epoch})
            results_path = Path(
                f"{self.__results_path}/test/{exp_key}/{epoch}/results.csv"
            )
            if not results_path.is_file() and exp_key not in ignored_keys:
                runner.test(exp_key=exp_key, load_epoch=epoch)

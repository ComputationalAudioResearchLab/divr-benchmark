import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from divr_diagnosis import DiagnosisMap

from .model_cache import ModelCache
from .. import Runner
from ..trainer import Trainer
from ..trainer_sep import TrainerSep
from ...data_loader import BaseDataLoader, DataLoader, CachedDataLoader
from ...tasks_generator import TaskGenerator
from ..trainer_multicrit import TrainerMultiCrit
from ..trainer_multitask import TrainerMultiTask
from ..trainer_transformer import TrainerTransformer
from ...model import (
    Normalized,
    NormalizedMultiCrit,
    NormalizedMultitask,
    Feature,
    SimpleTransformer,
)


class TestAllSelf:

    __model_map = {
        Trainer: Normalized,
        TrainerSep: Normalized,
        TrainerMultiCrit: NormalizedMultiCrit,
        TrainerMultiTask: NormalizedMultitask,
        TrainerTransformer: SimpleTransformer,
    }
    __device = torch.device("cuda")
    __sampling_rate = 16000
    __random_seed = 42
    __batch_size = 1

    def __init__(
        self,
        research_data_path: Path,
        cache_path: Path,
        results_path: Path,
        tasks_path: Path,
    ):
        self.__research_data_path = research_data_path
        self.__cache_path = cache_path
        self.__ckpt_path = Path(f"{cache_path}/checkpoints")
        self.__results_path = Path(f"{results_path}/self")
        self.__results_path.mkdir(parents=True, exist_ok=True)
        self.__task_generator = TaskGenerator(
            research_data_path=self.__research_data_path,
            tasks_path=tasks_path,
        )
        self.__test_func_map = {
            Normalized: self.__test_single,
            NormalizedMultiCrit: self.__test_multi_crit,
            NormalizedMultitask: self.__test_multi_task,
        }

    @torch.no_grad()
    def run(self):
        tests = {}
        completed_keys = self.__get_completed_tests()
        total_exps = 0
        for key, items in Runner._exp.items():
            if key in completed_keys:
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
            if key.startswith("USVAC-"):
                task_key = f"USVAC-{task_key}"
            elif key.startswith("daSilvaMoura-"):
                task_key = f"daSilvaMoura-{task_key}"
            elif key.startswith("superset-"):
                diag_map_key = key.split("-")[1]
                task_key = f"superset-{diag_map_key}-{task_key}"
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
            total_exps += 1
        pbar_top = tqdm(desc="Testing models", total=total_exps)
        model_cache = ModelCache(device=self.__device)

        for feature_cls, feature_tests in tests.items():
            feature_function: Feature = feature_cls(
                device=self.__device, sampling_rate=self.__sampling_rate
            )
            for task_key, task_tests in feature_tests.items():
                for diag_levels_str, task_diag_tests in task_tests.items():
                    diag_levels = [int(x) for x in diag_levels_str.split(",")]
                    if task_key.startswith("USVAC-") or task_key.startswith(
                        "daSilvaMoura-"
                    ):
                        dmap_key, d_task_key = task_key.split("-", maxsplit=1)
                        diagnosis_map = self.__task_generator.get_diagnosis_map(
                            task_key=dmap_key, allow_unmapped=False
                        )
                    elif task_key.startswith("superset-"):
                        prefix, dmap_key, d_task_key = task_key.split("-")
                        diagnosis_map = self.__task_generator.get_diagnosis_map(
                            task_key=dmap_key, allow_unmapped=True
                        )
                    else:
                        diagnosis_map = self.__task_generator.get_diagnosis_map(
                            task_key="USVAC_2025", allow_unmapped=False
                        )
                        d_task_key = task_key
                    data_loader = self.__get_data_loader(
                        task_key=d_task_key,
                        diag_levels=diag_levels,
                        diagnosis_map=diagnosis_map,
                        feature_function=feature_function,
                        cache_path=Path(
                            f"{self.__cache_path}/.data_loader/{d_task_key}/{feature_cls.__name__}"
                        ),
                        return_id_tensor=True,
                    )
                    for model_cls, model_tests in task_diag_tests.items():
                        model = model_cache.get_model(
                            data_loader=data_loader,
                            model_cls=model_cls,
                        )
                        test_func = self.__test_func_map[model_cls]
                        for model_key, model_key_tests in model_tests.items():
                            pbar_top.set_postfix({"model": model_key})
                            results_path = Path(f"{self.__results_path}/{model_key}")
                            results_path.mkdir(exist_ok=True)
                            for epoch_ckpt in model_key_tests:
                                epoch = epoch_ckpt.stem
                                model.load_checkpoint(epoch_ckpt)
                                result = test_func(
                                    data_loader=data_loader,
                                    model=model,
                                    diag_levels=diag_levels,
                                )
                                result.to_csv(
                                    f"{results_path}/{epoch}.csv", index=False
                                )
                            pbar_top.update(1)

    def __get_completed_tests(self):
        completed_keys = []
        for key in Runner._exp:
            completed = True
            ckpts = list(sorted(Path(f"{self.__ckpt_path}/{key}").glob("*.h5")))
            for ckpt in ckpts:
                epoch = ckpt.stem
                if not Path(f"{self.__results_path}/{key}/{epoch}.csv").is_file():
                    completed = False
            if completed:
                completed_keys += [key]
        return completed_keys

    @torch.no_grad()
    def __test_single(
        self,
        data_loader: BaseDataLoader,
        model: Normalized,
        diag_levels: list[int],
    ) -> pd.DataFrame:
        assert len(diag_levels) == 1
        diag_level = diag_levels[0]
        results = []
        all_ids = []
        for batch in tqdm(data_loader.test(), desc="Testing", leave=False):
            inputs, labels, id_tensor, ids = batch
            labels = labels.squeeze(1)
            probabilities, _, _ = model(inputs)
            predicted_labels = probabilities.argmax(dim=1)
            data = torch.cat(
                [labels[:, None], predicted_labels[:, None], probabilities],
                dim=1,
            )
            all_ids += ids
            results += [data]
        results = torch.cat(results, dim=0).round(decimals=2)
        columns = ["actual", "predicted"] + data_loader.unique_diagnosis[diag_level]
        results = pd.DataFrame(
            data=results.cpu().numpy(),
            columns=columns,
        )
        results["id"] = all_ids

        def diag_map(idx):
            return data_loader.idx_to_diag_name(int(idx), diag_level)

        results["actual"] = results["actual"].apply(diag_map)
        results["predicted"] = results["predicted"].apply(diag_map)
        return results

    @torch.no_grad()
    def __test_multi_crit(
        self,
        data_loader: BaseDataLoader,
        model: NormalizedMultiCrit,
        diag_levels: list[int],
    ) -> pd.DataFrame:
        all_results = []
        all_ids = []
        for batch in tqdm(data_loader.test(), desc="Testing", leave=False):
            inputs, labels, id_tensor, ids = batch
            labels = labels.squeeze(1)
            probabilities, _, _ = model(inputs)
            data_at_level = []
            data = []
            for idx, key in enumerate(data_loader.unique_diagnosis):
                labels_at_level = model.labels_at_level(
                    probabilities, level=key
                ).argmax(dim=1)
                data_at_level += [
                    labels[:, idx : idx + 1],
                    labels_at_level[:, None],
                ]
            all_ids += ids
            data = torch.cat(data_at_level + [probabilities], dim=1)
            all_results += [data]
        all_results = torch.cat(all_results, dim=0).round(decimals=2)
        column_names: list[str] = []
        for key, val in data_loader.unique_diagnosis.items():
            column_names += [f"actual_{key}", f"predicted_{key}"]
        column_names += val
        all_results = pd.DataFrame(
            data=all_results.cpu().numpy(),
            columns=column_names,
        )
        all_results["id"] = all_ids
        all_results = all_results[["id"] + column_names]
        for cname in column_names:
            if cname.startswith("actual") or cname.startswith("predicted"):
                prefix, suffix = cname.split("_")
                level = int(suffix)
                all_results[cname] = all_results[cname].apply(
                    lambda idx: data_loader.idx_to_diag_name(int(idx), level)
                )
        return all_results

    @torch.no_grad()
    def __test_multi_task(
        self,
        data_loader: BaseDataLoader,
        model: NormalizedMultitask,
        diag_levels: list[int],
    ) -> pd.DataFrame:
        all_results = []
        all_ids = []
        for batch in tqdm(data_loader.test(), desc="Testing", leave=False):
            inputs, labels, id_tensor, ids = batch
            labels = labels.squeeze(1)
            results = model(inputs)
            data_at_level = []
            data = []
            for i, result in enumerate(results):
                probabilities, _, _ = result
                predicted_labels = probabilities.argmax(dim=1)
                data_at_level += [
                    labels[:, i : i + 1],
                    predicted_labels[:, None],
                    probabilities,
                ]
            all_ids += ids
            data = torch.cat(data_at_level, dim=1)
            all_results += [data]
        all_results = torch.cat(all_results, dim=0).round(decimals=2)
        column_names: list[str] = []
        for key, val in data_loader.unique_diagnosis.items():
            column_names += [f"actual_{key}", f"predicted_{key}"] + val
        all_results = pd.DataFrame(
            data=all_results.cpu().numpy(),
            columns=column_names,
        )
        all_results["id"] = all_ids
        all_results = all_results[["id"] + column_names]
        for cname in column_names:
            if cname.startswith("actual") or cname.startswith("predicted"):
                prefix, suffix = cname.split("_")
                level = int(suffix)
                all_results[cname] = all_results[cname].apply(
                    lambda idx: data_loader.idx_to_diag_name(int(idx), level)
                )
        return all_results

    def __get_cached_data_loader(
        self,
        task_key: str,
        diag_levels: list[int],
        diagnosis_map: DiagnosisMap,
        feature_function: Feature,
        cache_path: Path,
        return_id_tensor: bool,
    ) -> BaseDataLoader:
        task = self.__task_generator.load_task(
            task=task_key,
            diagnosis_map=diagnosis_map,
            diag_level=max(diag_levels),
        )
        return CachedDataLoader(
            random_seed=self.__random_seed,
            shuffle_train=False,
            batch_size=self.__batch_size,
            device=self.__device,
            diag_levels=diag_levels,
            task=task,
            feature_function=feature_function,
            return_ids=True,
            cache_path=cache_path,
            test_only=True,
            allow_inter_level_comparison=False,
            return_id_tensor=return_id_tensor,
        )

    def __get_data_loader(
        self,
        task_key: str,
        diag_levels: list[int],
        diagnosis_map: DiagnosisMap,
        feature_function: Feature,
        cache_path: Path,
        return_id_tensor: bool,
    ) -> BaseDataLoader:
        task = self.__task_generator.load_task(
            task=task_key,
            diagnosis_map=diagnosis_map,
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
            return_ids=True,
            test_only=True,
            allow_inter_level_comparison=False,
            return_id_tensor=return_id_tensor,
        )

    def __find_checkpoints(self, exp_key: str) -> list[Path]:
        checkpoints = sorted(list(Path(f"{self.__ckpt_path}/{exp_key}").glob("*.h5")))
        if len(checkpoints) < 1:
            raise ValueError(f"No checkpoints found for {exp_key}")
        return checkpoints

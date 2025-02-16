import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from divr_diagnosis import DiagnosisMap

from .model_cache import ModelCache
from .. import Runner
from ..trainer import Trainer
from ...data_loader import BaseDataLoader, DataLoader, CachedDataLoader
from ...tasks_generator import TaskGenerator
from ..trainer_multicrit import TrainerMultiCrit
from ..trainer_multitask import TrainerMultiTask
from ...model import (
    Normalized,
    NormalizedMultiCrit,
    NormalizedMultitask,
    Feature,
)


class TestAllCross:

    __model_map = {
        Trainer: Normalized,
        TrainerMultiCrit: NormalizedMultiCrit,
        TrainerMultiTask: NormalizedMultitask,
    }
    __device = torch.device("cuda")
    __sampling_rate = 16000
    __random_seed = 42
    __batch_size = 1
    __cross_test_tasks = [
        # "cross_test_avfad",
        # "cross_test_meei",
        # "cross_test_torgo",
        # "cross_test_uaspeech",
        # "cross_test_uncommon_voice",
        # "cross_test_voiced",
    ]

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
        self.__results_path = Path(f"{results_path}/cross")
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

    def cache_tasks(self):
        feature_classes = set()
        for key, items in Runner._exp.items():
            (
                task_key,
                diag_levels,
                feature_cls,
                num_epochs,
                batch_size,
                trainer_cls,
                lr,
            ) = items
            feature_classes.add(feature_cls)
        feature_classes = sorted(feature_classes, key=lambda x: x.__name__)
        diagnosis_map = self.__task_generator.get_diagnosis_map(task_key="USVAC_2025")
        pbar = tqdm(
            desc="Caching tasks",
            total=len(feature_classes) * len(self.__cross_test_tasks),
        )
        diag_levels = [0]
        for feature_cls in feature_classes:
            feature_function: Feature = feature_cls(
                device=self.__device, sampling_rate=self.__sampling_rate
            )
            for ctk in self.__cross_test_tasks:
                pbar.write(f"fc: {feature_cls.__name__}, ctk: {ctk}")
                data_loader = self.__get_cached_cross_data_loader(
                    task_key=ctk,
                    diag_levels=diag_levels,
                    feature_function=feature_function,
                    cache_path=Path(
                        f"{self.__cache_path}/.data_loader/{ctk}/{feature_cls.__name__}"
                    ),
                    diagnosis_map=diagnosis_map,
                )
                del data_loader
                pbar.update(1)

    @torch.no_grad()
    def run(self, selected_exps: dict[str, int] | None = None):
        tests = {}
        completed_keys = self.__get_completed_tests(selected_exps=selected_exps)
        total_exps = 0
        for key, items in Runner._exp.items():
            if selected_exps is not None and key not in selected_exps:
                continue
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
            checkpoint_restriction = (
                None if selected_exps is None else selected_exps[key]
            )
            model_tests[key] = self.__find_checkpoints(key, checkpoint_restriction)
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
                    data_loader = self.__get_model_data_loader(
                        task_key=d_task_key,
                        diag_levels=diag_levels,
                        feature_function=feature_function,
                        diagnosis_map=diagnosis_map,
                    )
                    cross_data_loaders = {
                        ctk: self.__get_cross_data_loader(
                            task_key=ctk,
                            diag_levels=diag_levels,
                            feature_function=feature_function,
                            # cache_path=Path(
                            #     f"{self.__cache_path}/.data_loader/{ctk}/{feature_cls.__name__}"
                            # ),
                            diagnosis_map=diagnosis_map,
                        )
                        for ctk in tqdm(
                            self.__cross_test_tasks,
                            desc="Loading cross test data loaders",
                            leave=False,
                        )
                    }
                    for model_cls, model_tests in task_diag_tests.items():
                        model = model_cache.get_model(
                            data_loader=data_loader,
                            model_cls=model_cls,
                        )
                        test_func = self.__test_func_map[model_cls]
                        for model_key, model_key_tests in model_tests.items():
                            pbar_top.set_postfix({"model": model_key})
                            model_results_path = Path(
                                f"{self.__results_path}/{model_key}"
                            )
                            model_results_path.mkdir(exist_ok=True)
                            for epoch_ckpt in model_key_tests:
                                epoch = epoch_ckpt.stem
                                model.load_checkpoint(epoch_ckpt)
                                for ctk, cross_dl in cross_data_loaders.items():
                                    ctk_results_path = Path(
                                        f"{model_results_path}/{ctk}"
                                    )
                                    ctk_results_path.mkdir(exist_ok=True)
                                    result = test_func(
                                        cross_dl=cross_dl,
                                        data_loader=data_loader,
                                        model=model,
                                        diag_levels=diag_levels,
                                    )
                                    result.to_csv(
                                        f"{ctk_results_path}/{epoch}.csv", index=False
                                    )
                            pbar_top.update(1)

            # for cdl in cross_data_loaders.values():
            #     cdl.empty_cache()

    def __get_completed_tests(self, selected_exps: dict[str, int] | None = None):
        completed_keys = []
        for key in Runner._exp:
            completed = True
            ckpts = list(sorted(Path(f"{self.__ckpt_path}/{key}").glob("*.h5")))
            for ctk in self.__cross_test_tasks:
                for ckpt in ckpts:
                    epoch = ckpt.stem
                    if selected_exps is None or (
                        key in selected_exps and int(epoch) == selected_exps[key]
                    ):
                        if not Path(
                            f"{self.__results_path}/{key}/{ctk}/{epoch}.csv"
                        ).is_file():
                            completed = False
            if completed:
                completed_keys += [key]
        return completed_keys

    @torch.no_grad()
    def __test_single(
        self,
        cross_dl: BaseDataLoader,
        data_loader: BaseDataLoader,
        model: Normalized,
        diag_levels: list[int],
    ) -> pd.DataFrame:
        assert len(diag_levels) == 1
        diag_level = diag_levels[0]
        results = []
        all_ids = []
        for batch in tqdm(cross_dl.test(), desc="Testing", leave=False):
            if len(batch) == 2:
                inputs, labels = batch
            else:
                inputs, labels, ids = batch
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

        def model_diag_map(idx):
            return data_loader.idx_to_diag_name(int(idx), diag_level)

        def test_diag_map(idx):
            return cross_dl.idx_to_diag_name(int(idx), diag_level)

        results["actual"] = results["actual"].apply(test_diag_map)
        results["predicted"] = results["predicted"].apply(model_diag_map)
        return results

    @torch.no_grad()
    def __test_multi_crit(
        self,
        cross_dl: BaseDataLoader,
        data_loader: BaseDataLoader,
        model: NormalizedMultiCrit,
        diag_levels: list[int],
    ) -> pd.DataFrame:
        all_results = []
        all_ids = []
        for batch in tqdm(cross_dl.test(), desc="Testing", leave=False):
            if len(batch) == 2:
                inputs, labels = batch
            else:
                inputs, labels, ids = batch
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
            if cname.startswith("actual"):
                prefix, suffix = cname.split("_")
                level = int(suffix)
                all_results[cname] = all_results[cname].apply(
                    lambda idx: cross_dl.idx_to_diag_name(int(idx), level)
                )
            elif cname.startswith("predicted"):
                prefix, suffix = cname.split("_")
                level = int(suffix)
                all_results[cname] = all_results[cname].apply(
                    lambda idx: data_loader.idx_to_diag_name(int(idx), level)
                )
        return all_results

    @torch.no_grad()
    def __test_multi_task(
        self,
        cross_dl: BaseDataLoader,
        data_loader: BaseDataLoader,
        model: NormalizedMultitask,
        diag_levels: list[int],
    ) -> pd.DataFrame:
        all_results = []
        all_ids = []
        for batch in tqdm(cross_dl.test(), desc="Testing", leave=False):
            if len(batch) == 2:
                inputs, labels = batch
            else:
                inputs, labels, ids = batch
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
            if cname.startswith("actual"):
                prefix, suffix = cname.split("_")
                level = int(suffix)
                all_results[cname] = all_results[cname].apply(
                    lambda idx: cross_dl.idx_to_diag_name(int(idx), level)
                )
            elif cname.startswith("predicted"):
                prefix, suffix = cname.split("_")
                level = int(suffix)
                all_results[cname] = all_results[cname].apply(
                    lambda idx: data_loader.idx_to_diag_name(int(idx), level)
                )
        return all_results

    def __get_cross_data_loader(
        self,
        task_key: str,
        diag_levels: list[int],
        feature_function: Feature,
        diagnosis_map: DiagnosisMap,
    ) -> DataLoader:
        task = self.__task_generator.load_task(
            task=task_key,
            diag_level=max(diag_levels),
            diagnosis_map=diagnosis_map,
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
            allow_inter_level_comparison=True,
        )

    def __get_cached_cross_data_loader(
        self,
        task_key: str,
        diag_levels: list[int],
        feature_function: Feature,
        cache_path: Path,
        diagnosis_map: DiagnosisMap,
    ) -> CachedDataLoader:
        task = self.__task_generator.load_task(
            task=task_key,
            diag_level=max(diag_levels),
            diagnosis_map=diagnosis_map,
            # This is a heurestic of checking for existence of cache
            # it's posisble cache was incomplete, in which case the folder will
            # need to be deleted for audios to load again
            load_audios=not cache_path.is_dir(),
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
            allow_inter_level_comparison=True,
        )

    def __get_model_data_loader(
        self,
        task_key: str,
        diag_levels: list[int],
        feature_function: Feature,
        diagnosis_map,
    ) -> BaseDataLoader:
        task = self.__task_generator.load_task(
            task=task_key,
            diag_level=max(diag_levels),
            diagnosis_map=diagnosis_map,
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
        )

    def __find_checkpoints(
        self, exp_key: str, checkpoint_restriction: int | None
    ) -> list[Path]:
        sel = (
            "*.h5" if checkpoint_restriction is None else f"{checkpoint_restriction}.h5"
        )
        checkpoints = sorted(list(Path(f"{self.__ckpt_path}/{exp_key}").glob(sel)))
        if len(checkpoints) < 1:
            raise ValueError(f"No checkpoints found for {exp_key}")
        return checkpoints

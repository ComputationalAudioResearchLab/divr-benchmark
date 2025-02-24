import torch
from pathlib import Path

from ...data_loader import BaseDataLoader
from ...model import (
    Normalized,
    NormalizedMultiCrit,
    NormalizedMultitask,
    SimpleTransformer,
)


class ModelCache:

    def __init__(self, device: torch.device) -> None:
        self.__cache: dict[
            str,
            Normalized | NormalizedMultiCrit | NormalizedMultitask | SimpleTransformer,
        ] = {}
        self.__tmp_path = Path("/tmp")
        self.__device = device

    def get_model(
        self, data_loader: BaseDataLoader, model_cls
    ) -> Normalized | NormalizedMultiCrit | NormalizedMultitask | SimpleTransformer:
        input_size = data_loader.feature_size
        num_classes = data_loader.num_classes
        if model_cls == NormalizedMultitask:
            all_num_classes = ",".join(
                map(str, data_loader.num_unique_diagnosis.values())
            )
            model_cache_key = f"{model_cls.__name__}.{input_size}.{all_num_classes}"
        else:
            model_cache_key = f"{model_cls.__name__}.{input_size}.{num_classes}"
        if model_cache_key in self.__cache:
            model = self.__cache[model_cache_key]
        else:
            if model_cls == Normalized:
                model = model_cls(
                    input_size=input_size,
                    num_classes=data_loader.num_classes,
                    checkpoint_path=self.__tmp_path,
                )
            elif model_cls == NormalizedMultiCrit:
                model = model_cls(
                    input_size=input_size,
                    num_classes=data_loader.num_classes,
                    checkpoint_path=self.__tmp_path,
                    levels_map=data_loader.levels_map,
                )
            elif model_cls == NormalizedMultitask:
                model = model_cls(
                    input_size=input_size,
                    num_classes=data_loader.num_unique_diagnosis,
                    checkpoint_path=self.__tmp_path,
                )
            elif model_cls == SimpleTransformer:
                max_diag_level = max(data_loader.num_unique_diagnosis.keys())
                num_classes = data_loader.num_unique_diagnosis[max_diag_level]
                model = model_cls(
                    input_size=input_size,
                    num_classes=num_classes,
                    checkpoint_path=self.__tmp_path,
                    num_speakers=data_loader.total_speakers,
                )
            self.__cache[model_cache_key] = model.cpu()

        if isinstance(model, NormalizedMultiCrit):
            # otherwise this can be different in the cache
            model.set_levels_map(levels_map=data_loader.levels_map)
        model.to(self.__device)
        model.eval()
        return model

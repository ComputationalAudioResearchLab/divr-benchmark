import json
import torch
import random
import numpy as np
from typing import Dict
from src.trainers import Base as Trainer
from src.models import Base as Model
from src.experiment.features import features, FeatureMap
from src.experiment.data import Data
from src.logger import Logger


class Experiment:
    def __init__(self, config: Dict) -> None:
        key = config["key"]
        logger = Logger(log_path=config["log_path"], key=key)
        logger.info(f"key: {key}")
        logger.info(f"config: {json.dumps(config)}\n\n")
        self.seed(config["random_seed"])
        model = self.load_model(config["model"], logger=logger, key=key)
        self.trainer = self.load_trainer(
            config=config["trainer"],
            model=model,
            data=Data(**config["data"]),
            logger=logger,
            key=key,
        )

    def seed(self, random_seed):
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

    def load_model(self, config, key: str, logger: Logger) -> Model:
        cls = config["type"]
        module = __import__("src.models", fromlist=[cls])
        cls = getattr(module, cls)
        return cls(**config, key=key, logger=logger)

    def load_trainer(
        self, config, key: str, model: Model, data: Data, logger: Logger
    ) -> Trainer:
        cls = config["type"]
        module = __import__("src.trainers", fromlist=[cls])
        cls = getattr(module, cls)
        return cls(**config, key=key, model=model, data=data, logger=logger)

    def load_features(self, feature_params) -> FeatureMap:
        data = {}
        common_params = feature_params["common"]
        data["sampling_frequency"] = common_params["sampling_frequency"]
        for key, val in features.items():
            if key in feature_params:
                key_params = feature_params[key]
                params = {
                    **common_params,
                    **(key_params if key_params is not None else {}),
                }
                data[key] = val(**params)
        return data

    def run(self) -> None:
        self.trainer.run()

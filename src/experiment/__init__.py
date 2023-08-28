from pathlib import Path
from yaml import load, FullLoader
from src.trainers import Base as Trainer
from src.models import Base as Model
from src.experiment.features import features, FeatureMap
from src.experiment.data import Data


class Experiment:
    def __init__(self, experiment_yaml: Path) -> None:
        with open(experiment_yaml, "r") as yamlfile:
            data = load(yamlfile, Loader=FullLoader)
            self.key = data["key"]
            self.random_seed = data["random_seed"]
            model = self.load_model(data["model"])
            self.trainer = self.load_trainer(
                config=data["trainer"],
                model=model,
                data=Data(**data["data"]),
            )

    def load_model(self, config) -> Model:
        cls = config["type"]
        module = __import__("src.models", fromlist=[cls])
        cls = getattr(module, cls)
        return cls(**config)

    def load_trainer(self, config, model: Model, data: Data) -> Trainer:
        cls = config["type"]
        module = __import__("src.trainers", fromlist=[cls])
        cls = getattr(module, cls)
        return cls(**config, model=model, data=data)

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

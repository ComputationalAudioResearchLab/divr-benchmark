from pathlib import Path
from yaml import load, FullLoader
from .config import ConfigFactory
from .data import Data


class Experiment:
    def __init__(self, experiment_yaml: Path) -> None:
        with open(experiment_yaml, "r") as yamlfile:
            data = load(yamlfile, Loader=FullLoader)
            config_factory = ConfigFactory()
            self.config = config_factory.load_experiment_config(data)

    def run(self) -> None:
        Data(self.config)

from pathlib import Path
from src.models import Base as Model
from src.experiment.data import Data
from src.logger import Logger


class Base:
    def __init__(
        self,
        epochs: int,
        tensorboard_path: str,
        results_path: str,
        model: Model,
        data: Data,
        key: str,
        logger: Logger,
        **kwargs,
    ) -> None:
        self.epochs = epochs
        self.tensorboard_path = Path(f"{tensorboard_path}/{key}")
        self.results_path = Path(f"{results_path}/{key}.log")
        self.tensorboard_path.mkdir(parents=True, exist_ok=True)
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.data = data
        self.logger = logger
        self.key = key

    def run(self):
        raise NotImplementedError()


class PyTorchBase(Base):
    pass

from pathlib import Path
from src.models import Base as Model
from src.experiment.data import Data


class Base:
    def __init__(
        self,
        epochs: int,
        tensorboard_path: Path,
        model: Model,
        data: Data,
        **kwargs,
    ) -> None:
        self.epochs = epochs
        self.tensorboard_path = tensorboard_path
        self.model = model
        self.data = data

    def run(self):
        raise NotImplementedError()


class PyTorchBase(Base):
    pass

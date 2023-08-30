import torch
from torch import nn
from pathlib import Path
from src.logger import Logger


class Base:
    def __init__(
        self, checkpoint_path: str, key: str, logger: Logger, **kwargs
    ) -> None:
        super().__init__()
        self.__checkpoint_path = Path(f"{checkpoint_path}/{key}/")
        self.__checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.key = key

    def save(self, epoch: int) -> None:
        raise NotImplementedError()

    def load(self, epoch: int = 0) -> None:
        raise NotImplementedError()

    def weight_file_name(self, epoch):
        return f"{self.__checkpoint_path}/{self.__class__.__name__}_{epoch}.h5"


class PytorchBase(Base, nn.Module):
    def save(self, epoch: int) -> None:
        torch.save(self.state_dict(), self.weight_file_name(epoch))

    def load(self, epoch: int = 0) -> None:
        self.load_state_dict(
            torch.load(self.weight_file_name(epoch), map_location=self.device)
        )

    def freeze(self, layer):
        for param in layer.parameters():
            param.requires_grad = False

    def unfreeze(self, layer):
        for param in layer.parameters():
            param.requires_grad = True

    def to(self, device):
        self.device = device
        return super().to(device)

    def init_orthogonal_weights(self):
        def init(m):
            if type(m) == nn.Linear:
                nn.init.orthogonal_(m.weight)
                m.bias.data.fill_(0.01)

        self.apply(init)

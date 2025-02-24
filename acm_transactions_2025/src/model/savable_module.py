from __future__ import annotations
import torch
from torch import nn
from pathlib import Path


class SavableModule(nn.Module):
    def __init__(self, checkpoint_path: Path):
        super().__init__()
        self.__current_checkpoint_path = None
        self.checkpoint_path = checkpoint_path
        self.checkpoint_path.mkdir(exist_ok=True, parents=True)

    def save(self, epoch):
        torch.save(self.state_dict(), self.__weight_file_name(epoch))

    def load(self, epoch=0):
        ckpt = self.__weight_file_name(epoch)
        try:
            self.load_state_dict(torch.load(ckpt, map_location=self.device))
        except Exception as err:
            print(f"Failed to load checkpoint: {ckpt}")
            raise err

    def load_checkpoint(self, checkpoint: Path):
        try:
            self.load_state_dict(torch.load(checkpoint, map_location=self.device))
        except Exception as err:
            print(
                f"Unable to load checkpoint {checkpoint}. Last loaded checkpoint: {self.__current_checkpoint_path}"
            )
            raise err
        self.__current_checkpoint_path = checkpoint

    def __weight_file_name(self, epoch):
        return f"{self.checkpoint_path}/{epoch}.h5"

    def init_orthogonal_weights(self):
        def init(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                m.bias.data.fill_(0.01)

        self.apply(init)

    def to(self, device: torch.device) -> SavableModule:
        super().to(device)
        self.device = device
        return self

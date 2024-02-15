import json
import torch
from typing import Dict
from pathlib import Path


def retry_me(max_retries):
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_err = None
            for _ in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as err:
                    last_err = err

            if last_err is not None:
                print(f"Retried {max_retries} times, but consitent failure")
                raise last_err

            raise RuntimeError("exhausted retries and did not get an err :/")

        return wrapper

    return decorator


class RDS:
    """
    Class that deals with I/O issues in USyd's RDS
    save and loads data or retries until it can
    """

    @retry_me(max_retries=64)
    def load_tensor(self, file_path: Path, device: torch.device) -> torch.Tensor:
        return torch.load(f=file_path, map_location=device)

    @retry_me(max_retries=64)
    def load_json(self, file_path: Path) -> Dict:
        with open(file=file_path, mode="r") as file:
            return json.load(fp=file)

    @retry_me(max_retries=64)
    def save_tensor(self, tensor: torch.Tensor, file_path: Path):
        torch.save(tensor, file_path)

    @retry_me(max_retries=64)
    def save_json(self, object: Dict, file_path: Path):
        with open(file=file_path, mode="w") as file:
            json.dump(obj=object, fp=file)

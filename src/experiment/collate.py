import numpy as np
from typing import Dict, Callable


def one(input: Dict[str, np.ndarray]) -> np.ndarray:
    return input["mean_mfcc"]


def two(input: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            input["jitter"],
            input["shimmer"],
            input["mean_mfcc"],
        ]
    )


CollateFunc = Callable[[Dict[str, np.ndarray]], np.ndarray]

collate_funcs = {
    "one": one,
    "two": two,
}

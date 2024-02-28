import os
from typing import Literal
from class_argparse import ClassArgParser

Exp = Literal[
    "Data2Vec", "Mfcc", "MfccWithDeltas", "ModifiedCPC", "UnispeechSAT", "Wav2Vec"
]


class Shell(ClassArgParser):

    def __init__(self) -> None:
        super().__init__(name="trainer")

    def train(
        self,
        stream: Literal["0", "1", "2", "3"],
        exp: Exp,
    ):
        tasks = {
            "1": [2],
        }
        for task in tasks[stream]:
            key = f"S{stream}/T{task}/{exp}/Simple4"
            cmd = f"docker compose run --rm experiment {key}"
            os.system(cmd)


if __name__ == "__main__":
    Shell()()

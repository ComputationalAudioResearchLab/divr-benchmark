import os
from typing import Literal
from class_argparse import ClassArgParser


# exps = ["Data2Vec", "Mfcc", "MfccWithDeltas", "ModifiedCPC", "UnispeechSAT", "Wav2Vec"]
exps = ["UnispeechSAT", "Wav2Vec"]


class Shell(ClassArgParser):

    def __init__(self) -> None:
        super().__init__(name="trainer")

    def test(
        self,
        stream: Literal["0", "1", "2", "3"],
    ):
        tasks = {
            "0": [1, 2, 3, 4],
            "1": [6],
            "2": [1, 2, 3, 4],
            "3": [1, 2, 3, 4, 5],
        }
        for exp in exps:
            if stream == "0":
                if exp in ["Data2Vec", "UnispeechSAT"]:
                    model = "Simple"
                else:
                    model = "Simple2"
            else:
                model = "Simple4"
            for task in tasks[stream]:
                key = f"S{stream}/T{task}/{exp}/{model}"
                cmd = f"docker compose run --rm test {key}"
                os.system(cmd)


if __name__ == "__main__":
    Shell()()

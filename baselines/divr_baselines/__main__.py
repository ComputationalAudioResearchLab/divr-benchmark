from class_argparse import ClassArgParser
from .experiments import experiment, test


class Main(ClassArgParser):
    def __init__(self) -> None:
        super().__init__(name="DiVR Baselines")

    def experiment(self, experiment_key: str) -> None:
        experiment(experiment_key=experiment_key)

    def test(self, experiment_key: str) -> None:
        test(experiment_key=experiment_key)


if __name__ == "__main__":
    Main()()

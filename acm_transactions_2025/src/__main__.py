from class_argparse import ClassArgParser

from . import env
from .tasks_generator import TaskGenerator
from .experiments import Runner


class Main(ClassArgParser):

    def __init__(self) -> None:
        super().__init__(name="ACM Transaction 2025")

    def generate_tasks(self):
        print("Generating tasks")
        TaskGenerator(research_data_path=env.RESEARCH_DATA_PATH).generate()
        print("Tasks generated")

    def train(self, exp_key: Runner.EXP_KEYS, tboard_enabled: bool = True):
        tasks_generator = TaskGenerator(
            research_data_path=env.RESEARCH_DATA_PATH,
        )
        runner = Runner(
            tasks_generator=tasks_generator,
            cache_path=env.CACHE_PATH,
            results_path=env.RESULTS_PATH,
        )
        runner.train(exp_key=exp_key, tboard_enabled=tboard_enabled)

    def eval(self, exp_key: Runner.EXP_KEYS, load_epoch: int):
        tasks_generator = TaskGenerator(
            research_data_path=env.RESEARCH_DATA_PATH,
        )
        runner = Runner(
            tasks_generator=tasks_generator,
            cache_path=env.CACHE_PATH,
            results_path=env.RESULTS_PATH,
        )
        runner.eval(exp_key=exp_key, load_epoch=load_epoch)


if __name__ == "__main__":
    Main()()

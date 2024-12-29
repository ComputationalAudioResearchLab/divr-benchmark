from class_argparse import ClassArgParser
from . import env
from .tasks_generator import TaskGenerator


class Main(ClassArgParser):

    def __init__(self) -> None:
        super().__init__(name="ACM Transaction 2025")

    def prepare_tasks(self):
        print("Preparing tasks")
        TaskGenerator(research_data_path=env.RESEARCH_DATA_PATH).generate()


if __name__ == "__main__":
    Main()()

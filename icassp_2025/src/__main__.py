from pathlib import Path
from typing import List, get_args
from class_argparse import ClassArgParser
from .data_loader.tasks import Tasks
from .experiments import Runner

class Main(ClassArgParser):
    

    def __init__(self) -> None:
        super().__init__(name="ICASSP 2025 work")
        data_path = Path("/home/data")
        proj_path = Path(__file__).resolve().parent.parent
        tasks_path = Path(f"{proj_path}/tasks")
        self.__tasks = Tasks(data_path=data_path, tasks_path=tasks_path)
        cache_path = Path(f"{proj_path}/.cache")
        self.__experiment_runner = Runner(tasks=self.__tasks, data_path=data_path, cache_path=cache_path)

    def prepare_tasks(self):
        for key in get_args(Tasks.keys):
            for level in range(4):
                self.__tasks.prepare_task(task_key=key, diagnosis_level=level)
    
    def experiment(self, exp_key: Runner.EXP_KEYS):
        self.__experiment_runner.run(exp_key=exp_key)
        
    def experiment_short(self, exp_key: Runner.EXP_SHORT_KEYS):
        self.__experiment_runner.run_short(exp_key=exp_key)

if __name__ == "__main__":
    Main()()

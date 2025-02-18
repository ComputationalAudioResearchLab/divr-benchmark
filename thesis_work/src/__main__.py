import pandas as pd
from pathlib import Path
from class_argparse import ClassArgParser

from . import env
from .tasks_generator import TaskGenerator
from .experiments import Runner
from .experiments.testers.all_self import TestAllSelf
from .experiments.testers.all_cross import TestAllCross
from .analyser import Analyser


class Main(ClassArgParser):

    def __init__(self) -> None:
        super().__init__(name="Interspeech 2025")

    def train(
        self,
        exp_key: Runner.EXP_KEYS,  # type: ignore
        tboard_enabled: bool = True,
        use_cache_loader: bool = False,
        limit_vram: float = None,
    ):
        tasks_generator = TaskGenerator(
            research_data_path=env.RESEARCH_DATA_PATH,
            tasks_path=env.TASKS_PATH,
        )
        runner = Runner(
            tasks_generator=tasks_generator,
            cache_path=env.CACHE_PATH,
            results_path=env.RESULTS_PATH,
            research_data_path=env.RESEARCH_DATA_PATH,
        )
        runner.train(
            exp_key=exp_key,
            tboard_enabled=tboard_enabled,
            use_cache_loader=use_cache_loader,
            limit_vram=limit_vram,
        )

    def test(self, exp_key: Runner.EXP_KEYS, load_epoch: int):  # type: ignore
        tasks_generator = TaskGenerator(
            research_data_path=env.RESEARCH_DATA_PATH,
            tasks_path=env.TASKS_PATH,
        )
        runner = Runner(
            tasks_generator=tasks_generator,
            cache_path=env.CACHE_PATH,
            results_path=Path(f"{env.RESULTS_PATH}/test"),
            research_data_path=env.RESEARCH_DATA_PATH,
        )
        runner.test(exp_key=exp_key, load_epoch=load_epoch)

    async def test_all_self(self):
        tester = TestAllSelf(
            research_data_path=env.RESEARCH_DATA_PATH,
            cache_path=env.CACHE_PATH,
            results_path=env.RESULTS_PATH,
            tasks_path=env.TASKS_PATH,
        )
        tester.run()

    async def test_best_cross(self):
        tester = TestAllCross(
            research_data_path=env.RESEARCH_DATA_PATH,
            cache_path=env.CACHE_PATH,
            results_path=env.RESULTS_PATH,
            tasks_path=env.TASKS_PATH,
        )
        df = pd.read_csv(f"{env.RESULTS_PATH}/best_epoch_self.csv")
        selected_exps = df.set_index(keys=["exp_key"])["epoch"].to_dict()
        tester.run(selected_exps=selected_exps)

    def collate_self_results(self):
        Analyser(results_path=env.RESULTS_PATH).collate_self()

    def analyse_self_results(self):
        Analyser(results_path=env.RESULTS_PATH).analyse_self()

    def collate_cross_results(self):
        Analyser(results_path=env.RESULTS_PATH).collate_cross()

    def plot_results_self(self):
        Analyser(results_path=env.RESULTS_PATH).plot_results_self()

    def plot_results_cross(self):
        Analyser(results_path=env.RESULTS_PATH).plot_results_cross()


if __name__ == "__main__":
    Main()()

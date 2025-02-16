import pandas as pd
from pathlib import Path
from class_argparse import ClassArgParser

from . import env
from .tasks_generator import TaskGenerator
from .experiments import Runner
from .experiments.testers.all_cross import TestAllCross
from .experiments.testers.all_self import TestAllSelf
from .analyser import Analyser
from .reporter import Reporter


class Main(ClassArgParser):

    def __init__(self) -> None:
        super().__init__(name="ACM Transaction 2025")

    async def generate_tasks(self):
        print("Generating tasks")
        await TaskGenerator(
            research_data_path=env.RESEARCH_DATA_PATH,
            tasks_path=env.TASKS_PATH,
        ).generate()
        print("Tasks generated")

    def train(
        self,
        exp_key: Runner.EXP_KEYS,  # type: ignore
        tboard_enabled: bool = True,
        use_cache_loader: bool = True,
    ):
        tasks_generator = TaskGenerator(
            research_data_path=env.RESEARCH_DATA_PATH,
            tasks_path=env.TASKS_PATH,
        )
        runner = Runner(
            tasks_generator=tasks_generator,
            cache_path=env.CACHE_PATH,
            results_path=env.RESULTS_PATH,
        )
        runner.train(
            exp_key=exp_key,
            tboard_enabled=tboard_enabled,
            use_cache_loader=use_cache_loader,
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
        )
        runner.test(exp_key=exp_key, load_epoch=load_epoch)

    async def test_all_cross(self):
        tester = TestAllCross(
            research_data_path=env.RESEARCH_DATA_PATH,
            cache_path=env.CACHE_PATH,
            results_path=env.RESULTS_PATH,
            tasks_path=env.TASKS_PATH,
        )
        tester.run()

    async def test_cross_best_self(self):
        tester = TestAllCross(
            research_data_path=env.RESEARCH_DATA_PATH,
            cache_path=env.CACHE_PATH,
            results_path=env.RESULTS_PATH,
            tasks_path=env.TASKS_PATH,
        )
        df = pd.read_csv(f"{env.RESULTS_PATH}/best_single_task_results.csv")
        selected_exps = df.set_index(keys=["exp_key"])["epoch"].to_dict()
        tester.run(selected_exps=selected_exps)

    async def test_cross_selected(self, selection_file: Path):
        tester = TestAllCross(
            research_data_path=env.RESEARCH_DATA_PATH,
            cache_path=env.CACHE_PATH,
            results_path=env.RESULTS_PATH,
            tasks_path=env.TASKS_PATH,
        )
        df = pd.read_csv(selection_file)
        selected_exps = df.set_index(keys=["exp_key"])["epoch"].to_dict()
        tester.run(selected_exps=selected_exps)

    async def test_cross_for_interspeech(self):
        tester = TestAllCross(
            research_data_path=env.RESEARCH_DATA_PATH,
            cache_path=env.CACHE_PATH,
            results_path=env.RESULTS_PATH,
            tasks_path=env.TASKS_PATH,
        )
        df = pd.read_csv(f"{env.RESULTS_PATH}/best_epoch_per_feature.csv")
        selected_exps = (
            df[
                df["task_key"].isin(["a_n", "phrase"])
                & df["max_diag_level"].isin([0, 1, 4])
                & df["feature"].isin(["Wav2Vec", "UnispeechSAT", "MFCCDD"])
            ]
            .set_index(keys=["exp_key"])["epoch"]
            .to_dict()
        )
        tester.run(selected_exps=selected_exps)

    async def cache_all_cross_tests(self):
        tester = TestAllCross(
            research_data_path=env.RESEARCH_DATA_PATH,
            cache_path=env.CACHE_PATH,
            results_path=env.RESULTS_PATH,
            tasks_path=env.TASKS_PATH,
        )
        tester.cache_tasks()

    async def test_all_self(self):
        tester = TestAllSelf(
            research_data_path=env.RESEARCH_DATA_PATH,
            cache_path=env.CACHE_PATH,
            results_path=env.RESULTS_PATH,
            tasks_path=env.TASKS_PATH,
        )
        tester.run()

    def collate_self_results(self):
        Analyser(results_path=env.RESULTS_PATH).collate_self()

    def analyse_self_results(self):
        Analyser(results_path=env.RESULTS_PATH).analyse_self()

    def collate_cross_results(self):
        Analyser(results_path=env.RESULTS_PATH).collate_cross()

    def autogen_hierarchies(self):
        Analyser(results_path=env.RESULTS_PATH).autogen_hierarchies()

    def reporter(self, key: Reporter.reports()):  # type: ignore
        task_generator = TaskGenerator(
            research_data_path=env.RESEARCH_DATA_PATH,
            tasks_path=env.TASKS_PATH,
        )
        reporter = Reporter(
            results_path=env.RESULTS_PATH,
            task_generator=task_generator,
        )
        reporter.report(method_name=key)


if __name__ == "__main__":
    Main()()

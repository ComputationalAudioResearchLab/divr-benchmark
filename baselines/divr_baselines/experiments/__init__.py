from ..trainer import Trainer, Tester
from .s0 import s0_experiments
from .s1 import s1_experiments
from .s2 import s2_experiments
from .s3 import s3_experiments


experiments = s0_experiments + s1_experiments + s2_experiments + s3_experiments


def experiment(experiment_key: str) -> None:
    hparams = next(filter(lambda x: x.results_key == experiment_key, experiments))
    Trainer(hparams=hparams).run()


def test(experiment_key: str) -> None:
    hparams = next(filter(lambda x: x.results_key == experiment_key, experiments))
    Tester(hparams=hparams).run()

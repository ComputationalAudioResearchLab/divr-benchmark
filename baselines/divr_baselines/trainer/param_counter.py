from pathlib import Path
from .hparams import HParams


class ParamCounter:
    def __init__(self, hparams: HParams) -> None:
        checkpoint_path = Path(
            f"{hparams.base_path}/checkpoints/{hparams.checkpoint_key}"
        )
        results_path = Path(f"{hparams.base_path}/results/{hparams.results_key}")
        results_path.mkdir(parents=True, exist_ok=True)
        self.results_path = results_path
        self.data_loader = hparams.DataLoaderClass(
            benchmark_path=hparams.benchmark_path,
            benchmark_version=hparams.benchmark_version,
            stream=hparams.stream,
            task=hparams.task,
            device=hparams.device,
            batch_size=hparams.batch_size,
            random_seed=hparams.random_seed,
            shuffle_train=hparams.shuffle_train,
            loader_type=hparams.loader_type,
            cache_base_path=hparams.cache_base_path,
            cache_key=hparams.cache_key,
        )
        self.task = hparams.task
        model = hparams.ModelClass(
            input_size=self.data_loader.feature_size,
            num_classes=self.data_loader.num_unique_diagnosis,
            checkpoint_path=checkpoint_path,
        ).to(hparams.device)
        total_params = sum(p.numel() for p in model.parameters())
        with open(f"{self.results_path}/params.log") as param_file:
            param_file.write(str(total_params))

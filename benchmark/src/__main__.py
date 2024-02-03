import json
import asyncio
from pathlib import Path
from typing import List, TYPE_CHECKING
from argparse import ArgumentParser, _SubParsersAction
from tqdm import tqdm
from .download import Download
from .preprocess import Preprocess
from .diagnosis import DiagnosisMap
from .experiment import Experiment
from .experiment.MultivalueYaml import MultivalueYaml

SubparserType = _SubParsersAction[ArgumentParser] if TYPE_CHECKING else None


class Main:
    def __init__(self, database_path, lib_path, audio_extraction_path) -> None:
        self.lib_path = Path(lib_path).resolve()
        self.database_path = Path(database_path).resolve()
        self.audio_extraction_path = Path(audio_extraction_path).resolve()

    async def download_openaccess(
        self, all: bool, datasets: List[str], **kwargs
    ) -> None:
        downloader = Download(self.lib_path, self.database_path)
        if all:
            await downloader.all()
        elif datasets is not None:
            await downloader.selected(datasets)
        else:
            print("Must specify either --all or --datasets")

    def add_download_openaccess_parser(self, subparsers: SubparserType) -> None:
        preprocess_parser = subparsers.add_parser(self.download_openaccess.__name__)
        preprocess_parser.add_argument(
            "--all",
            action="store_true",
            help="Downloads all the open access databases",
        )
        preprocess_parser.add_argument(
            "--datasets",
            type=str,
            nargs="*",
            help="Downloads only specified databases",
        )

    async def preprocess(
        self, preprocessed_data_path: Path, all: bool, datasets: List[str], **kwargs
    ) -> None:
        preprocesser = Preprocess(
            self.lib_path, self.database_path, self.audio_extraction_path
        )
        if all:
            await preprocesser.all(preprocessed_data_path)
        elif datasets is not None:
            await preprocesser.selected(preprocessed_data_path, datasets)
        else:
            print("Must specify either --all or --datasets")

    def add_preprocess_parser(self, subparsers: SubparserType) -> None:
        preprocess_parser = subparsers.add_parser(self.preprocess.__name__)
        preprocess_parser.add_argument(
            "preprocessed_data_path",
            type=Path,
            help="Path where to store all the data",
        )
        preprocess_parser.add_argument(
            "--all",
            action="store_true",
            help="Figures out all the databases based on directory names",
        )
        preprocess_parser.add_argument(
            "--datasets",
            type=str,
            nargs="*",
            help="Explicitly specify which database folders to preprocess",
        )

    async def setup_diagnosis_map(
        self, input_tsv: Path, output_json: Path, **kwargs
    ) -> None:
        diagnosis_map = DiagnosisMap()
        output_json.parent.mkdir(exist_ok=True, parents=True)
        with open(output_json, "w") as output_json_file:
            json.dump(
                diagnosis_map.from_tsv(input_tsv),
                output_json_file,
                indent=2,
                ensure_ascii=False,
            )

    def add_setup_diagnosis_map_parser(self, subparsers: SubparserType) -> None:
        preprocess_parser = subparsers.add_parser(self.setup_diagnosis_map.__name__)
        preprocess_parser.add_argument(
            "input_tsv",
            type=Path,
            help="The tsv file to read from",
        )
        preprocess_parser.add_argument(
            "output_json",
            type=Path,
            help="The json file to write the output to",
        )

    async def experiment(self, experiment_yaml: Path, **kwargs) -> None:
        yaml_matrix = MultivalueYaml()
        configs = yaml_matrix.parse(experiment_yaml)
        for config in tqdm(configs, desc="running_experiment"):
            Experiment(config).run()

    def add_experiment_parser(self, subparsers: SubparserType) -> None:
        preprocess_parser = subparsers.add_parser(self.experiment.__name__)
        preprocess_parser.add_argument(
            "experiment_yaml",
            type=Path,
            help="The experiment file to run",
        )


if __name__ == "__main__":
    main = Main(
        database_path="/home/storage/databases",
        lib_path="/home/workspace/lib",
        audio_extraction_path="/home/workspace/data/extracted",
    )
    parser = ArgumentParser("VDML Benchmark")
    subparsers = parser.add_subparsers(dest="action", required=True)
    main.add_download_openaccess_parser(subparsers)
    main.add_preprocess_parser(subparsers)
    main.add_setup_diagnosis_map_parser(subparsers)
    main.add_experiment_parser(subparsers)
    args = parser.parse_args()
    func = getattr(main, args.action)
    asyncio.run(func(**vars(args)))

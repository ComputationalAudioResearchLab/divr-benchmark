import asyncio
from pathlib import Path
from typing import List, TYPE_CHECKING
from argparse import ArgumentParser, _SubParsersAction
from .download import Download

SubparserType = _SubParsersAction[ArgumentParser] if TYPE_CHECKING else None


class Main:
    def __init__(self, database_path, lib_path) -> None:
        self.lib_path = Path(lib_path).resolve()
        self.database_path = Path(database_path).resolve()

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

    async def preprocess(self, all: bool, datasets: List[str], **kwargs) -> None:
        pass

    def add_preprocess_parser(self, subparsers: SubparserType) -> None:
        preprocess_parser = subparsers.add_parser(self.preprocess.__name__)
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


if __name__ == "__main__":
    main = Main(database_path="data", lib_path="lib")
    parser = ArgumentParser("VDML Benchmark")
    subparsers = parser.add_subparsers(dest="action", required=True)
    main.add_download_openaccess_parser(subparsers)
    main.add_preprocess_parser(subparsers)
    args = parser.parse_args()
    func = getattr(main, args.action)
    asyncio.run(func(**vars(args)))

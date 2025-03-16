import asyncio
from pathlib import Path
from asyncio.subprocess import create_subprocess_exec
from .logger import Logger
from .prepare_dataset.processed import ProcessedFile


class Download:
    def __init__(self, database_path: Path, logger: Logger) -> None:
        self.__database_path = database_path
        self.__logger = logger

    async def all(self):
        await asyncio.gather(*[self.svd(), self.torgo(), self.voiced()])

    async def selected(self, datasets):
        await asyncio.gather(*[getattr(self, db)() for db in datasets])

    async def svd(self) -> None:
        db_key = "svd"
        db_path = Path(f"{self.__database_path}/{db_key}")
        db_path.mkdir(exist_ok=True, parents=True)
        proc = await create_subprocess_exec("python", "-m", "svd_downloader", db_path)
        await proc.wait()

    async def torgo(self) -> None:
        db_key = "torgo"
        torgo_path = Path(f"{self.__database_path}/{db_key}")
        torgo_path.mkdir(exist_ok=True, parents=True)
        base_link = "http://www.cs.toronto.edu/~complingweb/data/TORGO"

        async def download_and_extract(key):
            self.__logger.info(f"Downloading {db_key}/{key}")
            proc = await create_subprocess_exec(
                "wget", f"{base_link}/{key}", cwd=torgo_path
            )
            await proc.wait()
            self.__logger.info(f"Extracting {db_key}/{key}")
            proc = await create_subprocess_exec("tar", "-xf", key, cwd=torgo_path)
            await proc.wait()

        procs = map(
            download_and_extract, ["F.tar.bz2", "FC.tar.bz2", "M.tar.bz2", "MC.tar.bz2"]
        )
        await asyncio.gather(*procs)

    async def voiced(self) -> None:
        db_key = "voiced"
        voiced_path = Path(f"{self.__database_path}/{db_key}")
        voiced_path.mkdir(exist_ok=True, parents=True)
        zip_link = "https://physionet.org/static/published-projects/voiced/voiced-database-1.0.0.zip"
        proc = await create_subprocess_exec("wget", zip_link, cwd=voiced_path)
        await proc.wait()
        proc = await create_subprocess_exec(
            "unzip", "voiced-database-1.0.0.zip", cwd=voiced_path
        )
        await proc.wait()

        dat_files = list(Path(f"{voiced_path}").rglob("*.dat"))
        processes = []
        for file in dat_files:

            processes.append(
                ProcessedFile.from_wfdb(
                    dat_path=Path(str(file).replace(".dat", "")),
                    extraction_path=file.parent.resolve(),
                )
            )
        await asyncio.gather(*processes)

from pathlib import Path
from .generator import Generator
from .databases import SVD, Torgo, Voiced


class GeneratorV1(Generator):

    def __call__(self, source_path: Path, tasks_path: Path) -> None:
        print("Generating benchmark v1 tasks")
        svd = SVD(source_path=source_path)
        torgo = Torgo(source_path=source_path)
        voiced = Voiced(source_path=source_path)
        self.__stream0(
            stream_path=Path(f"{tasks_path}/streams/0"),
            svd=svd,
            torgo=torgo,
            voiced=voiced,
        )

    def __stream0(self, stream_path: Path, svd: SVD, voiced: Voiced, torgo: Torgo):
        print(f"Generating stream 0 from at {stream_path}")
        stream_path.mkdir(exist_ok=True)
        train = (
            svd.all_train(level=0)
            + torgo.all_train(level=0)
            + voiced.all_train(level=0)
        )
        self.to_task_file(train, Path(f"{stream_path}/train"))
        val = svd.all_val(level=0) + torgo.all_val(level=0) + voiced.all_val(level=0)
        self.to_task_file(val, Path(f"{stream_path}/val"))

        # task 1
        Path(f"{stream_path}/{1}").mkdir(exist_ok=True)
        self.to_task_file(
            svd.test_set_neutral_vowels(level=0),
            Path(f"{stream_path}/{1}/test.yml"),
        )

        # task 2
        Path(f"{stream_path}/{2}").mkdir(exist_ok=True)
        self.to_task_file(
            svd.test_set_connected_speech(level=0),
            Path(f"{stream_path}/{2}/test.yml"),
        )

        # task 3
        Path(f"{stream_path}/{3}").mkdir(exist_ok=True)
        self.to_task_file(
            voiced.all_test(level=0),
            Path(f"{stream_path}/{3}/test.yml"),
        )

        # task 4
        # Path(f"{stream_path}/{4}").mkdir(exist_ok=True)
        # self.to_task_file(
        #     torgo.test_set_connected_speech(level=0),
        #     Path(f"{stream_path}/{4}/test.yml"),
        # )

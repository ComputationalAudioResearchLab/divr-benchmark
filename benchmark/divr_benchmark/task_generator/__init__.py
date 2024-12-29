import typing
from pathlib import Path
from typing import Literal
from .GeneratorV1 import GeneratorV1
from .generator import DatabaseFunc, Dataset

VERSIONS = Literal["v1"]
versions = typing.get_args(VERSIONS)
generator_map = {"v1": GeneratorV1()}


def generate_tasks(version: VERSIONS, source_path: Path) -> None:
    module_path = Path(__file__).parent.parent.resolve()
    tasks_path = Path(f"{module_path}/tasks/{version}")
    generator_map[version](source_path=source_path, tasks_path=tasks_path)

from pathlib import Path
from ..diagnosis import DiagnosisMap


class BaseProcessor:
    def __init__(self) -> None:
        self.diagnosis_map = DiagnosisMap()

    async def __call__(self, source_path: Path, dest_path: Path) -> None:
        raise NotImplementedError()

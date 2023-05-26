from .base import BaseProcessor
from pathlib import Path


class SVD(BaseProcessor):
    async def __call__(self, source_path: Path, dest_path: Path) -> None:
        print("processing svd")

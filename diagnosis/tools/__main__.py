from pathlib import Path
from typing import Literal
from class_argparse import ClassArgParser

from .analysis import (
    analysis as analyse_diagnosis,
    reclassification_candidates,
)
from .level_3_confusion import level_3_confusion
from .processor import Processor
from .validate_terms import ValidateTermsOthers, ValidateTermsUSVAC
from .logger import Logger
from .diag_maps import DIAGNOSIS_MAPS, diagnosis_maps
from .match_manual import MatchManual


class Main(ClassArgParser):
    def __init__(self) -> None:
        super().__init__(name="DiVR Benchmark")
        self.logger = Logger(log_path="/tmp/main.log", key="main")

    def analyse_diagnosis_classifications(
        self,
        source_path: Path,
        output_confusion_path: Path,
    ):
        analyse_diagnosis(
            source_path=source_path,
            output_confusion_path=output_confusion_path,
        )

    def reclassification_candidates(self, output_path: Path):
        reclassification_candidates(output_path=output_path)

    def level_3_confusion(self):
        level_3_confusion()

    def process_diagnosis_list(self):
        Processor().run()

    def validate_terms(self, map: Literal[tuple(DIAGNOSIS_MAPS.keys())]):  # type: ignore -- appease pylance
        if map == diagnosis_maps.USVAC_2025.__name__:
            validator = ValidateTermsUSVAC
        else:
            validator = ValidateTermsOthers
        validator(diagnosis_map=DIAGNOSIS_MAPS[map]()).run()

    def match_manual(self):
        MatchManual().run()


if __name__ == "__main__":
    main = Main()
    main()

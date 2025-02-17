import numpy as np
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
from .reporter import Reporter


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

    def reporter(self):
        Reporter().run()

    def dissensus(self):
        dmap = diagnosis_maps.USVAC_2025()
        all_diags = dmap.unique_diags()
        print(len(all_diags))
        votes_per_chosen_class = {}
        for diag in all_diags:
            if len(diag.votes) == 0:
                continue
            parent_link = diag.best_parent_link
            if parent_link is None:
                continue
            parent_name = parent_link.parent.name
            parent_weight = parent_link.weight
            if parent_name not in votes_per_chosen_class:
                votes_per_chosen_class[parent_name] = [parent_weight]
            else:
                votes_per_chosen_class[parent_name] += [parent_weight]
        avg_vote_per_chosen = {}
        total_classes = 0
        for key, votes in votes_per_chosen_class.items():
            mean = np.mean(votes) * 100
            std = np.std(votes) * 100
            total_classes += len(votes)
            avg_vote_per_chosen[key] = (len(votes), f"{mean:.2f}\\pm{std:.2f}")
        avg_vote_per_chosen = dict(
            sorted(avg_vote_per_chosen.items(), key=lambda x: x[1][0], reverse=True)
        )
        print(total_classes)
        print(avg_vote_per_chosen)
        for key, val in avg_vote_per_chosen.items():
            print(f"{key} & ${val[0]}$ & ${val[1]}$ \\\\")


if __name__ == "__main__":
    main = Main()
    main()

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from ..diagnosis import Diagnosis


class Result:

    def __init__(self, data: List[Tuple[Diagnosis, Diagnosis]]) -> None:
        confusion: Dict[str, Dict[str, int]] = {}
        for actual, predicted in data:
            if actual.name not in confusion:
                confusion[actual.name] = {}
            actual = confusion[actual.name]
            if predicted.name not in actual:
                actual[predicted.name] = 1
            else:
                actual[predicted.name] += 1
        self.confusion = (
            pd.DataFrame(confusion).fillna(0).sort_index(axis=0).sort_index(axis=1)
        )

    @property
    def top_1_accuracy(self) -> float:
        confusion = self.confusion.to_numpy()
        total_per_class = np.maximum(1, confusion.sum(axis=1))
        corrects = confusion.diagonal()
        per_class_accuracy = corrects / total_per_class
        accuracy = per_class_accuracy.mean()
        return accuracy

    # good metric here is an open research question

    # maybe a multi-level top-k accuracy score that penalizes
    # higher level classifications more than lower

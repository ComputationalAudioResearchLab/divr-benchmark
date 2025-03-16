import itertools
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from divr_diagnosis import Diagnosis


class Result:
    def __init__(self, data: List[Tuple[Diagnosis, Diagnosis]]) -> None:
        confusion = self.__create_confusion_matrix(data)
        for actual, predicted in data:
            confusion[predicted.name][actual.name] += 1
        self.confusion = (
            pd.DataFrame(confusion).fillna(0).sort_index(axis=0).sort_index(axis=1)
        )
        self.results = data

    @property
    def top_1_accuracy(self) -> float:
        confusion = self.confusion.to_numpy()
        total_per_class = np.maximum(1, confusion.sum(axis=1))
        corrects = confusion.diagonal()
        per_class_accuracy = corrects / total_per_class
        accuracy = per_class_accuracy.mean()
        return accuracy

    def __create_confusion_matrix(self, data):
        confusion: Dict[str, Dict[str, int]] = {}
        all_names = list(
            set(
                [actual.name for (actual, _) in data]
                + [predicted.name for (_, predicted) in data]
            )
        )
        product_name_pair = itertools.product(all_names, all_names)
        for actual, predicted in product_name_pair:
            if actual not in confusion:
                confusion[actual] = {}
            actual = confusion[actual]
            if predicted not in actual:
                actual[predicted] = 0
        return confusion

    # good metric here is an open research question

    # maybe a multi-level top-k accuracy score that penalizes
    # higher level classifications more than lower

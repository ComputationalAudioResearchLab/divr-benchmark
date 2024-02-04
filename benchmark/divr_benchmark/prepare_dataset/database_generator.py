import random
from typing import List, Tuple
from .processed import ProcessedDataset, ProcessedSession


class DatabaseGenerator:
    """
    This class does takes a best effort approach to put a proportional amount
    of diagnosis, age and gender in the three different datasets i.e. train, val and test.

    First priority goes to diagnosis, then gender and then age.

    As we don't have a lot of diagnosis at every level of diagnosis, if a diagnosis can not
    be distributed equitably at a given level it would be resolved later at a parent level
    along with other unresolved diagnosis.
    Further leaf level diagnosis that can not be distributed equitably are distributed in
    test, train and validation set in that order so that they definitely appear in test set.

    Since a diagnosis can have multiple parents, the parent that it gets grouped with for
    distribution is decided with majority vote across the parent weight. In case of a tie
    classes are chosen in the order of pathological, normal and then unclassified. In case
    ties are within a subgroup then choice is made via random selection. This randomness
    is achieved by the presorting logic of the sessions list.
    """

    def __init__(self) -> None:
        pass

    def generate(
        self,
        db: str,
        sessions: List[ProcessedSession],
        split: Tuple[float, float] = (0.7, 0.1),
        seed: int = 42,
    ):
        total_data = len(sessions)
        train_start = 0
        train_end = int(train_start + split[0] * total_data)
        val_start = train_end
        val_end = int(val_start + split[1] * total_data)
        test_start = val_end
        test_end = total_data

        random.Random(seed).shuffle(sessions)

        return ProcessedDataset(
            db=db,
            train_sessions=sessions[train_start:train_end],
            val_sessions=sessions[val_start:val_end],
            test_sessions=sessions[test_start:test_end],
        )

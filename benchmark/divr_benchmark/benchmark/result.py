from dataclasses import dataclass


@dataclass
class Result:
    correct: int
    incorrect: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total

    @property
    def total(self) -> int:
        return self.correct + self.incorrect

    # good metric here is an open research question

    # maybe a multi-level top-k accuracy score that penalizes
    # higher level classifications more than lower

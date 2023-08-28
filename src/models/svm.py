from .Base import Base
from sklearn import svm
from typing import Literal


class SVM(Base):
    def __init__(
        self,
        degree: int,
        kernel: Literal["linear", "poly", "rbf", "sigmoid", "precomputed"],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = svm.SVC(degree=degree, kernel=kernel)

    def fit(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)
        print(
            f"Trained SVM:: fit_status: {self.model.fit_status_}, n_iters: {self.model.n_iter_}"
        )

    def predict(self, X_val):
        return self.model.predict(X_val)

from .Base import Base
from sklearn import svm, linear_model
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
        self.logger.info(
            f"Trained SVM({self.key}):: fit_status: {self.model.fit_status_}, n_iters: {self.model.n_iter_}"
        )

    def predict(self, X_val):
        return self.model.predict(X_val)


class SGDSVM(Base):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = linear_model.SGDClassifier(
            loss="hinge"
        )  # using hinge loss ensures SVM

    def fit(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)
        self.logger.info(f"Trained SVM({self.key}):: n_iters: {self.model.n_iter_}")

    def predict(self, X_val):
        return self.model.predict(X_val)

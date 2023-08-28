from src.models import SVM as SVMModel
from .Base import Base
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


class SVM(Base):
    model: SVMModel

    def run(self):
        self.data.load()
        (accuracy, precision, recall, f1) = self.train()
        print(accuracy, precision, recall, f1)
        (accuracy, precision, recall, f1) = self.eval()
        print(accuracy, precision, recall, f1)

    def train(self):
        self.model.fit(self.data.train_X, self.data.train_Y)
        pred_Y = self.model.predict(self.data.train_X)
        return self.metrics(target_Y=self.data.train_Y, pred_Y=pred_Y)

    def eval(self):
        pred_Y = self.model.predict(self.data.val_X)
        return self.metrics(target_Y=self.data.val_Y, pred_Y=pred_Y)

    def metrics(self, target_Y, pred_Y):
        accuracy = accuracy_score(target_Y, pred_Y)
        precision = precision_score(
            target_Y, pred_Y, average="weighted", zero_division=0
        )
        recall = recall_score(target_Y, pred_Y, average="weighted", zero_division=0)
        f1 = f1_score(target_Y, pred_Y, average="weighted", zero_division=0)
        return (accuracy, precision, recall, f1)

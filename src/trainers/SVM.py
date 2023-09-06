import numpy as np
from src.models import SVM as SVMModel
from .Base import Base
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt


class SVM(Base):
    model: SVMModel

    def run(self):
        self.data.load()
        with open(self.results_file, "w") as results_file:
            results_file.write(f"Running for key={self.key}")
            results_file.write("accuracy, precision, recall, f1\n")
            (accuracy, precision, recall, f1, confusion_train) = self.train()
            self.save_confusion(confusion_data=confusion_train, key="train")
            results_file.write(f"{accuracy}, {precision}, {recall}, {f1}\n")
            (accuracy, precision, recall, f1, confusion_val) = self.eval()
            self.save_confusion(confusion_data=confusion_val, key="val")
            results_file.write(f"{accuracy}, {precision}, {recall}, {f1}\n")
            self.model.save(0)

    def save_confusion(self, confusion_data, key):
        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion_data[0], display_labels=confusion_data[1]
        )
        disp.plot()
        plt.savefig(f"{self.results_path}/{self.key}_{key}.png", bbox_inches="tight")
        plt.close()

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
        confusion = confusion_matrix(target_Y, pred_Y)
        all_keys = np.unique(np.concatenate((target_Y, pred_Y))).tolist()
        display_labels = [
            self.data.diagnosis_map.from_int(i).name for i in np.unique(all_keys)
        ]
        return (accuracy, precision, recall, f1, (confusion, display_labels))

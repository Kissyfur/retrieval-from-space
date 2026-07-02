import logging
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO)


def conf_matr(ty, py):
    return confusion_matrix(ty, py, normalize='true')


class Metrics:
    M_FUNCTIONS = {
        "precision": lambda y, yhat: precision_score(y, yhat, average="macro"),
        "accuracy": lambda y, yhat: accuracy_score(y, yhat),
        "recall": lambda y, yhat: recall_score(y, yhat, average="macro"),
        "f1": lambda y, yhat: f1_score(y, yhat, average="macro"),
        "CM": conf_matr,
    }
    MET_NAMES = M_FUNCTIONS.keys()

    def __init__(self, labels, met_names=MET_NAMES):
        self.labels = labels
        self.metrics = {m_name:  self.M_FUNCTIONS[m_name] for m_name in met_names}
        self.metric_val = {m_name:  None for m_name in met_names}

    def compute_metrics(self, ty, py):
        self.metric_val = {m_name: met(ty, py) for m_name, met in self.metrics.items()}

    def plot_conf_matr(self):
        disp = ConfusionMatrixDisplay(confusion_matrix=self.metric_val["CM"],
                                      display_labels=self.labels)
        disp.plot()
        plt.show()

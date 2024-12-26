import torch
import torch.nn.functional as F
from typing import Dict, Union, Optional
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score


class Metrics_classification():
    def __init__(self, num_classes: int = 2, threshold: float = 0.5, average: str ="micro", mode="one-out-one-label"):
        """
        average:
            micro:  Calculate metrics globally by counting the total true positives, false negatives and false positives.
            macro:  Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
            binary
        """
        self.num_classes = num_classes
        self.threshold = 0.5
        if num_classes == 2 or mode == "binary":
            self.average = "binary"
        else:
            self.average = average
        self.mode = mode

    def accuracy(self, y_true, y_pred) -> float:
        return accuracy_score(y_true, y_pred)

    def f1(self, y_true, y_pred, average='micro'):
        return f1_score(y_true, y_pred, average=average)

    def roc_auc(self, y_true, y_pred_proba):
        if self.mode == "binary":
            return roc_auc_score(y_true, y_pred_proba)
        return roc_auc_score(y_true, y_pred_proba, average='micro', multi_class='ovr')

    def __call__(self, logits, labels, *args, **kwds):
        logits = np.array(logits)
        labels = np.array(labels)

        if self.mode == "binary":
            predictions =  np.argmax(logits, axis=1)
            logits = logits[:,1]
        else:
            predictions = (logits > self.threshold).astype(float)

        # print(logits.shape, labels.shape, predictions.shape)
        return {
                "accuracy" : self.accuracy(labels, predictions),
                "f1_score": self.f1(labels, predictions, average=self.average),
                "auc": self.roc_auc(labels, logits)
                }
    
from pathlib import Path
from typing import Union

import numpy as np

from .base import BaseDataset


class MultiLabelDataset(BaseDataset):
    NO_FINDINGS_CLASS_ID = 14

    def _load_labels(self, label_path: Path):
        with label_path.open("r") as f:
            lines = f.readlines()
            class_ids = np.unique([int(line.split()[0]) for line in lines])
            labels = np.zeros(self.num_classes, dtype=np.float32)
            if len(class_ids) == 1 and class_ids[0] == self.NO_FINDINGS_CLASS_ID:
                return labels
            labels[class_ids] = 1.0
            return labels

    def get_weights(self) -> np.ndarray:
        labels = np.array(self.labels)
        non_zero_mask = np.any(labels != 0, axis=1)
        non_zero_labels = labels[non_zero_mask]

        n_samples, n_classes = non_zero_labels.shape
        class_counts = np.sum(non_zero_labels, axis=0)
        self.weights = n_samples / (n_classes * class_counts)
        assert len(self.weights) == self.num_classes
        return self.weights
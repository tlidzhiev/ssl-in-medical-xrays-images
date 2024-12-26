from pathlib import Path

from typing import Union
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from .base import BaseDataset


class BinaryLabelDataset(BaseDataset):
    NO_FINDINGS_CLASS_ID = 14

    def __init__(
        self,
        images_dir: Union[str, Path],
        labels_dir: Union[str, Path] = None,
        transform=None,
    ):
        super().__init__(
            num_classes=2,
            images_dir=images_dir,
            labels_dir=labels_dir,
            transform=transform,
        )

    def _load_labels(self, label_path: Path):
        with label_path.open("r") as f:
            lines = f.readlines()
            class_ids = np.unique([int(line.split()[0]) for line in lines])
            if len(class_ids) == 1 and class_ids[0] == self.NO_FINDINGS_CLASS_ID:
                return 0
            else:
                return 1

    def get_weights(self) -> np.ndarray:
        self.weights = compute_class_weight(class_weight="balanced", classes=np.unique(self.labels), y=self.labels)
        assert len(self.weights) == 2 # binary classification
        return self.weights

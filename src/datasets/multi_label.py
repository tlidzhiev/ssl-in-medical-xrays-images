from pathlib import Path
from typing import Union

import numpy as np

from .base import BaseDataset


class MultiLabelDataset(BaseDataset):
    def __init__(
        self,
        num_classes: int,
        images_dir: Union[str, Path],
        labels_dir: Union[str, Path] = None,
        transform=None,
    ):
        super().__init__(images_dir, labels_dir, transform)
        self.num_classes = num_classes

    def _load_labels(self, label_path: Path):
        with label_path.open("r") as f:
            lines = f.readlines()
            class_ids = np.unique([int(line.split()[0]) for line in lines])
            labels = np.zeros(self.num_classes, dtype=np.float32)
            labels[class_ids] = 1.0
            return labels

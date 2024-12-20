from pathlib import Path

import numpy as np

from .base import BaseDataset


class BinaryLabelDataset(BaseDataset):
    NO_FINDINGS_CLASS_ID = 14

    def _load_labels(self, label_path: Path):
        with label_path.open("r") as f:
            lines = f.readlines()
            class_ids = np.unique([int(line.split()[0]) for line in lines])
            if len(class_ids) == 1 and class_ids[0] == self.NO_FINDINGS_CLASS_ID:
                return 0
            else:
                return 1

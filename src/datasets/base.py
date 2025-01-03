from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent.parent


class BaseDataset(Dataset):
    NO_FINDINGS_CLASS_ID = 14

    def __init__(
        self,
        num_classes: int,
        images_dir: Union[str, Path],
        labels_dir: Union[str, Path] = None,
        transform=None,
    ):
        self.num_classes = num_classes
        self.images_dir = ROOT_PATH / Path(images_dir)
        self.labels_dir = ROOT_PATH / Path(labels_dir) if labels_dir else None
        self.transform = transform
        self.image_files = sorted(self.images_dir.glob("*.png"))
        self.labels_available = self.labels_dir is not None
        if self.labels_available:
            self.label_files = sorted(self.labels_dir.glob("*.txt"))
            assert len(self.image_files) == len(
                self.label_files
            ), "Mismatch between images and labels!"

        print("Downloading images and labels...")
        self.images = [self._load_image(img_path) for img_path in self.image_files]
        self.labels = (
            [self._load_labels(label_path) for label_path in self.label_files]
            if self.labels_available
            else None
        )
        self.weights = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        return image

    def _load_image(self, img_path: Path):
        return Image.open(img_path).convert("RGB")

    def _load_labels(self, label_path: Path):
        raise NotImplementedError("Subclasses must implement _load_labels method")

    def get_weights(self) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement get_weights method")

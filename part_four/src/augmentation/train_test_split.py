from pathlib import Path
import shutil
from typing import Optional
import random
from ..utils.constants import IMAGE_EXTENSIONS


def train_test_split(root: Path, train_size: float = 0.8, seed: Optional[int]=None):
    """
    This function splits the dataset into train/test sets
    """
    rng = random.Random(seed) if seed is not None else random

    train_folder = Path('./augmented_data/train')
    test_folder = Path('./augmented_data/test')

    train_folder.mkdir(parents=True, exist_ok=True)
    test_folder.mkdir(parents=True, exist_ok=True)

    for item in root.iterdir():
        if item.is_dir():
            images = list(
                img for img in item.iterdir()
                if img.is_file() and img.suffix[1:] in IMAGE_EXTENSIONS
            )
            
            train_images = rng.sample(images, int(train_size * len(images)))
            test_images = [img for img in images if img not in train_images]

            label = item.name
            label_train_path = train_folder / label
            label_test_path = test_folder / label
            
            label_train_path.mkdir(parents=True, exist_ok=True)
            label_test_path.mkdir(parents=True, exist_ok=True)
            
            for img in train_images:
                shutil.copy2(img, label_train_path / img.name)
            
            for img in test_images:
                shutil.copy2(img, label_test_path / img.name)

    return (train_folder, test_folder)
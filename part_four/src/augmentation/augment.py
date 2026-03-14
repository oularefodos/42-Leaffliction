from pathlib import Path
import sys
from typing import Optional
from .train_test_split import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[1]))

from part_two.utils.augmentation import balance_classes

def augment(root: Path, train_size=0.8, seed: Optional[int] = None):
    """
    This function does:
     
    1. splits into train/test sets
    2. augments and balances the training dataset

    It returns: [test_path, test_path]
    """
    train_path, test_path = train_test_split(root, train_size, seed)

    balance_classes(train_path, save_in_same_folder=True)

    return (train_path, test_path)
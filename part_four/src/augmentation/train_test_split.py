from pathlib import Path
from typing import Optional


def train_test_split(root: Path, train_size: float = 0.8, seed: Optional[int]=None):
    """
    This function splits the dataset into train/test sets
    """
    return root
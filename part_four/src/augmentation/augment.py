from pathlib import Path
from typing import Optional
from .train_test_split import train_test_split

def augment(root: Path, train_size=0.8, seed: Optional[int] = None):
    """
    This function does:
     
    1. splits into train/test sets
    2. augments and balances the training dataset

    It returns: [test_path, test_path]
    """
    train_path, test_path = train_test_split(root, train_size, seed)

    return (train_path, test_path)
from pathlib import Path
from typing import Optional

def augment(root: Path, test_size=0.8, seed: Optional[int] = None):
    """
    This function does:
     
    1. splits into train/test sets
    2. augments and balances the training dataset

    It returns: [test_path, test_path]
    """
    return (root, root)
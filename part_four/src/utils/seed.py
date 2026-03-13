import torch
from torch.cpu import is_available

def set_seed(seed: int):
    """
    Sets the seed for reproducibility
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


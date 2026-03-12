import torch

def set_seed(seed: int):
    """
    Sets the seed for reproducibility
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.mps.manual_seed(seed) # For Mac/MPS backend


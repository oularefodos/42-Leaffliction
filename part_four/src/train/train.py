from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from ..model.cnn import CNN

def train(
    loader: DataLoader,
    model: CNN, # change to CNN
    device: torch.device,
    epochs = 20,
):
    return model

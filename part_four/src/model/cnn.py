import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(CNN, self).__init__()
    
    def forward(self, x):
        return x
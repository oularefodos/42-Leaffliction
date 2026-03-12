from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

BATCH_SIZE = 32

def make_dataloaders(train_path: Path, test_path: Path):
    transform = transforms.Compose([
        transforms.Resize(size=(128, 128)),
        transforms.ToTensor()
    ])

    train_set = ImageFolder(train_path, transform=transform)
    test_set = ImageFolder(test_path, transform=transform)

    train_dataloader = DataLoader(
        dataset=train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_dataloader = DataLoader(
        dataset=test_set,
        batch_size=BATCH_SIZE
    )

    return (train_dataloader, test_dataloader)

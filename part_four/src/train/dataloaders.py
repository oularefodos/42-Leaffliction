from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

BATCH_SIZE=64
NUM_WORKERS=0
transform = transforms.Compose([
    transforms.Resize(size=(128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def make_dataloaders(train_path: Path, test_path: Path):

    train_set = ImageFolder(train_path, transform=transform)
    test_set = ImageFolder(test_path, transform=transform)

    train_dataloader = DataLoader(
        dataset=train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    test_dataloader = DataLoader(
        dataset=test_set,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    classes = train_set.classes

    return (train_dataloader, test_dataloader, classes)

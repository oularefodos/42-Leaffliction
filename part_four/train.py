#!.venv/bin/python
import sys
from zipfile import Path
import torch
from torch.cpu import is_available
from src.utils.seed import set_seed
from src.train.dataloaders import make_dataloaders
from src.train.train import train
from src.augmentation.augment import augment
from src.utils.parse_folder import parse_folder
from src.model.cnn import CNN
sys.path.append(str(Path(__file__).resolve().parents[1]))
from part_three.utils.transform_utils import get_image_transformations
from part_two.utils.augmentation import balance_classes


def main():
    # preparing data
    root = parse_folder(sys.argv)
    train_path, test_path = augment(root, train_size=0.8, seed=42)

    # using device agnostic execution
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    
    print(f"INFO: running on {device}")
    
    # training the model
    set_seed(42)
    train_dl, test_dl, classes = make_dataloaders(train_path, test_path)
    model = CNN(output_size=len(classes)).to(device)
    model = train(
        train_loader=train_dl,
        test_loader=test_dl,
        model=model,
        device=device,
        epochs=30
    )

    # saving the model parameters
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "classes": classes
    }
    torch.save(checkpoint, './model.pth')

if __name__ == "__main__":
    main()
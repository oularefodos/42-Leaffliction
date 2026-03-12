#!.venv/bin/python
import sys
import torch
from src.utils.seed import set_seed
from src.train.dataloaders import make_dataloaders
from src.train.train import train
from src.augmentation.augment import augment
from src.utils.parse_folder import parse_folder
from src.model.cnn import CNN


def main():
    # preparing data
    root = parse_folder(sys.argv)
    train_path, test_path = augment(root)

    # using device agnostic execution
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # training the model
    set_seed(42)
    train_dl, _ = make_dataloaders(train_path, test_path)
    model = CNN().to(device)
    model = train(
        loader=train_dl,
        model=model,
        device=device,
        epochs=20
    )

    # saving the model parameters
    torch.save(model.state_dict(), './model.pth')

if __name__ == "__main__":
    main()
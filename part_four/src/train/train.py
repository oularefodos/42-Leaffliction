from torch.utils.data import DataLoader
from torch import nn
import torch
from ..model.cnn import CNN
from tqdm import tqdm

def train(
    train_loader: DataLoader,
    test_loader: DataLoader,
    model: CNN,
    device: torch.device,
    epochs = 20,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters())

    for i in range(epochs):
        model.train()
        for X, y in tqdm(train_loader, f"Epoch #{i}"):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = criterion(y_pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        with torch.inference_mode():
            train_acc = 0
            train_size = 0
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                train_size += len(y)
                train_acc += (y == torch.argmax(y_pred, dim=1)).sum().item()

            test_acc = 0
            test_size = 0
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                test_size += len(y)
                test_acc += (y == torch.argmax(y_pred, dim=1)).sum().item()

            train_acc = round(train_acc / train_size, 2) * 100
            test_acc = round(test_acc / test_size, 2) * 100
        print(f"Epoch #{i}: Train Accuracy: {train_acc}%  | Test Accuracy: {test_acc}%", flush=True)

    return model

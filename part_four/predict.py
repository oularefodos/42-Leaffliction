#!.venv/bin/python
import torch
import sys
from src.train.dataloaders import transform
from src.model.cnn import CNN
from PIL import Image


def main():
    # prepare image
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path_to_image>", file=sys.stderr)
        sys.exit(1)

    try:
        img = Image.open(sys.argv[1]).convert('RGB')
    except Exception as e:
        print("Cannot open image", file=sys.stderr)
        sys.exit(1)

    img = transform(img).unsqueeze(dim=0)

    # load model
    checkpoint = torch.load('./model.pth')
    classes = checkpoint['classes']
    model = CNN()
    model.load_state_dict(checkpoint['model_state_dict'])

    # predict
    label = model(img)
    label = torch.argmax(label, dim=1).item()
    label = classes[label]

    print(label)

if __name__ == "__main__":
    main()
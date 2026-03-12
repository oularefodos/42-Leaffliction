#!.venv/bin/python
import torch
import sys
from src.train.dataloaders import transform
from src.model.cnn import CNN
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import plantcv.plantcv as pcv

sys.path.append(str(Path(__file__).resolve().parents[1]))
from part_three.utils.transform_utils import get_image_transformations

def main():
    # prepare image
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path_to_image>", file=sys.stderr)
        sys.exit(1)

    try:
        img_path = Path(sys.argv[1])
        img_raw = Image.open(sys.argv[1]).convert('RGB')
    except:
        print("Cannot open image", file=sys.stderr)
        sys.exit(1)

    img = transform(img_raw).unsqueeze(dim=0)

    # load model
    checkpoint = torch.load('./model.pth')
    classes = checkpoint['classes']
    model = CNN()
    model.load_state_dict(checkpoint['model_state_dict'])

    # predict
    label = model(img)
    label = torch.argmax(label, dim=1).item()
    label = classes[label]
    actual_label = img_path.parent.name


    # plot result
    transformed_img = get_image_transformations(
        pcv.readimage(img_path)[0],
        mode='plot'
    )['mask']

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2) 
    fig.suptitle(
        f"Predicted: {label}\nActual: {actual_label}",
        color="green" if actual_label==label else "red",
        fontsize=16,
        fontweight="bold",
    )
    ax1.imshow(img_raw)
    ax1.axis('off')
    ax2.imshow(transformed_img)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()
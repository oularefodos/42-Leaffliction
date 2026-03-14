from utils.histogram import get_histogram
from utils.transform_utils import get_image_transformations
import matplotlib.pyplot as plt

def plot_transformations(img):
    transformations = get_image_transformations(img, 'plot')
    images = [
        {"image": transformations['original'], "title": 'Fig 1: Original'},
        {"image": transformations['blur'], "title": 'Fig 2: Gaussian blur'},
        {"image": transformations['mask'], "title": 'Fig 3: Mask'},
        {"image": transformations['roi'], "title": 'Fig 4: Roi objects'},
        {"image": transformations['analyse'], "title": 'Fig 5:  Analyze object'},
        {"image": transformations['landmarks'], "title": 'Fig 6: Pseudolandmarks'},
    ]

    fig, axs = plt.subplots(3, 2)
    fig.suptitle("Image Transformations", fontsize=16, fontweight='bold')

    for image, ax in zip(images, axs.flatten()):
        ax.imshow(image['image'], cmap='gray')
        ax.set_title(image['title'])
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('./transformations.jpg')
    plt.close()

    hist = get_histogram(img)
    hist.savefig('./histogram.jpg')
    plt.close()
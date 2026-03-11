from utils.histogram import get_histogram
from utils.transform_utils import get_image_transformations
from pathlib import Path
from utils.output import Output
import plantcv.plantcv as pcv
import matplotlib.pyplot as plt
from tqdm import tqdm


def transform_batch(src: Path, dst: Path, filter: Output.Filter):
    extensions = ['.jpg', 'png', '.jpeg', '.JPG', '.PNG', '.JPEG']
    mode = 'save'

    image_paths = list(
        img_path for ext in extensions for img_path in src.glob(f'*{ext}')
    )

    for img_path in tqdm(image_paths, desc="Processing images"):
        img = pcv.readimage(img_path)[0]
        name = img_path.stem
        ext = img_path.suffix
        output_path = dst / f'{name}_{filter}{ext}'

        if filter == 'histogram':
            hist = get_histogram(img)
            hist.savefig(output_path)
            plt.close()
        
        else:
            transformed = get_image_transformations(img, mode)[filter]
            pcv.print_image(transformed, output_path)

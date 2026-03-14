import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt

IMAGE_EXTENSIONS = ['PNG', 'JPEG', 'JPG', 'jpg', 'png', 'jpg']

def is_image(path: Path):
    if not path.is_file():
        return False
    return path.suffix.upper().lstrip('.') in IMAGE_EXTENSIONS

def plot_distrubution(distribution_data):
    classes = list(distribution_data.keys())
    counts = list(distribution_data.values())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.get_cmap('tab10')(range(len(classes)))


    ax1.pie(counts, labels=classes, colors=colors)
    ax1.set_title('Distribution (Pie)') 

    ax2.bar(classes, counts, color=colors)
    ax2.set_title('Distribution (Bar)')
    ax2.set_xlabel('Classes')
    ax2.set_ylabel('Count')
    
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

def distribution(path: Path, distribution_data: dict):
    for item in path.iterdir():
        current_full_path = item
        parent_name = path.name

        if current_full_path.is_dir():
            distribution(current_full_path, distribution_data)
        else:
            isImg = is_image(current_full_path);

            if not isImg:
                continue;
            
            if parent_name not in distribution_data:
                distribution_data[parent_name] = 0;
            distribution_data[parent_name] += 1;

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("folderPath", help="the database folder path");

    args = parser.parse_args()

    folderPath = Path(args.folderPath)

    distribution_data = {}
    if folderPath.is_dir():
        distribution(folderPath, distribution_data);
        sorted_data_desc = dict(sorted(distribution_data.items(), key=lambda item: item[1], reverse=True))
        plot_distrubution(sorted_data_desc)
    else:
        print("No folder with this path")
    
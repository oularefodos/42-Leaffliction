import argparse
import os
import matplotlib.pyplot as plt
import imghdr

def is_image(file_path):
    if not os.path.isfile(file_path):
        return False
    try:
        return imghdr.what(file_path) is not None
    except:
        return False

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

def distribution(path, distribution_data):
    for item in os.listdir(path):
        current_full_path = os.path.join(path, item)
        parent_name = os.path.basename(path)

        if os.path.isdir(current_full_path):
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

    folderPath = args.folderPath

    distribution_data = {}
    if os.path.isdir(folderPath):
        distribution(folderPath, distribution_data);
        sorted_data_desc = dict(sorted(distribution_data.items(), key=lambda item: item[1], reverse=True))
        plot_distrubution(sorted_data_desc)
    else:
        print("No folder with this path")
    
#!./menv/bin/python

import argparse
from pathlib import Path
import sys
from utils.augmentation import augment_and_save_display_image, balance_classes
from utils.parse_folder import (
    parse_folder, 
    is_image
)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="the image path to augment");
    args = parser.parse_args()
    path = Path(args.data_path)

    if not path.exists():
        print("The provided path does not exist.", file=sys.stderr)
        sys.exit(1)

    if is_image(path):
        augment_and_save_display_image(path)
    elif parse_folder(path):
        balance_classes(path)
    else:
        print("The provided path is neither a valid image nor a valid dataset folder.", file=sys.stderr)
        sys.exit(1)
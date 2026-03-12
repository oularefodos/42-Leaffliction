import sys
from pathlib import Path
from .constants import IMAGE_EXTENSIONS


def is_valid_dataset(root: Path):
    """checks that the folder have at least
    two subfolders filled with images"""
    valid_folders = 0
    for item in root.iterdir():
        if item.is_dir():
            n_images = sum([len(list(item.glob(f"*.{ext}"))) for ext in IMAGE_EXTENSIONS])
            if n_images >= 1:
                valid_folders += 1
        if valid_folders >= 2:
            return True
    return False

def parse_folder(argv):    
    if len(argv) != 2:
        print(f"Usage: {argv[0]} <images_folder>", file=sys.stderr)
        sys.exit(1)
    
    root = Path(argv[1])

    if not root.exists():
        print("Folder does not exist", file=sys.stderr)
        sys.exit(1)
    
    if not root.is_dir():
        print(f"{root} is not a directory", file=sys.stderr)
        sys.exit(1)
    
    if not is_valid_dataset(root):
        print("Invalid dataset", file=sys.stderr)
        sys.exit(1)
    
    return root

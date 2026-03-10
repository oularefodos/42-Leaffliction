import argparse
import imghdr
import os

def is_image(file_path):
    if not os.path.isfile(file_path):
        return False
    try:
        return imghdr.what(file_path) is not None
    except:
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", help="Image path to augment");

    args = parser.parse_args()

    file_path = args.file_path
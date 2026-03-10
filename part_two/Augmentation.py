import argparse
import os
import cv2 as cv
import imghdr

def is_image(file_path):
    if not os.path.isfile(file_path):
        return False
    try:
        return imghdr.what(file_path) is not None
    except:
        return False

def augment_image(image_path):
    print(f"Augmenting image: {image_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("image_path", help="the image path to augment");

    args = parser.parse_args()

    image_path = args.image_path

    if is_image(image_path):
        augment_image(image_path)
    else:
        print(f"The provided path '{image_path}' is not a valid image.");
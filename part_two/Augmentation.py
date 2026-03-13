import argparse
import os
from pathlib import Path
import sys
import cv2 as cv
from utils.parse_folder import (get_min_image_number, parse_folder, is_image)
from utils.transformation import (
    get_projective_image,
    add_contrast,
    scale_image,
    blur_image,
    rotate_image,
    flip_image,
    display_images
)

def augment_image(image_path: Path, transfor_max=6):
    img = cv.imread(str(image_path))
    if img is None:
        print(f"Failed to read the image at {image_path}. Please check the file path and ensure it's a valid image.", file=sys.stderr)
        sys.exit(1)
    images = []
    transformations = [
        ('Flipped', flip_image(img, 1)),
        ('Rotated', rotate_image(img, 45)),
        ('Gaussian', blur_image(img, 5)),
        ('Scaled', scale_image(img, 1.3)),
        ('Increased', add_contrast(img, 1.5)),
        ('Projective', get_projective_image(img))
    ]
    if transfor_max > len(transformations):
        transfor_max = len(transformations)
    for title, transformed in transformations[:transfor_max]:
        images.append((title, transformed))
    return images

def balance_classes(root: Path):
    min_images = get_min_image_number(root)
    number_of_augmentations = 6
    max_images = min_images * number_of_augmentations

    aumented_dataset_main_path = "augmented_dataset"

    for item in root.iterdir():
        if item.is_dir():
            images = [img for img in item.iterdir() if is_image(img)]
            n_images = len(images)
            needed_augmentations = max_images - n_images

            print(f"needed_augmentations: {needed_augmentations}, max images: {max_images}, n_images: {n_images}")
            if needed_augmentations > 0:
                for im in images:
                    if needed_augmentations >= 0:
                        augmented_images = augment_image(im, transfor_max=needed_augmentations)
                        for prefix, aug_img in augmented_images:
                            if needed_augmentations <= 0:
                                break
                            output_dir = Path(aumented_dataset_main_path) / item.name
                            output_dir.mkdir(parents=True, exist_ok=True)
                            output_path = output_dir / f"{im.stem}_{prefix}{im.suffix}"
                            cv.imwrite(str(output_path), aug_img)
                            needed_augmentations -= 1
                    # write the original image to the augmented dataset as well
                    output_dir = Path(aumented_dataset_main_path) / item.name
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / im.name
                    cv.imwrite(str(output_path), cv.imread(str(im)))
                    print(f"Saved augmented image to {output_path}. Remaining augmentations needed: {needed_augmentations}")
                     
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("data_path", help="the image path to augment");

    args = parser.parse_args()

    path = Path(args.data_path)

    if not path.exists():
        print("The provided path does not exist.", file=sys.stderr)
        sys.exit(1)

    if is_image(path):
        images = augment_image(path)
        display_images(images)
    elif parse_folder(path):
        balance_classes(path)
    else:
        print("The provided path is neither a valid image nor a valid dataset folder.", file=sys.stderr)
        sys.exit(1)
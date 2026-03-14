from pathlib import Path
import sys

from .transformation import (
    get_projective_image,
    add_contrast,
    scale_image,
    blur_image,
    rotate_image,
    flip_image,
    display_images
)
import cv2 as cv
from .parse_folder import (
    get_max_image_number, 
    get_min_image_number, 
    is_image
)

def augment_and_save_display_image(path: Path):
        images = augment_image(path)
        # save augmented images to the same directory as the original image
        for i, (title, aug_img) in enumerate(images):
            output_path = path.parent / f"{path.stem}_{title}{path.suffix}"
            cv.imwrite(str(output_path), aug_img)
            print(f"Saved augmented image to {output_path}.")
        images.insert(0, ("Original", cv.imread(str(path))))
        display_images(images)

def augment_image(image_path: Path, transfor_max=6):
    img = cv.imread(str(image_path))
    if img is None:
        print(f"Failed to read the image at {image_path}. Please check the file path and ensure it's a valid image.", file=sys.stderr)
        sys.exit(1)
    images = []
    transformations = [
        ('Flipped', flip_image(img, 1)),
        ('Rotated', rotate_image(img, 45)),
        ('Blurred', blur_image(img, 5)),
        ('Scaled', scale_image(img, 1.3)),
        ('Contrast', add_contrast(img, 1.5)),
        ('Projective', get_projective_image(img))
    ]
    if transfor_max > len(transformations):
        transfor_max = len(transformations)
    for title, transformed in transformations[:transfor_max]:
        images.append((title, transformed))
    return images

def balance_classes(root: Path, save_in_same_folder=False):
    min_images = get_min_image_number(root)
    max_images = get_max_image_number(root)
    number_of_augmentations = 6
    total_images_after_augmentation = min_images * number_of_augmentations

    if total_images_after_augmentation < max_images:
        print("The minimum number of class should not be less than 6 times the maximum number of images in any class.", file=sys.stderr)
        sys.exit(1)

    aumented_dataset_main_path = "augmented_directory"

    for item in root.iterdir():
        if item.is_dir():
            images = [img for img in item.iterdir() if is_image(img)]
            n_images = len(images)
            needed_augmentations = total_images_after_augmentation - n_images

            print(f"needed_augmentations: {needed_augmentations}, max images: {total_images_after_augmentation}, n_images: {n_images}")
            if needed_augmentations > 0:
                for im in images:
                    if needed_augmentations >= 0:
                        augmented_images = augment_image(im, transfor_max=needed_augmentations)
                        for prefix, aug_img in augmented_images:
                            if needed_augmentations <= 0:
                                break
                            if save_in_same_folder:
                                output_path = im.parent / f"{im.stem}_{prefix}{im.suffix}"
                            else:
                                output_dir = Path(aumented_dataset_main_path) / item.name
                                output_dir.mkdir(parents=True, exist_ok=True)
                                output_path = output_dir / f"{im.stem}_{prefix}{im.suffix}"
                            cv.imwrite(str(output_path), aug_img)
                            needed_augmentations -= 1
                    # write the original image to the augmented dataset as well
                    if not save_in_same_folder:
                        output_dir = Path(aumented_dataset_main_path) / item.name
                        output_dir.mkdir(parents=True, exist_ok=True)
                        output_path = output_dir / im.name
                        cv.imwrite(str(output_path), cv.imread(str(im)))
                        print(f"Saved augmented image to {output_path}. Remaining augmentations needed: {needed_augmentations}")
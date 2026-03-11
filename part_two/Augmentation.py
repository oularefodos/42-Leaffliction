import argparse
import os
import sys
import cv2 as cv
import imghdr
import matplotlib.pyplot as plt


def display_images(images, titles):
    plt.figure(figsize=(10, 5))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(cv.cvtColor(images[i], cv.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

def scale_image(image, scale_factor):
    scaled_image = cv.resize(image, None, fx=scale_factor, fy=scale_factor)
    return scaled_image

def blur_image(image, kernel_size):
    blurred_image = cv.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred_image

def rotate_image(image, angle):
    w, h = image.shape[:2]
    center = (h // 2, w // 2)
    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv.warpAffine(image, rotation_matrix, (h, w))
    return rotated_image

def flip_image(image, flip_code):
    flipped_image = cv.flip(image, flip_code)
    return flipped_image

def augment_image(image_path):
    img = cv.imread(image_path)
    base, ext = os.path.splitext(image_path)
    if img is None:
        sys.exit(f"Could not read the image at '{image_path}'")
    
    images = [img]
    titles = ['Original Image']

    # Flip the image horizontally
    flipped_horizontally = flip_image(img, 45)
    images.append(flipped_horizontally)
    titles.append('Flipped Horizontally')

    # rotate the image by 45 degrees
    rotated_45 = rotate_image(img, 45)
    images.append(rotated_45)
    titles.append('Rotated 45 Degrees')

    # blur the image using a Gaussian blur
    blurred = blur_image(img, 5)
    images.append(blurred)
    titles.append('Gaussian Blurred')

    # scale the image by a factor of 0.5
    scaled = scale_image(img, 4)
    images.append(scaled)
    titles.append('Scaled by 0.5')

    display_images(images, titles)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("image_path", help="the image path to augment");

    args = parser.parse_args()

    image_path = os.path.join(args.image_path)

    augment_image(image_path)
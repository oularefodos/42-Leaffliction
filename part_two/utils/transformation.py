import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def display_images(images, global_title="Transformed Images"):
    plt.figure(figsize=(10, 5))
    plt.suptitle(global_title, fontsize=16, fontweight='bold')
    for i, (title, img) in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("transformed_images.png", format='png', dpi=300)

def get_projective_image(image):
    h, w = image.shape[:2]

    src = np.float32([
        [0,     0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0,     h - 1]
    ])

    dst = np.float32([
        [0, 0],
        [(w - 1) * 0.9 , (h - 1) * 0.1],
        [w - 1, (h - 1)],
        [(w - 1) * 0.1, (h - 1) * 0.9 ]
    ])

    M = cv.getPerspectiveTransform(src, dst)
    return cv.warpPerspective(image, M, (w, h))

def add_contrast(image, alpha):
    contrasted_image = cv.convertScaleAbs(image, alpha=alpha, beta=0)
    return contrasted_image

def scale_image(image, scale_factor):
    h, w = image.shape[:2]

    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)

    resized = cv.resize(image, (new_w, new_h))

    cx = new_w // 2
    cy = new_h // 2

    x1 = cx - w // 2
    y1 = cy - h // 2

    return resized[y1:y1+h, x1:x1+w]

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